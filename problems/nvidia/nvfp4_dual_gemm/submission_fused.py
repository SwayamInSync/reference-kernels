#!POPCORN leaderboard nvfp4_dual_gemm
#!POPCORN gpu NVIDIA

import sys
import os

# Fix for multiprocessing: ensure stdout/stderr are valid for torch.utils.cpp_extension
if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w')
if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w')

import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

# Fused dual GEMM kernel: loads A once, computes A@B1.T and A@B2.T, fuses silu in epilogue
src = """
#include <cudaTypedefs.h>
#include <cuda_fp16.h>

#include <torch/library.h>
#include <ATen/core/Tensor.h>

constexpr int WARP_SIZE = 32;
constexpr int MMA_K = 64;  // 32 bytes

// https://github.com/NVIDIA/cutlass/blob/v4.3.2/include/cute/arch/copy_sm90_desc.hpp#L193-L197
constexpr uint64_t EVICT_FIRST = 0x12F0000000000000;
constexpr uint64_t EVICT_LAST = 0x14F0000000000000;

__device__ inline
constexpr uint64_t desc_encode(uint64_t x) { return (x & 0x3'FFFFULL) >> 4ULL; }

// https://github.com/NVIDIA/cutlass/blob/v4.2.1/include/cute/arch/cluster_sm90.hpp#L180
__device__
uint32_t elect_sync() {
  uint32_t pred = 0;
  asm volatile(
    "{\\n\\t"
    ".reg .pred %%px;\\n\\t"
    "elect.sync _|%%px, %1;\\n\\t"
    "@%%px mov.s32 %0, 1;\\n\\t"
    "}"
    : "+r"(pred)
    : "r"(0xFFFFFFFF)
  );
  return pred;
}

__device__ inline
void mbarrier_init(int mbar_addr, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_addr), "r"(count));
}

// https://github.com/NVIDIA/cutlass/blob/v4.2.1/include/cutlass/arch/barrier.h#L408
__device__
void mbarrier_wait(int mbar_addr, int phase) {
  uint32_t ticks = 0x989680;  // this is optional
  asm volatile(
    "{\\n\\t"
    ".reg .pred P1;\\n\\t"
    "LAB_WAIT:\\n\\t"
    "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\\n\\t"
    "@P1 bra.uni DONE;\\n\\t"
    "bra.uni LAB_WAIT;\\n\\t"
    "DONE:\\n\\t"
    "}"
    :: "r"(mbar_addr), "r"(phase), "r"(ticks)
  );
}

__device__ inline
void tma_gmem2smem(int dst, const void *src, int size, int mbar_addr, uint64_t cache_policy) {
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
              :: "r"(dst), "l"(src), "r"(size), "r"(mbar_addr), "l"(cache_policy));
}

__device__ inline
void tma_3d_gmem2smem(int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr, uint64_t cache_policy) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint "
              "[%0], [%1, {%2, %3, %4}], [%5], %6;"
              :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(mbar_addr), "l"(cache_policy)
              : "memory");
}

__device__ inline
void tcgen05_cp_nvfp4(int taddr, uint64_t s_desc) {
  asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(taddr), "l"(s_desc));
}

// MMA instruction that writes to a specific TMEM offset
__device__ inline
void tcgen05_mma_nvfp4_offset(
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t i_desc,
  int scale_A_tmem,
  int scale_B_tmem,
  int enable_input_d,
  int d_tmem_offset
) {
  asm volatile(
    "{\\n\\t"
    ".reg .pred p;\\n\\t"
    "setp.ne.b32 p, %6, 0;\\n\\t"
    "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%4], [%5], p;\\n\\t"
    "}"
    :: "r"(d_tmem_offset), "l"(a_desc), "l"(b_desc), "r"(i_desc),
       "r"(scale_A_tmem), "r"(scale_B_tmem), "r"(enable_input_d)
  );
}

// MMA with collector::a::fill - loads A into collector buffer for reuse
__device__ inline
void tcgen05_mma_nvfp4_fill(
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t i_desc,
  int scale_A_tmem,
  int scale_B_tmem,
  int enable_input_d,
  int d_tmem_offset
) {
  asm volatile(
    "{\\n\\t"
    ".reg .pred p;\\n\\t"
    "setp.ne.b32 p, %6, 0;\\n\\t"
    "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::fill [%0], %1, %2, %3, [%4], [%5], p;\\n\\t"
    "}"
    :: "r"(d_tmem_offset), "l"(a_desc), "l"(b_desc), "r"(i_desc),
       "r"(scale_A_tmem), "r"(scale_B_tmem), "r"(enable_input_d)
  );
}

// MMA with collector::a::lastuse - reuses A from collector buffer (no memory read!)
__device__ inline
void tcgen05_mma_nvfp4_lastuse(
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t i_desc,
  int scale_A_tmem,
  int scale_B_tmem,
  int enable_input_d,
  int d_tmem_offset
) {
  asm volatile(
    "{\\n\\t"
    ".reg .pred p;\\n\\t"
    "setp.ne.b32 p, %6, 0;\\n\\t"
    "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::lastuse [%0], %1, %2, %3, [%4], [%5], p;\\n\\t"
    "}"
    :: "r"(d_tmem_offset), "l"(a_desc), "l"(b_desc), "r"(i_desc),
       "r"(scale_A_tmem), "r"(scale_B_tmem), "r"(enable_input_d)
  );
}

// Standard MMA (d_tmem = 0)
__device__ inline
void tcgen05_mma_nvfp4(
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t i_desc,
  int scale_A_tmem,
  int scale_B_tmem,
  int enable_input_d
) {
  const int d_tmem = 0;  // assume
  asm volatile(
    "{\\n\\t"
    ".reg .pred p;\\n\\t"
    "setp.ne.b32 p, %6, 0;\\n\\t"
    "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%4], [%5], p;\\n\\t"
    "}"
    :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(i_desc),
       "r"(scale_A_tmem), "r"(scale_B_tmem), "r"(enable_input_d)
  );
}

// see https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html
struct SHAPE {
  static constexpr char _32x32b[]  = ".32x32b";   // 32x1 tile for each warp
  static constexpr char _16x256b[] = ".16x256b";  // 16x8 tile
};

struct NUM {
  static constexpr char x4[]  = ".x4";
  static constexpr char x8[]  = ".x8";
  static constexpr char x16[] = ".x16";
  static constexpr char x32[] = ".x32";
  static constexpr char x64[] = ".x64";
  static constexpr char x128[] = ".x128";
};

template <const char *SHAPE, const char *NUM>
__device__ inline
void tcgen05_ld_16regs(float *tmp, int row, int col) {
  asm volatile("tcgen05.ld.sync.aligned%17%18.b32 "
              "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
              "  %8,  %9, %10, %11, %12, %13, %14, %15}, [%16];"
              : "=f"(tmp[ 0]), "=f"(tmp[ 1]), "=f"(tmp[ 2]), "=f"(tmp[ 3]), "=f"(tmp[ 4]), "=f"(tmp[ 5]), "=f"(tmp[ 6]), "=f"(tmp[ 7]),
                "=f"(tmp[ 8]), "=f"(tmp[ 9]), "=f"(tmp[10]), "=f"(tmp[11]), "=f"(tmp[12]), "=f"(tmp[13]), "=f"(tmp[14]), "=f"(tmp[15])
              : "r"((row << 16) | col), "C"(SHAPE), "C"(NUM));
}

template <const char *SHAPE, const char *NUM>
__device__ inline
void tcgen05_ld_32regs(float *tmp, int row, int col) {
  asm volatile("tcgen05.ld.sync.aligned%33%34.b32 "
              "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
              "  %8,  %9, %10, %11, %12, %13, %14, %15, "
              " %16, %17, %18, %19, %20, %21, %22, %23, "
              " %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
              : "=f"(tmp[ 0]), "=f"(tmp[ 1]), "=f"(tmp[ 2]), "=f"(tmp[ 3]), "=f"(tmp[ 4]), "=f"(tmp[ 5]), "=f"(tmp[ 6]), "=f"(tmp[ 7]),
                "=f"(tmp[ 8]), "=f"(tmp[ 9]), "=f"(tmp[10]), "=f"(tmp[11]), "=f"(tmp[12]), "=f"(tmp[13]), "=f"(tmp[14]), "=f"(tmp[15]),
                "=f"(tmp[16]), "=f"(tmp[17]), "=f"(tmp[18]), "=f"(tmp[19]), "=f"(tmp[20]), "=f"(tmp[21]), "=f"(tmp[22]), "=f"(tmp[23]),
                "=f"(tmp[24]), "=f"(tmp[25]), "=f"(tmp[26]), "=f"(tmp[27]), "=f"(tmp[28]), "=f"(tmp[29]), "=f"(tmp[30]), "=f"(tmp[31])
              : "r"((row << 16) | col), "C"(SHAPE), "C"(NUM));
}

template <const char *SHAPE, const char *NUM>
__device__ inline
void tcgen05_ld_64regs(float *tmp, int row, int col) {
  asm volatile("tcgen05.ld.sync.aligned%65%66.b32 "
              "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
              "  %8,  %9, %10, %11, %12, %13, %14, %15, "
              " %16, %17, %18, %19, %20, %21, %22, %23, "
              " %24, %25, %26, %27, %28, %29, %30, %31, "
              " %32, %33, %34, %35, %36, %37, %38, %39, "
              " %40, %41, %42, %43, %44, %45, %46, %47, "
              " %48, %49, %50, %51, %52, %53, %54, %55, "
              " %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
              : "=f"(tmp[ 0]), "=f"(tmp[ 1]), "=f"(tmp[ 2]), "=f"(tmp[ 3]), "=f"(tmp[ 4]), "=f"(tmp[ 5]), "=f"(tmp[ 6]), "=f"(tmp[ 7]),
                "=f"(tmp[ 8]), "=f"(tmp[ 9]), "=f"(tmp[10]), "=f"(tmp[11]), "=f"(tmp[12]), "=f"(tmp[13]), "=f"(tmp[14]), "=f"(tmp[15]),
                "=f"(tmp[16]), "=f"(tmp[17]), "=f"(tmp[18]), "=f"(tmp[19]), "=f"(tmp[20]), "=f"(tmp[21]), "=f"(tmp[22]), "=f"(tmp[23]),
                "=f"(tmp[24]), "=f"(tmp[25]), "=f"(tmp[26]), "=f"(tmp[27]), "=f"(tmp[28]), "=f"(tmp[29]), "=f"(tmp[30]), "=f"(tmp[31]),
                "=f"(tmp[32]), "=f"(tmp[33]), "=f"(tmp[34]), "=f"(tmp[35]), "=f"(tmp[36]), "=f"(tmp[37]), "=f"(tmp[38]), "=f"(tmp[39]),
                "=f"(tmp[40]), "=f"(tmp[41]), "=f"(tmp[42]), "=f"(tmp[43]), "=f"(tmp[44]), "=f"(tmp[45]), "=f"(tmp[46]), "=f"(tmp[47]),
                "=f"(tmp[48]), "=f"(tmp[49]), "=f"(tmp[50]), "=f"(tmp[51]), "=f"(tmp[52]), "=f"(tmp[53]), "=f"(tmp[54]), "=f"(tmp[55]),
                "=f"(tmp[56]), "=f"(tmp[57]), "=f"(tmp[58]), "=f"(tmp[59]), "=f"(tmp[60]), "=f"(tmp[61]), "=f"(tmp[62]), "=f"(tmp[63])
              : "r"((row << 16) | col), "C"(SHAPE), "C"(NUM));
}

template <const char *SHAPE, const char *NUM>
__device__ inline
void tcgen05_ld_128regs(float *tmp, int row, int col) {
  asm volatile("tcgen05.ld.sync.aligned%129%130.b32 "
              "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
              "  %8,  %9, %10, %11, %12, %13, %14, %15, "
              " %16, %17, %18, %19, %20, %21, %22, %23, "
              " %24, %25, %26, %27, %28, %29, %30, %31, "
              " %32, %33, %34, %35, %36, %37, %38, %39, "
              " %40, %41, %42, %43, %44, %45, %46, %47, "
              " %48, %49, %50, %51, %52, %53, %54, %55, "
              " %56, %57, %58, %59, %60, %61, %62, %63, "
              " %64, %65, %66, %67, %68, %69, %70, %71, "
              " %72, %73, %74, %75, %76, %77, %78, %79, "
              " %80, %81, %82, %83, %84, %85, %86, %87, "
              " %88, %89, %90, %91, %92, %93, %94, %95, "
              " %96, %97, %98, %99,%100,%101,%102,%103, "
              "%104,%105,%106,%107,%108,%109,%110,%111, "
              "%112,%113,%114,%115,%116,%117,%118,%119, "
              "%120,%121,%122,%123,%124,%125,%126,%127}, [%128];"
              : "=f"(tmp[ 0]), "=f"(tmp[ 1]), "=f"(tmp[ 2]), "=f"(tmp[ 3]), "=f"(tmp[ 4]), "=f"(tmp[ 5]), "=f"(tmp[ 6]), "=f"(tmp[ 7]),
                "=f"(tmp[ 8]), "=f"(tmp[ 9]), "=f"(tmp[10]), "=f"(tmp[11]), "=f"(tmp[12]), "=f"(tmp[13]), "=f"(tmp[14]), "=f"(tmp[15]),
                "=f"(tmp[16]), "=f"(tmp[17]), "=f"(tmp[18]), "=f"(tmp[19]), "=f"(tmp[20]), "=f"(tmp[21]), "=f"(tmp[22]), "=f"(tmp[23]),
                "=f"(tmp[24]), "=f"(tmp[25]), "=f"(tmp[26]), "=f"(tmp[27]), "=f"(tmp[28]), "=f"(tmp[29]), "=f"(tmp[30]), "=f"(tmp[31]),
                "=f"(tmp[32]), "=f"(tmp[33]), "=f"(tmp[34]), "=f"(tmp[35]), "=f"(tmp[36]), "=f"(tmp[37]), "=f"(tmp[38]), "=f"(tmp[39]),
                "=f"(tmp[40]), "=f"(tmp[41]), "=f"(tmp[42]), "=f"(tmp[43]), "=f"(tmp[44]), "=f"(tmp[45]), "=f"(tmp[46]), "=f"(tmp[47]),
                "=f"(tmp[48]), "=f"(tmp[49]), "=f"(tmp[50]), "=f"(tmp[51]), "=f"(tmp[52]), "=f"(tmp[53]), "=f"(tmp[54]), "=f"(tmp[55]),
                "=f"(tmp[56]), "=f"(tmp[57]), "=f"(tmp[58]), "=f"(tmp[59]), "=f"(tmp[60]), "=f"(tmp[61]), "=f"(tmp[62]), "=f"(tmp[63]),
                "=f"(tmp[64]), "=f"(tmp[65]), "=f"(tmp[66]), "=f"(tmp[67]), "=f"(tmp[68]), "=f"(tmp[69]), "=f"(tmp[70]), "=f"(tmp[71]),
                "=f"(tmp[72]), "=f"(tmp[73]), "=f"(tmp[74]), "=f"(tmp[75]), "=f"(tmp[76]), "=f"(tmp[77]), "=f"(tmp[78]), "=f"(tmp[79]),
                "=f"(tmp[80]), "=f"(tmp[81]), "=f"(tmp[82]), "=f"(tmp[83]), "=f"(tmp[84]), "=f"(tmp[85]), "=f"(tmp[86]), "=f"(tmp[87]),
                "=f"(tmp[88]), "=f"(tmp[89]), "=f"(tmp[90]), "=f"(tmp[91]), "=f"(tmp[92]), "=f"(tmp[93]), "=f"(tmp[94]), "=f"(tmp[95]),
                "=f"(tmp[96]), "=f"(tmp[97]), "=f"(tmp[98]), "=f"(tmp[99]), "=f"(tmp[100]),"=f"(tmp[101]),"=f"(tmp[102]),"=f"(tmp[103]),
                "=f"(tmp[104]),"=f"(tmp[105]),"=f"(tmp[106]),"=f"(tmp[107]),"=f"(tmp[108]),"=f"(tmp[109]),"=f"(tmp[110]),"=f"(tmp[111]),
                "=f"(tmp[112]),"=f"(tmp[113]),"=f"(tmp[114]),"=f"(tmp[115]),"=f"(tmp[116]),"=f"(tmp[117]),"=f"(tmp[118]),"=f"(tmp[119]),
                "=f"(tmp[120]),"=f"(tmp[121]),"=f"(tmp[122]),"=f"(tmp[123]),"=f"(tmp[124]),"=f"(tmp[125]),"=f"(tmp[126]),"=f"(tmp[127])
              : "r"((row << 16) | col), "C"(SHAPE), "C"(NUM));
}

__device__ inline void tcgen05_ld_32x32bx32(float *tmp, int row, int col) { tcgen05_ld_32regs<SHAPE::_32x32b, NUM::x32>(tmp, row, col); }
__device__ inline void tcgen05_ld_32x32bx64(float *tmp, int row, int col) { tcgen05_ld_64regs<SHAPE::_32x32b, NUM::x64>(tmp, row, col); }
__device__ inline void tcgen05_ld_32x32bx128(float *tmp, int row, int col) { tcgen05_ld_128regs<SHAPE::_32x32b, NUM::x128>(tmp, row, col); }

__device__ inline void tcgen05_ld_16x256bx4(float *tmp, int row, int col) { tcgen05_ld_16regs<SHAPE::_16x256b, NUM::x4>(tmp, row, col); }
__device__ inline void tcgen05_ld_16x256bx8(float *tmp, int row, int col) { tcgen05_ld_32regs<SHAPE::_16x256b, NUM::x8>(tmp, row, col); }
__device__ inline void tcgen05_ld_16x256bx16(float *tmp, int row, int col) { tcgen05_ld_64regs<SHAPE::_16x256b, NUM::x16>(tmp, row, col); }

void check_cu(CUresult err) {
  if (err == CUDA_SUCCESS) return;
  const char *error_msg_ptr;
  if (cuGetErrorString(err, &error_msg_ptr) != CUDA_SUCCESS)
    error_msg_ptr = "unable to get error string";
  TORCH_CHECK(false, "cuTensorMapEncodeTiled error: ", error_msg_ptr);
}

void init_AB_tmap(
  CUtensorMap *tmap,
  const char *ptr,
  uint64_t global_height, uint64_t global_width,
  uint32_t shared_height, uint32_t shared_width
) {
  constexpr uint32_t rank = 3;
  uint64_t globalDim[rank]       = {256, global_height, global_width / 256};
  uint64_t globalStrides[rank-1] = {global_width / 2, 128};  // in bytes
  uint32_t boxDim[rank]          = {256, shared_height, shared_width / 256};
  uint32_t elementStrides[rank]  = {1, 1, 1};

  auto err = cuTensorMapEncodeTiled(
    tmap,
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
    rank,
    (void *)ptr,
    globalDim,
    globalStrides,
    boxDim,
    elementStrides,
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
  );
  check_cu(err);
}

// ====================================================================================
// FUSED DUAL GEMM KERNEL
// This kernel computes C = silu(A @ B1.T) * (A @ B2.T) in a single pass
// Key optimizations:
// - A is loaded once and reused for both B1 and B2 computations
// - Both B1 and B2 are loaded into shared memory (doubled B size)
// - Two separate TMEM regions for accumulators D1 and D2
// - silu fusion happens in the epilogue
// ====================================================================================
template <
  int K,
  int BLOCK_M,
  int BLOCK_N,
  int BLOCK_K,
  int NUM_STAGES
>
__global__
__launch_bounds__(BLOCK_M + 2 * WARP_SIZE)
void fused_dual_gemm_kernel(
  const __grid_constant__ CUtensorMap A_tmap,
  const __grid_constant__ CUtensorMap B1_tmap,
  const __grid_constant__ CUtensorMap B2_tmap,
  const char *SFA_ptr,
  const char *SFB1_ptr,
  const char *SFB2_ptr,
  half *C_ptr,
  int M, int N
) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;

  const int lane_id = tid % WARP_SIZE;
  const int warp_id = tid / WARP_SIZE;

  const int grid_m = M / BLOCK_M;
  const int grid_n = N / BLOCK_N;
  const int bid_m = bid / grid_n;
  const int bid_n = bid % grid_n;

  const int off_m = bid_m * BLOCK_M;
  const int off_n = bid_n * BLOCK_N;

  constexpr int NUM_WARPS = BLOCK_M / WARP_SIZE + 2;

  // Shared memory layout for fused dual GEMM:
  // Per stage: A | B1 | B2 | SFA | SFB1 | SFB2
  extern __shared__ __align__(1024) char smem_ptr[];
  const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));
  constexpr int A_size = BLOCK_M * BLOCK_K / 2;
  constexpr int B_size = BLOCK_N * BLOCK_K / 2;  // per B matrix
  constexpr int SFA_size = 128 * BLOCK_K / 16;
  constexpr int SFB_size = 128 * BLOCK_K / 16;  // per B matrix
  constexpr int STAGE_SIZE = A_size + B_size * 2 + SFA_size + SFB_size * 2;

  // Mbarriers: NUM_STAGES for TMA, NUM_STAGES for MMA, 1 for mainloop
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ int64_t mbars[NUM_STAGES * 2 + 1];
  const int tma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mbars));
  const int mma_mbar_addr = tma_mbar_addr + NUM_STAGES * 8;
  const int mainloop_mbar_addr = mma_mbar_addr + NUM_STAGES * 8;

  // TMEM layout for fused dual GEMM:
  // D1 accumulator: columns 0 to BLOCK_N*2-1 (128 columns for BLOCK_N=64)
  // D2 accumulator: columns BLOCK_N*2 to BLOCK_N*4-1
  // SFA: after D2
  // SFB1, SFB2: after SFA
  constexpr int D1_tmem = 0;
  constexpr int D2_tmem = BLOCK_N * 2;  // Second accumulator starts here
  constexpr int SFA_tmem = BLOCK_N * 4;  // Scale factors for A
  constexpr int SFB1_tmem = SFA_tmem + 4 * (BLOCK_K / MMA_K);
  constexpr int SFB2_tmem = SFB1_tmem + 4 * (BLOCK_K / MMA_K);

  if (warp_id == 0 && elect_sync()) {
    for (int i = 0; i < NUM_STAGES * 2 + 1; i++)
      mbarrier_init(tma_mbar_addr + i * 8, 1);
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  else if (warp_id == 1) {
    // Allocate TMEM: 2x BLOCK_N*2 for two accumulators = BLOCK_N * 4
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(smem), "r"(BLOCK_N * 4));
  }
  __syncthreads();

  constexpr int num_iters = K / BLOCK_K;

  // Warp-specialization
  if (warp_id == NUM_WARPS - 2 && elect_sync()) {
    // TMA warp - loads A, B1, B2, SFA, SFB1, SFB2
    uint64_t cache_A = EVICT_LAST;
    uint64_t cache_B = EVICT_FIRST;

    auto issue_tma = [&](int iter_k, int stage_id) {
      const int mbar_addr = tma_mbar_addr + stage_id * 8;
      const int A_smem = smem + stage_id * STAGE_SIZE;
      const int B1_smem = A_smem + A_size;
      const int B2_smem = B1_smem + B_size;
      const int SFA_smem = B2_smem + B_size;
      const int SFB1_smem = SFA_smem + SFA_size;
      const int SFB2_smem = SFB1_smem + SFB_size;

      const int off_k = iter_k * BLOCK_K;

      // Load A (once per iteration)
      tma_3d_gmem2smem(A_smem, &A_tmap, 0, off_m, off_k / 256, mbar_addr, cache_A);
      // Load B1 and B2
      tma_3d_gmem2smem(B1_smem, &B1_tmap, 0, off_n, off_k / 256, mbar_addr, cache_B);
      tma_3d_gmem2smem(B2_smem, &B2_tmap, 0, off_n, off_k / 256, mbar_addr, cache_B);

      // Scale factors
      const int rest_k = K / 16 / 4;
      const char *SFA_src = SFA_ptr + ((off_m / 128) * rest_k + off_k / (16 * 4)) * 512;
      const char *SFB1_src = SFB1_ptr + ((off_n / 128) * rest_k + off_k / (16 * 4)) * 512;
      const char *SFB2_src = SFB2_ptr + ((off_n / 128) * rest_k + off_k / (16 * 4)) * 512;
      tma_gmem2smem(SFA_smem, SFA_src, SFA_size, mbar_addr, cache_A);
      tma_gmem2smem(SFB1_smem, SFB1_src, SFB_size, mbar_addr, cache_B);
      tma_gmem2smem(SFB2_smem, SFB2_src, SFB_size, mbar_addr, cache_B);

      // Signal TMA done - total bytes transferred
      constexpr int total_tx = A_size + B_size * 2 + SFA_size + SFB_size * 2;
      asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                  :: "r"(mbar_addr), "r"(total_tx) : "memory");
    };

    // Issue initial TMA without waiting for MMA
    for (int iter_k = 0; iter_k < NUM_STAGES; iter_k++)
      issue_tma(iter_k, iter_k);

    for (int iter_k = NUM_STAGES; iter_k < num_iters; iter_k++) {
      const int stage_id = iter_k % NUM_STAGES;
      const int mma_phase = (iter_k / NUM_STAGES - 1) % 2;
      mbarrier_wait(mma_mbar_addr + stage_id * 8, mma_phase);
      issue_tma(iter_k, stage_id);
    }
  }
  else if (warp_id == NUM_WARPS - 1 && elect_sync()) {
    // MMA warp - issues MMA for both D1 (A@B1) and D2 (A@B2)
    constexpr int MMA_N = BLOCK_N;
    constexpr int MMA_M = 128;
    constexpr uint32_t i_desc = (1U << 7U)   // atype=E2M1
                              | (1U << 10U)  // btype=E2M1
                              | ((uint32_t)MMA_N >> 3U << 17U)
                              | ((uint32_t)MMA_M >> 7U << 27U)
                              ;

    for (int iter_k = 0; iter_k < num_iters; iter_k++) {
      const int stage_id = iter_k % NUM_STAGES;
      const int tma_phase = (iter_k / NUM_STAGES) % 2;
      mbarrier_wait(tma_mbar_addr + stage_id * 8, tma_phase);

      const int A_smem = smem + stage_id * STAGE_SIZE;
      const int B1_smem = A_smem + A_size;
      const int B2_smem = B1_smem + B_size;
      const int SFA_smem = B2_smem + B_size;
      const int SFB1_smem = SFA_smem + SFA_size;
      const int SFB2_smem = SFB1_smem + SFB_size;

      // Shared memory descriptors
      auto make_desc_AB = [](int addr) -> uint64_t {
        const int SBO = 8 * 128;
        return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
      };
      auto make_desc_SF = [](int addr) -> uint64_t {
        const int SBO = 8 * 16;
        return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL);
      };

      constexpr uint64_t SF_desc = make_desc_SF(0);
      const uint64_t SFA_desc = SF_desc + ((uint64_t)SFA_smem >> 4ULL);
      const uint64_t SFB1_desc = SF_desc + ((uint64_t)SFB1_smem >> 4ULL);
      const uint64_t SFB2_desc = SF_desc + ((uint64_t)SFB2_smem >> 4ULL);

      // Copy scale factors to TMEM
      for (int k = 0; k < BLOCK_K / MMA_K; k++) {
        uint64_t sfa_desc = SFA_desc + (uint64_t)k * (512ULL >> 4ULL);
        uint64_t sfb1_desc = SFB1_desc + (uint64_t)k * (512ULL >> 4ULL);
        uint64_t sfb2_desc = SFB2_desc + (uint64_t)k * (512ULL >> 4ULL);
        tcgen05_cp_nvfp4(SFA_tmem + k * 4, sfa_desc);
        tcgen05_cp_nvfp4(SFB1_tmem + k * 4, sfb1_desc);
        tcgen05_cp_nvfp4(SFB2_tmem + k * 4, sfb2_desc);
      }

      // Issue MMA for both D1 and D2 using A collector buffer
      // D1 MMA fills the collector with A, D2 MMA reuses A from collector
      for (int k1 = 0; k1 < BLOCK_K / 256; k1++) {
        for (int k2 = 0; k2 < 256 / MMA_K; k2++) {
          uint64_t a_desc = make_desc_AB(A_smem + k1 * BLOCK_M * 128 + k2 * 32);
          uint64_t b1_desc = make_desc_AB(B1_smem + k1 * BLOCK_N * 128 + k2 * 32);
          uint64_t b2_desc = make_desc_AB(B2_smem + k1 * BLOCK_N * 128 + k2 * 32);

          int k_sf = k1 * 4 + k2;
          const int scale_A_tmem = SFA_tmem + k_sf * 4 + (bid_m % (128 / BLOCK_M)) * (BLOCK_M / 32);
          const int scale_B1_tmem = SFB1_tmem + k_sf * 4 + (bid_n % (128 / BLOCK_N)) * (BLOCK_N / 32);
          const int scale_B2_tmem = SFB2_tmem + k_sf * 4 + (bid_n % (128 / BLOCK_N)) * (BLOCK_N / 32);

          const int enable_input_d = (k1 == 0 && k2 == 0) ? iter_k : 1;

          // MMA for D1 = A @ B1.T - FILL collector buffer with A
          tcgen05_mma_nvfp4_fill(a_desc, b1_desc, i_desc, scale_A_tmem, scale_B1_tmem, enable_input_d, D1_tmem);
          // MMA for D2 = A @ B2.T - REUSE A from collector buffer (lastuse)
          tcgen05_mma_nvfp4_lastuse(a_desc, b2_desc, i_desc, scale_A_tmem, scale_B2_tmem, enable_input_d, D2_tmem);
        }
      }

      asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                  :: "r"(mma_mbar_addr + stage_id * 8) : "memory");
    }

    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                :: "r"(mainloop_mbar_addr) : "memory");
  }
  else if (tid < BLOCK_M) {
    // Epilogue warps - fuse silu: C = silu(D1) * D2

    mbarrier_wait(mainloop_mbar_addr, 0);
    asm volatile("tcgen05.fence::after_thread_sync;");

    // N-major epilogue with fused silu
    for (int m = 0; m < 32 / 16; m++) {
      // Load D1 and D2 in parallel for better throughput
      float tmp1[BLOCK_N / 2];
      float tmp2[BLOCK_N / 2];
      
      if constexpr (BLOCK_N == 128) {
        tcgen05_ld_16x256bx16(tmp1, warp_id * 32 + m * 16, D1_tmem);
        tcgen05_ld_16x256bx16(tmp2, warp_id * 32 + m * 16, D2_tmem);
        asm volatile("tcgen05.wait::ld.sync.aligned;");
      }
      if constexpr (BLOCK_N == 64) {
        tcgen05_ld_16x256bx8(tmp1, warp_id * 32 + m * 16, D1_tmem);
        tcgen05_ld_16x256bx8(tmp2, warp_id * 32 + m * 16, D2_tmem);
        asm volatile("tcgen05.wait::ld.sync.aligned;");
      }
      if constexpr (BLOCK_N == 32) {
        tcgen05_ld_16x256bx4(tmp1, warp_id * 32 + m * 16, D1_tmem);
        tcgen05_ld_16x256bx4(tmp2, warp_id * 32 + m * 16, D2_tmem);
        asm volatile("tcgen05.wait::ld.sync.aligned;");
      }

      // Apply silu and multiply, then store to fp16
      for (int i = 0; i < BLOCK_N / 8; i++) {
        const int row = off_m + warp_id * 32 + m * 16 + lane_id / 4;
        const int col = off_n + i * 8 + (lane_id % 4) * 2;

        // Compute silu(tmp1) * tmp2 for each element pair
        float x0 = tmp1[i * 4 + 0], x1 = tmp1[i * 4 + 1], x2 = tmp1[i * 4 + 2], x3 = tmp1[i * 4 + 3];
        float silu0 = x0 / (1.0f + expf(-x0));
        float silu1 = x1 / (1.0f + expf(-x1));
        float silu2 = x2 / (1.0f + expf(-x2));
        float silu3 = x3 / (1.0f + expf(-x3));

        float out0 = silu0 * tmp2[i * 4 + 0];
        float out1 = silu1 * tmp2[i * 4 + 1];
        float out2 = silu2 * tmp2[i * 4 + 2];
        float out3 = silu3 * tmp2[i * 4 + 3];

        // Store as half2
        reinterpret_cast<half2 *>(C_ptr + (row + 0) * N + col)[0] = __float22half2_rn({out0, out1});
        reinterpret_cast<half2 *>(C_ptr + (row + 8) * N + col)[0] = __float22half2_rn({out2, out3});
      }
    }

    asm volatile("bar.sync 1, %0;" :: "r"(BLOCK_M) : "memory");
    if (warp_id == 0)
      asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(0), "r"(BLOCK_N * 4));
  }
}

template <
  int K,
  int BLOCK_M,
  int BLOCK_N,
  int BLOCK_K,
  int NUM_STAGES
>
void fused_dual_gemm_launch(
  const at::Tensor& A,
  const at::Tensor& B1,
  const at::Tensor& B2,
  const at::Tensor& SFA,
  const at::Tensor& SFB1,
  const at::Tensor& SFB2,
        at::Tensor& C
) {
  static_assert(BLOCK_K % 256 == 0);

  const int M = A.size(0);
  const int N = B1.size(0);

  auto A_ptr   = reinterpret_cast<const char *>(A.data_ptr());
  auto B1_ptr  = reinterpret_cast<const char *>(B1.data_ptr());
  auto B2_ptr  = reinterpret_cast<const char *>(B2.data_ptr());
  auto SFA_ptr = reinterpret_cast<const char *>(SFA.data_ptr());
  auto SFB1_ptr = reinterpret_cast<const char *>(SFB1.data_ptr());
  auto SFB2_ptr = reinterpret_cast<const char *>(SFB2.data_ptr());
  auto C_ptr   = reinterpret_cast<half *>(C.data_ptr());

  CUtensorMap A_tmap, B1_tmap, B2_tmap;
  init_AB_tmap(&A_tmap, A_ptr, M, K, BLOCK_M, BLOCK_K);
  init_AB_tmap(&B1_tmap, B1_ptr, N, K, BLOCK_N, BLOCK_K);
  init_AB_tmap(&B2_tmap, B2_ptr, N, K, BLOCK_N, BLOCK_K);

  dim3 grid((M / BLOCK_M) * (N / BLOCK_N));
  int tb_size = BLOCK_M + 2 * WARP_SIZE;
  
  // Shared memory: per stage = A + 2*B + SFA + 2*SFB
  int A_size = BLOCK_M * BLOCK_K / 2;
  int B_size = BLOCK_N * BLOCK_K / 2;
  int SFA_size = 128 * BLOCK_K / 16;
  int SFB_size = 128 * BLOCK_K / 16;
  int smem_size = (A_size + B_size * 2 + SFA_size + SFB_size * 2) * NUM_STAGES;

  auto this_kernel = fused_dual_gemm_kernel<K, BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES>;
  if (smem_size > 48'000)
    cudaFuncSetAttribute(this_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  
  this_kernel<<<grid, tb_size, smem_size>>>(A_tmap, B1_tmap, B2_tmap, SFA_ptr, SFB1_ptr, SFB2_ptr, C_ptr, M, N);
}

at::Tensor dual_gemm(
  const at::Tensor& A,
  const at::Tensor& B1,
  const at::Tensor& B2,
  const at::Tensor& SFA,
  const at::Tensor& SFB1,
  const at::Tensor& SFB2,
        at::Tensor& C
) {
  const int K = A.size(1) * 2;
  const int M = A.size(0);
  const int N = B1.size(0);

#define LAUNCH(K_, N_min, BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES) \
  else if (K == K_ && N >= N_min && (N % BLOCK_N) == 0) { \
    fused_dual_gemm_launch<K_, BLOCK_M, BLOCK_N, BLOCK_K, NUM_STAGES>(A, B1, B2, SFA, SFB1, SFB2, C); \
  }

  if (false) {}
  // Benchmark cases: K=7168, N>=3072 - 5 stages for better latency hiding
  LAUNCH( 7168, 3072, 128, 64, 256, 5)
  // Benchmark cases: K=4096, N>=3072 - 5 stages  
  LAUNCH( 4096, 3072, 128, 64, 256, 5)
  // Fallback for other cases
  LAUNCH(16384, 64, 128, 64, 256, 4)
  LAUNCH( 7168, 64, 128, 64, 256, 4)
  LAUNCH( 4096, 64, 128, 64, 256, 4)
  LAUNCH( 2048, 64, 128, 64, 256, 4)
  LAUNCH( 1536, 64, 128, 64, 256, 4)
  LAUNCH( 2304, 64, 128, 64, 256, 4)
  LAUNCH(  256, 64, 128, 64, 256, 4)
  LAUNCH(  512, 64, 128, 64, 256, 4)

#undef LAUNCH

  return C;
}

TORCH_LIBRARY(dual_gemm_module, m) {
  m.def("dual_gemm(Tensor A, Tensor B1, Tensor B2, Tensor SFA, Tensor SFB1, Tensor SFB2, Tensor(a!) C) -> Tensor");
  m.impl("dual_gemm", &dual_gemm);
}
"""

for i, src_code in enumerate([src]):
    load_inline(
        f"dual_gemm_{i}",
        cpp_sources="",
        cuda_sources=src_code,
        verbose=True,
        is_python_module=False,
        no_implicit_headers=True,
        extra_cuda_cflags=[
            "-O3",
            "-gencode=arch=compute_100a,code=sm_100a",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "--relocatable-device-code=false",
            "-lineinfo",
            "-Xptxas=-v",
        ],
        extra_ldflags=["-lcuda"],
    )

dual_gemm_op = torch.ops.dual_gemm_module.dual_gemm


def compile_kernel():
    """Pre-compile the kernel to exclude compilation time from benchmarks."""
    pass


def custom_kernel(data: input_t) -> output_t:
    """
    Fused Dual GEMM with SiLU activation: C = silu(A @ B1.T) * (A @ B2.T)
    """
    a, b1, b2, sfa, sfb1, sfb2, sfa_permuted, sfb1_permuted, sfb2_permuted, c = data
    
    return dual_gemm_op(a, b1, b2, sfa_permuted, sfb1_permuted, sfb2_permuted, c)
