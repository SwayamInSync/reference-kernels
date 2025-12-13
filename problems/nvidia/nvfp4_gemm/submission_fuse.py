import torch
import sys
import io
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

GEMM_CPP = r"""
#include <torch/extension.h>

torch::Tensor cublaslt_nvfp4_gemm(
    torch::Tensor A,       // [M, K/2, L] float4_e2m1fn_x2 
    torch::Tensor B,       // [N, K/2, L] float4_e2m1fn_x2
    torch::Tensor SFA,     // Scale factors for A (CPU layout - converted on GPU)
    torch::Tensor SFB,     // Scale factors for B (CPU layout - converted on GPU)
    torch::Tensor C,       // [M, N, L] float16 output
    float alpha,
    float beta
);
"""

GEMM_CUDA = r"""
// cuBLASLt NVFP4 GEMM Implementation - Maximum Performance
// 
// Key optimizations:
// 1. Cache matmul descriptors and algorithm - avoid repeated setup overhead
// 2. Pre-allocate and reuse scale factor conversion buffers
// 3. Optimized GPU scale factor conversion kernel with 4-byte coalesced access
//
// Key requirements from cuBLAS documentation:
// 1. A must be transposed (CUBLAS_OP_T), B must be non-transposed (CUBLAS_OP_N) - "TN" format
// 2. Scale mode: CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3
// 3. Compute type: CUBLAS_COMPUTE_32F, Scale type: CUDA_R_32F

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <mutex>
#include <unordered_map>
#include <tuple>

#define checkCublasStatus(status) do { \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        throw std::runtime_error("cuBLAS error: " + std::to_string(status)); \
    } \
} while(0)

#define checkCudaStatus(status) do { \
    if (status != cudaSuccess) { \
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(status))); \
    } \
} while(0)

__host__ __device__ inline int ceilDiv(int a, int b) {
    return (a + b - 1) / b;
}

// ============================================================================
// Cached cuBLAS Resources
// ============================================================================

static cublasLtHandle_t g_ltHandle = nullptr;
static void* g_workspace = nullptr;
static size_t g_workspaceSize = 32 * 1024 * 1024;
static std::once_flag g_init_flag;

// Persistent scale factor buffers
static void* g_sfa_blocked = nullptr;
static void* g_sfb_blocked = nullptr;
static size_t g_sfa_blocked_size = 0;
static size_t g_sfb_blocked_size = 0;

// Cached algorithm and descriptors for specific problem sizes
struct CachedPlan {
    cublasLtMatmulDesc_t operationDesc;
    cublasLtMatrixLayout_t Adesc;
    cublasLtMatrixLayout_t Bdesc;
    cublasLtMatrixLayout_t Cdesc;
    cublasLtMatmulAlgo_t algo;
    bool valid;
};

static std::unordered_map<uint64_t, CachedPlan> g_plan_cache;
static std::mutex g_cache_mutex;

static void initGlobalResources() {
    checkCublasStatus(cublasLtCreate(&g_ltHandle));
    checkCudaStatus(cudaMalloc(&g_workspace, g_workspaceSize));
}

static void ensureScaleBuffers(size_t sfa_size, size_t sfb_size) {
    if (sfa_size > g_sfa_blocked_size) {
        if (g_sfa_blocked) cudaFree(g_sfa_blocked);
        checkCudaStatus(cudaMalloc(&g_sfa_blocked, sfa_size));
        g_sfa_blocked_size = sfa_size;
    }
    if (sfb_size > g_sfb_blocked_size) {
        if (g_sfb_blocked) cudaFree(g_sfb_blocked);
        checkCudaStatus(cudaMalloc(&g_sfb_blocked, sfb_size));
        g_sfb_blocked_size = sfb_size;
    }
}

// Hash function for M,N,K tuple
inline uint64_t makePlanKey(int m, int n, int k) {
    return ((uint64_t)m << 40) | ((uint64_t)n << 20) | (uint64_t)k;
}

// ============================================================================
// GPU Scale Factor Conversion - Fused kernel for both A and B
// ============================================================================
// Converts both SFA and SFB in a single kernel launch to reduce overhead
// Each warp handles one complete 128-row tile column (128 rows x 4 cols = 512 bytes)

__global__ void __launch_bounds__(256, 4) convertBothScaleFactorsKernel(
    const uint8_t* __restrict__ inputA,
    const uint8_t* __restrict__ inputB,
    uint8_t* __restrict__ outputA,
    uint8_t* __restrict__ outputB,
    int rowsA, int rowsB, int cols,
    int n_row_blocks_a, int n_row_blocks_b, int n_col_blocks
) {
    const int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int laneId = threadIdx.x % 32;
    
    // Total tile-columns for A and B combined
    const int total_tile_cols_a = n_row_blocks_a * n_col_blocks;
    const int total_tile_cols_b = n_row_blocks_b * n_col_blocks;
    const int total_tile_cols = total_tile_cols_a + total_tile_cols_b;
    
    if (warpId >= total_tile_cols) return;
    
    // Determine if we're processing A or B
    const bool is_a = (warpId < total_tile_cols_a);
    const int local_warp = is_a ? warpId : (warpId - total_tile_cols_a);
    const int n_row_blocks = is_a ? n_row_blocks_a : n_row_blocks_b;
    const int rows = is_a ? rowsA : rowsB;
    const uint8_t* input = is_a ? inputA : inputB;
    uint8_t* output = is_a ? outputA : outputB;
    
    // Decompose into tile coordinates
    const int tile_row = local_warp / n_col_blocks;
    const int tile_col = local_warp % n_col_blocks;
    
    const int base_row = tile_row * 128;
    const int base_col = tile_col * 4;
    
    // Each thread handles 4 consecutive rows
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        const int local_row = laneId * 4 + r;
        const int global_row = base_row + local_row;
        
        if (global_row < rows) {
            uint32_t val4 = *reinterpret_cast<const uint32_t*>(
                input + global_row * cols + base_col
            );
            
            const int outer = local_row;
            const int tile_offset = (outer % 32) * 16 + (outer / 32) * 4;
            const int output_offset = local_warp * 512 + tile_offset;
            
            *reinterpret_cast<uint32_t*>(output + output_offset) = val4;
        }
    }
}

// Single matrix version for edge cases
__global__ void __launch_bounds__(256, 4) convertScaleFactorsKernelOpt(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    int rows,
    int cols,
    int n_row_blocks,
    int n_col_blocks
) {
    const int warpId = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int laneId = threadIdx.x % 32;
    
    const int total_tile_cols = n_row_blocks * n_col_blocks;
    
    if (warpId >= total_tile_cols) return;
    
    const int tile_row = warpId / n_col_blocks;
    const int tile_col = warpId % n_col_blocks;
    
    const int base_row = tile_row * 128;
    const int base_col = tile_col * 4;
    
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        const int local_row = laneId * 4 + r;
        const int global_row = base_row + local_row;
        
        if (global_row < rows) {
            uint32_t val4 = *reinterpret_cast<const uint32_t*>(
                input + global_row * cols + base_col
            );
            
            const int outer = local_row;
            const int tile_offset = (outer % 32) * 16 + (outer / 32) * 4;
            const int output_offset = warpId * 512 + tile_offset;
            
            *reinterpret_cast<uint32_t*>(output + output_offset) = val4;
        }
    }
}

void convertBothScaleFactors(
    const void* inputA, const void* inputB,
    void* outputA, void* outputB,
    int rowsA, int rowsB, int cols,
    cudaStream_t stream = 0
) {
    int n_row_blocks_a = ceilDiv(rowsA, 128);
    int n_row_blocks_b = ceilDiv(rowsB, 128);
    int n_col_blocks = ceilDiv(cols, 4);
    
    int total_tile_cols = n_row_blocks_a * n_col_blocks + n_row_blocks_b * n_col_blocks;
    int warpsNeeded = total_tile_cols;
    int threadsPerBlock = 256;
    int numBlocks = ceilDiv(warpsNeeded * 32, threadsPerBlock);
    
    convertBothScaleFactorsKernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        static_cast<const uint8_t*>(inputA),
        static_cast<const uint8_t*>(inputB),
        static_cast<uint8_t*>(outputA),
        static_cast<uint8_t*>(outputB),
        rowsA, rowsB, cols,
        n_row_blocks_a, n_row_blocks_b, n_col_blocks
    );
}

void convertScaleFactorsToBlockedFormat(
    const void* input,
    void* output,
    int rows, 
    int cols,
    cudaStream_t stream = 0
) {
    int n_row_blocks = ceilDiv(rows, 128);
    int n_col_blocks = ceilDiv(cols, 4);
    
    int total_tile_cols = n_row_blocks * n_col_blocks;
    int warpsNeeded = total_tile_cols;
    int threadsPerBlock = 256;
    int numBlocks = ceilDiv(warpsNeeded * 32, threadsPerBlock);
    
    convertScaleFactorsKernelOpt<<<numBlocks, threadsPerBlock, 0, stream>>>(
        static_cast<const uint8_t*>(input),
        static_cast<uint8_t*>(output),
        rows, cols,
        n_row_blocks, n_col_blocks
    );
}

// ============================================================================
// Get or create cached plan for given problem size
// ============================================================================
CachedPlan& getOrCreatePlan(int m, int n, int k, const void* sfa_temp, const void* sfb_temp) {
    uint64_t key = makePlanKey(m, n, k);
    
    std::lock_guard<std::mutex> lock(g_cache_mutex);
    
    auto it = g_plan_cache.find(key);
    if (it != g_plan_cache.end() && it->second.valid) {
        return it->second;
    }
    
    // Create new plan
    CachedPlan& plan = g_plan_cache[key];
    plan.valid = false;
    
    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;
    
    // Create operation descriptor
    checkCublasStatus(cublasLtMatmulDescCreate(&plan.operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(plan.operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(plan.operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));
    
    // Set block scaling mode
    cublasLtMatmulMatrixScale_t scaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    checkCublasStatus(cublasLtMatmulDescSetAttribute(plan.operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(plan.operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
    
    // Set temporary scale pointers for heuristic search
    checkCublasStatus(cublasLtMatmulDescSetAttribute(plan.operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sfb_temp, sizeof(sfb_temp)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(plan.operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sfa_temp, sizeof(sfa_temp)));
    
    // Matrix layouts (swapped A and B for TN format)
    checkCublasStatus(cublasLtMatrixLayoutCreate(&plan.Adesc, CUDA_R_4F_E2M1, k, n, k));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&plan.Bdesc, CUDA_R_4F_E2M1, k, m, k));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&plan.Cdesc, CUDA_R_16F, n, m, n));
    
    // Get heuristic and cache the algorithm
    cublasLtMatmulPreference_t preference;
    checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
    checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                           &g_workspaceSize, sizeof(g_workspaceSize)));
    
    cublasLtMatmulHeuristicResult_t heuristicResult = {};
    int returnedResults = 0;
    checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(g_ltHandle, plan.operationDesc, plan.Adesc, plan.Bdesc, 
                                                     plan.Cdesc, plan.Cdesc, preference, 1,
                                                     &heuristicResult, &returnedResults));
    
    cublasLtMatmulPreferenceDestroy(preference);
    
    if (returnedResults == 0) {
        throw std::runtime_error("cuBLASLt: No algorithm found for NVFP4 GEMM");
    }
    
    plan.algo = heuristicResult.algo;
    plan.valid = true;
    
    return plan;
}

// ============================================================================
// Main entry point
// ============================================================================
torch::Tensor cublaslt_nvfp4_gemm(
    torch::Tensor A,       // [M, K/2, L] float4_e2m1fn_x2 
    torch::Tensor B,       // [N, K/2, L] float4_e2m1fn_x2
    torch::Tensor SFA,     // [M, K//16, L] float8_e4m3 - CPU layout
    torch::Tensor SFB,     // [N, K//16, L] float8_e4m3 - CPU layout
    torch::Tensor C,       // [M, N, L] float16 output
    float alpha,
    float beta)
{
    std::call_once(g_init_flag, initGlobalResources);

    const int M = A.size(0);
    const int K = A.size(1) * 2;
    const int L = A.size(2);
    const int N = B.size(0);
    const int sf_k = K / 16;
    
    // Calculate blocked format sizes
    int n_row_blocks_a = ceilDiv(M, 128);
    int n_col_blocks = ceilDiv(sf_k, 4);
    int n_row_blocks_b = ceilDiv(N, 128);
    
    size_t blocked_size_a = n_row_blocks_a * n_col_blocks * 512;
    size_t blocked_size_b = n_row_blocks_b * n_col_blocks * 512;
    
    ensureScaleBuffers(blocked_size_a * L, blocked_size_b * L);

    // Batch strides
    size_t a_batch_stride = M * (K / 2);
    size_t b_batch_stride = N * (K / 2);
    size_t c_batch_stride = M * N;
    size_t sfa_input_stride = M * sf_k;
    size_t sfb_input_stride = N * sf_k;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    // Convert first batch's scale factors to have valid pointers for plan creation
    const void* sfa_input_0 = static_cast<const uint8_t*>(SFA.data_ptr());
    const void* sfb_input_0 = static_cast<const uint8_t*>(SFB.data_ptr());
    void* sfa_output_0 = g_sfa_blocked;
    void* sfb_output_0 = g_sfb_blocked;
    
    // Fused conversion for both A and B scale factors
    convertBothScaleFactors(sfa_input_0, sfb_input_0, sfa_output_0, sfb_output_0, M, N, sf_k, stream);
    
    // Get or create cached plan for this problem size
    CachedPlan& plan = getOrCreatePlan(M, N, K, sfa_output_0, sfb_output_0);

    // Process each batch
    for (int batch = 0; batch < L; ++batch) {
        const void* a_ptr = static_cast<const char*>(A.data_ptr()) + batch * a_batch_stride;
        const void* b_ptr = static_cast<const char*>(B.data_ptr()) + batch * b_batch_stride;
        
        void* sfa_output = static_cast<uint8_t*>(g_sfa_blocked) + batch * blocked_size_a;
        void* sfb_output = static_cast<uint8_t*>(g_sfb_blocked) + batch * blocked_size_b;
        
        // Convert scale factors only for batches > 0 (batch 0 was already converted)
        if (batch > 0) {
            const void* sfa_input = static_cast<const uint8_t*>(SFA.data_ptr()) + batch * sfa_input_stride;
            const void* sfb_input = static_cast<const uint8_t*>(SFB.data_ptr()) + batch * sfb_input_stride;
            convertBothScaleFactors(sfa_input, sfb_input, sfa_output, sfb_output, M, N, sf_k, stream);
        }
        
        __half* c_ptr = reinterpret_cast<__half*>(C.data_ptr<at::Half>()) + batch * c_batch_stride;

        // Set scale pointers for this batch (must be done each time as pointers change)
        checkCublasStatus(cublasLtMatmulDescSetAttribute(plan.operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sfb_output, sizeof(sfb_output)));
        checkCublasStatus(cublasLtMatmulDescSetAttribute(plan.operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sfa_output, sizeof(sfa_output)));

        // Run matmul with cached algorithm
        checkCublasStatus(cublasLtMatmul(g_ltHandle, plan.operationDesc, &alpha, 
                                         b_ptr, plan.Adesc, 
                                         a_ptr, plan.Bdesc, 
                                         &beta, c_ptr, plan.Cdesc, 
                                         c_ptr, plan.Cdesc,
                                         &plan.algo, g_workspace, g_workspaceSize, stream));
    }

    return C;
}

"""

_cublaslt_gemm = None

def _get_kernel():
    """Lazy load and cache the compiled cuBLASLt kernel."""
    global _cublaslt_gemm
    if _cublaslt_gemm is None:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        if sys.stdout is None:
            sys.stdout = io.StringIO()
        if sys.stderr is None:
            sys.stderr = io.StringIO()
        try:
            _cublaslt_gemm = load_inline(
                name="cublaslt_nvfp4_gemm",
                cpp_sources=[GEMM_CPP],
                cuda_sources=[GEMM_CUDA],
                functions=["cublaslt_nvfp4_gemm"],
                extra_cuda_cflags=[
                    "-std=c++17",
                    "-gencode=arch=compute_100a,code=sm_100a",
                    "--ptxas-options=--gpu-name=sm_100a",
                    "-O3",
                    "-w",
                    "--use_fast_math",
                    "-allow-unsupported-compiler",
                ],
                extra_ldflags=["-lcuda", "-lcublasLt", "-lcublas"],
                verbose=False,
            )
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    return _cublaslt_gemm


def compile_kernel():
    """Pre-compile the kernel to exclude compilation time from benchmarks."""
    _get_kernel()


def custom_kernel(data: input_t) -> output_t:
    """Run cuBLASLt GEMM on NVFP4 block-scaled inputs.
    
    Scale factor conversion to cuBLAS tiled format is done on GPU for speed.
    """
    a, b, sfa_cpu, sfb_cpu, sfa_permuted, sfb_permuted, c = data
    kernel = _get_kernel()
    
    # Pass CPU-layout scale factors directly to kernel
    # The CUDA kernel will convert them to blocked format on GPU (very fast!)
    return kernel.cublaslt_nvfp4_gemm(
        a, b, 
        sfa_cpu, sfb_cpu,  # CPU layout - GPU kernel converts them
        c,
        1.0,  # alpha
        0.0   # beta
    )
