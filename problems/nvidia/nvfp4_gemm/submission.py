import torch
import sys
import io
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

GEMM_CPP = r"""
#include <torch/extension.h>

torch::Tensor cutlass_nvfp4_gemm_dispatch(
    torch::Tensor A,       // [M, K/2, L] float4_e2m1fn_x2 
    torch::Tensor B,       // [N, K/2, L] float4_e2m1fn_x2
    torch::Tensor SFA,     // Scale factors for A (permuted layout)
    torch::Tensor SFB,     // Scale factors for B (permuted layout)
    torch::Tensor C,       // [M, N, L] float16 output
    float alpha,
    float beta
);
"""

GEMM_CUDA = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

// ============================================================================
// Common types for all kernel configurations
// ============================================================================

// A matrix configuration
using ElementA    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag  = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;

// B matrix configuration  
using ElementB    = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutBTag  = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;

// Output configuration - FP16 output
using ElementC    = cutlass::half_t;
using ElementD    = cutlass::half_t;
using LayoutCTag  = cutlass::layout::RowMajor;
using LayoutDTag  = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

// Accumulator and compute types
using ElementAccumulator = float;
using ElementCompute     = float;
using ArchTag            = cutlass::arch::Sm100;
using OperatorClass      = cutlass::arch::OpClassBlockScaledTensorOp;

// Simple linear combination epilogue
using FusionOperation = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementCompute>;

// ============================================================================
// Configuration 1: Large K (K >= 8192) - Compute bound
// Use 2SM with 256x128x256 tiles, maximize compute throughput
// Benchmark case: M=128, N=7168, K=16384
// ============================================================================
namespace ConfigLargeK {
    using MmaTileShape = Shape<_256, _128, _256>;
    using ClusterShape = Shape<_2, _1, _1>;
    using PerSmTileShape_MNK = Shape<_128, _128, _256>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        PerSmTileShape_MNK, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementC, LayoutCTag, AlignmentC,
        ElementD, LayoutDTag, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto,
        FusionOperation
    >::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ElementA, LayoutATag, AlignmentA,
        ElementB, LayoutBTag, AlignmentB,
        ElementAccumulator,
        MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecialized2SmNvf4Sm100
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
}

// ============================================================================
// Configuration 2: Medium K (4096 <= K < 8192) - Balanced
// Use 2SM with 256x128x256 tiles
// Benchmark case: M=128, N=4096, K=7168
// ============================================================================
namespace ConfigMediumK {
    using MmaTileShape = Shape<_256, _128, _256>;
    using ClusterShape = Shape<_2, _1, _1>;
    using PerSmTileShape_MNK = Shape<_128, _128, _256>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        PerSmTileShape_MNK, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementC, LayoutCTag, AlignmentC,
        ElementD, LayoutDTag, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto,
        FusionOperation
    >::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ElementA, LayoutATag, AlignmentA,
        ElementB, LayoutBTag, AlignmentB,
        ElementAccumulator,
        MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecialized2SmNvf4Sm100
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
}

// ============================================================================
// Configuration 3: Small K (K < 4096) - Memory bound
// Use 1SM with 128x128x256 tiles for better occupancy
// Benchmark case: M=128, N=7168, K=2048
// ============================================================================
namespace ConfigSmallK {
    using MmaTileShape = Shape<_128, _128, _256>;
    using ClusterShape = Shape<_1, _1, _1>;
    using PerSmTileShape_MNK = Shape<_128, _128, _256>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        PerSmTileShape_MNK, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementC, LayoutCTag, AlignmentC,
        ElementD, LayoutDTag, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto,
        FusionOperation
    >::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ElementA, LayoutATag, AlignmentA,
        ElementB, LayoutBTag, AlignmentB,
        ElementAccumulator,
        MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecialized1SmNvf4Sm100
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
}

// ============================================================================
// Configuration 4: Large N with small K - Wider tiles
// Use 2SM with 256x256x256 tiles for maximum N coverage
// ============================================================================
namespace ConfigLargeN {
    using MmaTileShape = Shape<_256, _256, _256>;
    using ClusterShape = Shape<_2, _1, _1>;
    using PerSmTileShape_MNK = Shape<_128, _256, _256>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        PerSmTileShape_MNK, ClusterShape,
        cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator,
        ElementC, LayoutCTag, AlignmentC,
        ElementD, LayoutDTag, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto,
        FusionOperation
    >::CollectiveOp;

    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass,
        ElementA, LayoutATag, AlignmentA,
        ElementB, LayoutBTag, AlignmentB,
        ElementAccumulator,
        MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<
            static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecialized2SmNvf4Sm100
    >::CollectiveOp;

    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
        Shape<int, int, int, int>,
        CollectiveMainloop,
        CollectiveEpilogue,
        void>;

    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
}

// ============================================================================
// Templated kernel runner
// ============================================================================
template<typename Gemm, typename Sm1xxBlkScaledConfig>
void run_gemm_impl(
    const int M, const int N, const int K,
    void* a_ptr, void* b_ptr, void* sfa_ptr, void* sfb_ptr,
    void* c_ptr, void* d_ptr,
    float alpha, float beta,
    void* workspace_ptr)
{
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideC = typename Gemm::GemmKernel::StrideC;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
    
    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
        cute::make_shape(M, N, K, 1));
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
        cute::make_shape(M, N, K, 1));

    typename Gemm::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {
            reinterpret_cast<ElementA::DataType*>(a_ptr), stride_A,
            reinterpret_cast<ElementB::DataType*>(b_ptr), stride_B,
            reinterpret_cast<ElementA::ScaleFactorType*>(sfa_ptr), layout_SFA,
            reinterpret_cast<ElementB::ScaleFactorType*>(sfb_ptr), layout_SFB
        },
        {
            {alpha, beta},
            reinterpret_cast<ElementC*>(c_ptr), stride_C,
            reinterpret_cast<ElementD*>(d_ptr), stride_D
        }
    };

    // Tune scheduler parameters based on problem shape
    // Use heuristic to let CUTLASS decide
    arguments.scheduler.max_swizzle_size = 1;  // Minimal swizzle for small M
    arguments.scheduler.raster_order = cutlass::gemm::kernel::detail::RasterOrderOptions::Heuristic;

    Gemm gemm;
    auto status = gemm.initialize(arguments, workspace_ptr);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS GEMM initialization failed");
    }

    status = gemm.run();
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS GEMM execution failed");
    }
}

// ============================================================================
// Dispatch function - selects optimal kernel based on problem shape
// ============================================================================
torch::Tensor cutlass_nvfp4_gemm_dispatch(
    torch::Tensor A,
    torch::Tensor B, 
    torch::Tensor SFA,
    torch::Tensor SFB,
    torch::Tensor C,
    float alpha,
    float beta)
{
    const int M = A.size(0);
    const int K = A.size(1) * 2;
    const int N = B.size(0);

    void* a_ptr = A.data_ptr();
    void* b_ptr = B.data_ptr();
    void* sfa_ptr = SFA.data_ptr();
    void* sfb_ptr = SFB.data_ptr();
    void* c_ptr = C.data_ptr();

    // Allocate workspace (1MB should be sufficient)
    size_t workspace_size = 1024 * 1024;
    auto workspace = torch::empty({static_cast<long>(workspace_size)}, 
        torch::TensorOptions().dtype(torch::kUInt8).device(A.device()));

    // Dispatch based on problem shape
    // All cases use 2SM 256x128x256 which works best
    // Tune scheduling based on K
    if (K >= 8192) {
        // Large K: compute bound
        run_gemm_impl<ConfigLargeK::Gemm, ConfigLargeK::Sm1xxBlkScaledConfig>(
            M, N, K, a_ptr, b_ptr, sfa_ptr, sfb_ptr,
            c_ptr, c_ptr, alpha, beta, workspace.data_ptr());
    } else {
        // Medium/Small K: use same kernel
        run_gemm_impl<ConfigMediumK::Gemm, ConfigMediumK::Sm1xxBlkScaledConfig>(
            M, N, K, a_ptr, b_ptr, sfa_ptr, sfb_ptr,
            c_ptr, c_ptr, alpha, beta, workspace.data_ptr());
    }

    return C;
}
"""

_cutlass_gemm = None

def _get_kernel():
    """Lazy load and cache the compiled CUTLASS kernel."""
    global _cutlass_gemm
    if _cutlass_gemm is None:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        if sys.stdout is None:
            sys.stdout = io.StringIO()
        if sys.stderr is None:
            sys.stderr = io.StringIO()
        try:
            cutlass_include = "/workspace/cutlass/include"
            cutlass_tools_include = "/workspace/cutlass/tools/util/include"
            
            _cutlass_gemm = load_inline(
                name="cutlass_nvfp4_gemm_multi",
                cpp_sources=[GEMM_CPP],
                cuda_sources=[GEMM_CUDA],
                functions=["cutlass_nvfp4_gemm_dispatch"],
                extra_cuda_cflags=[
                    "-std=c++17",
                    "-gencode=arch=compute_100a,code=sm_100a",
                    "--ptxas-options=--gpu-name=sm_100a",
                    "-O3",
                    "-w",
                    "--use_fast_math",
                    "-allow-unsupported-compiler",
                    f"-I{cutlass_include}",
                    f"-I{cutlass_tools_include}",
                    "-DCUTLASS_ARCH_MMA_SM100_SUPPORTED=1",
                ],
                extra_ldflags=["-lcuda"],
                verbose=False,
            )
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    return _cutlass_gemm


def compile_kernel():
    """Pre-compile the kernel to exclude compilation time from benchmarks."""
    _get_kernel()


def custom_kernel(data: input_t) -> output_t:
    """
    Run CUTLASS GEMM on NVFP4 block-scaled inputs with shape-specialized dispatch.
    """
    a, b, sfa_cpu, sfb_cpu, sfa_permuted, sfb_permuted, c = data
    kernel = _get_kernel()
    
    return kernel.cutlass_nvfp4_gemm_dispatch(
        a, b, 
        sfa_permuted, sfb_permuted,
        c,
        1.0,  # alpha
        0.0   # beta
    )
