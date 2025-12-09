import torch
import sys
import io
from torch.utils.cpp_extension import load_inline
from task import input_t, output_t

GEMM_CPP = r"""
#include <torch/extension.h>

torch::Tensor cutlass_nvfp4_gemm(
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
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

// ============================================================================
// CUTLASS GEMM Configuration for NVFP4 block-scaled GEMM on Blackwell SM100
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

// Tile shapes - can be tuned for better performance
using MmaTileShape = Shape<_128, _128, _256>;
using ClusterShape = Shape<_1, _1, _1>;

constexpr int InputSFVectorSize = 16;

// Simple linear combination epilogue
using FusionOperation = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementCompute>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    MmaTileShape, ClusterShape,
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
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA   = typename Gemm::GemmKernel::StrideA;
using StrideB   = typename Gemm::GemmKernel::StrideB;
using StrideC   = typename Gemm::GemmKernel::StrideC;
using StrideD   = typename Gemm::GemmKernel::StrideD;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

// ============================================================================
// PyTorch binding
// ============================================================================

torch::Tensor cutlass_nvfp4_gemm(
    torch::Tensor A,
    torch::Tensor B, 
    torch::Tensor SFA,
    torch::Tensor SFB,
    torch::Tensor C,
    float alpha,
    float beta)
{
    // Get dimensions
    const int M = A.size(0);
    const int K = A.size(1) * 2;  // K/2 packed elements
    const int L = A.size(2);
    const int N = B.size(0);

    // Create strides
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {M, N, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
    
    // Create scale factor layouts
    auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
        cute::make_shape(M, N, K, 1));
    auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
        cute::make_shape(M, N, K, 1));

    // Process each batch
    for (int batch = 0; batch < L; ++batch) {
        // Get batch pointers
        auto* a_ptr = reinterpret_cast<ElementA::DataType*>(
            static_cast<char*>(A.data_ptr()) + batch * M * (K/2) * sizeof(ElementA::DataType));
        auto* b_ptr = reinterpret_cast<ElementB::DataType*>(
            static_cast<char*>(B.data_ptr()) + batch * N * (K/2) * sizeof(ElementB::DataType));
        auto* sfa_ptr = reinterpret_cast<ElementA::ScaleFactorType*>(SFA.data_ptr());
        auto* sfb_ptr = reinterpret_cast<ElementB::ScaleFactorType*>(SFB.data_ptr());
        auto* c_ptr = reinterpret_cast<ElementC*>(
            C.data_ptr<at::Half>()) + batch * M * N;
        auto* d_ptr = c_ptr;

        // Create GEMM arguments
        typename Gemm::Arguments arguments{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K, 1},
            {
                a_ptr, stride_A,
                b_ptr, stride_B,
                sfa_ptr, layout_SFA,
                sfb_ptr, layout_SFB
            },
            {
                {alpha, beta},
                c_ptr, stride_C,
                d_ptr, stride_D
            }
        };

        // Get workspace size and allocate
        Gemm gemm;
        size_t workspace_size = Gemm::get_workspace_size(arguments);
        auto workspace = torch::empty({static_cast<long>(workspace_size)}, 
            torch::TensorOptions().dtype(torch::kUInt8).device(A.device()));

        // Initialize and run
        auto status = gemm.initialize(arguments, workspace.data_ptr());
        if (status != cutlass::Status::kSuccess) {
            throw std::runtime_error("CUTLASS GEMM initialization failed");
        }

        status = gemm.run();
        if (status != cutlass::Status::kSuccess) {
            throw std::runtime_error("CUTLASS GEMM execution failed");
        }
    }

    cudaDeviceSynchronize();
    return C;
}
"""

_cutlass_gemm = None

def _get_kernel():
    """Lazy load and cache the compiled CUTLASS kernel."""
    global _cutlass_gemm
    if _cutlass_gemm is None:
        # Handle stdout/stderr for multiprocessing compatibility
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        if sys.stdout is None:
            sys.stdout = io.StringIO()
        if sys.stderr is None:
            sys.stderr = io.StringIO()
        try:
            # CUTLASS include paths - adjust if needed for your environment
            cutlass_include = "/workspace/cutlass/include"
            cutlass_tools_include = "/workspace/cutlass/tools/util/include"
            
            _cutlass_gemm = load_inline(
                name="cutlass_nvfp4_gemm",
                cpp_sources=[GEMM_CPP],
                cuda_sources=[GEMM_CUDA],
                functions=["cutlass_nvfp4_gemm"],
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
    Run CUTLASS GEMM on NVFP4 block-scaled inputs.
    
    Args:
        data: Tuple of (a, b, sfa, sfb, sfa_permuted, sfb_permuted, c)
            - a: [M, K/2, L] float4_e2m1fn_x2
            - b: [N, K/2, L] float4_e2m1fn_x2
            - sfa: [M, K//16, L] float8_e4m3fn (CPU layout)
            - sfb: [N, K//16, L] float8_e4m3fn (CPU layout)
            - sfa_permuted: Permuted scale factors for CUTLASS
            - sfb_permuted: Permuted scale factors for CUTLASS
            - c: [M, N, L] float16 output buffer
    
    Returns:
        c: [M, N, L] float16 output tensor
    """
    a, b, sfa_cpu, sfb_cpu, sfa_permuted, sfb_permuted, c = data
    kernel = _get_kernel()
    
    return kernel.cutlass_nvfp4_gemm(
        a, b, 
        sfa_permuted, sfb_permuted,
        c,
        1.0,  # alpha
        0.0   # beta
    )
