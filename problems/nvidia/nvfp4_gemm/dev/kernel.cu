// CUTLASS NVFP4 GEMM Kernel for Blackwell SM100
// This file is injected into submission.py by run.py

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
// Optimized for M=128 workloads using 2SM kernel
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

// =============================================================================
// OPTIMIZED Tile shapes for M=128 workloads
// Using 2SM kernel with larger tiles for better throughput
// =============================================================================
using MmaTileShape = Shape<_256, _128, _256>;      // 2SM: 256x256x256 (larger N tile)
using ClusterShape = Shape<_2, _1, _1>;            // 2 CTAs cooperating
using PerSmTileShape_MNK = Shape<_128, _128, _256>; // Per-SM tile for epilogue

constexpr int InputSFVectorSize = 16;

// Simple linear combination epilogue
using FusionOperation = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementCompute>;

// Use PerSmTileShape for epilogue builder (critical for 2SM kernels)
using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    PerSmTileShape_MNK, ClusterShape,  // Use PerSmTileShape, not MmaTileShape!
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

        auto status = gemm.initialize(arguments, workspace.data_ptr());
        if (status != cutlass::Status::kSuccess) {
            throw std::runtime_error("CUTLASS GEMM initialization failed");
        }

        status = gemm.run();
        if (status != cutlass::Status::kSuccess) {
            throw std::runtime_error("CUTLASS GEMM execution failed");
        }
    }

    return C;
}