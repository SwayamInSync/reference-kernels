# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a CUDA kernel optimization challenge for implementing a block-scaled dual GEMM (General Matrix Multiply) operation with SiLU activation, optimized for NVIDIA B200 GPUs (sm_100a architecture). The operation computes: `C = silu(A @ B1) * (A @ B2)` using FP4 (4-bit floating point) precision with block-scaled quantization.

## Core Concepts

### Data Format
- **FP4 Tensors**: Uses `torch.float4_e2m1fn_x2` (2-bit mantissa, 1-bit exponent)
- **Scale Factors**: Uses `torch.float8_e4m3fn` for block scaling (one scale factor per 16 elements, configurable via `sf_vec_size`)
- **Output**: FP16 (`torch.float16`)

### Tensor Layouts
All input matrices use K-major order:
- `a`: [M, K, L] in FP4
- `b1`, `b2`: [N, K, L] in FP4
- `sfa`: [M, K//16, L] in FP8 (scale factors for a)
- `sfb1`, `sfb2`: [N, K//16, L] in FP8 (scale factors for b1, b2)
- `c`: [M, N, L] in FP16 (output, pre-allocated)

The scale factor tensors come in two forms:
1. **Reference layout**: [M/N, K//16, L] - used by PyTorch's `torch._scaled_mm`
2. **Permuted layout**: [32, 4, rest_m/n, 4, rest_k, L] - optimized for CUTLASS/CuTe kernels

## Commands

### Running Tests
```bash
# Run all correctness tests
./run.sh test

# Run benchmarks (official evaluation mode)
./run.sh benchmark

# Run leaderboard mode
./run.sh leaderboard
```

### Environment Setup
The script automatically sets:
- `CUDA_VISIBLE_DEVICES=0` (single GPU)
- `POPCORN_FD=1` (structured output logging)
- `CUTE_DSL_ARCH=sm_100a` (target Blackwell architecture)

### Direct Python Execution
```bash
# For testing or debugging
CUDA_VISIBLE_DEVICES=0 POPCORN_FD=1 CUTE_DSL_ARCH=sm_100a python eval.py test task.yml

# For benchmarking
CUDA_VISIBLE_DEVICES=0 POPCORN_FD=1 CUTE_DSL_ARCH=sm_100a python eval.py benchmark task.yml
```

## Code Architecture

### File Structure
- **`submission.py`**: Main implementation file containing the `custom_kernel()` function and `compile_kernel()` for pre-compilation
- **`reference.py`**: Reference implementation using `torch._scaled_mm`, input generation, and correctness checking
- **`task.py`**: Type definitions (`input_t`, `output_t`, `TestSpec`)
- **`template.py`**: Starter template for new implementations
- **`utils.py`**: Shared utilities (seeding, device selection, tensor comparison, deterministic context)
- **`eval.py`**: Test harness that runs correctness tests and benchmarks
- **`task.yml`**: Configuration defining test cases, benchmarks, and speed-of-light targets

### Implementation Flow

1. **Input Generation** (`reference.py:generate_input`):
   - Creates random FP4 tensors with constrained bit patterns (only valid FP4 values)
   - Generates scale factors and produces both reference and permuted layouts
   - Returns 10-tuple: `(a, b1, b2, sfa, sfb1, sfb2, sfa_permuted, sfb1_permuted, sfb2_permuted, c)`

2. **Custom Kernel** (`submission.py:custom_kernel`):
   - Receives the 10-tuple input
   - Must compute `C = silu(A @ B1) * (A @ B2)` in-place into pre-allocated tensor `c`
   - Should use the permuted scale factor layouts for optimal performance
   - Returns the `c` tensor

3. **Reference Implementation** (`reference.py:ref_kernel`):
   - Uses PyTorch's `torch._scaled_mm` for each batch element
   - Converts scale factors to blocked format using `to_blocked()` helper
   - Applies SiLU activation and element-wise multiplication
   - Serves as ground truth for correctness checking

4. **Compilation Strategy** (`submission.py:compile_kernel`):
   - Optional function to pre-compile CUTLASS/CuTe kernels
   - Called once before benchmarking to exclude compilation time from measurements
   - Should handle kernel caching appropriately

### CUTLASS/CuTe Integration

The current implementation uses NVIDIA's CUTLASS library with CuTe DSL:
- **MMA Tile Shape**: Configurable via `_MMA_TILER_MN` (default 128x64)
- **Cluster Shape**: Multi-CTA configuration via `_CLUSTER_SHAPE_MN` (default 1x2)
- **Occupancy**: CTAs per SM (default 1)
- **TMA**: Tensor Memory Accelerator for efficient data movement
- **Block-Scaled MMA**: Uses `sm100_utils.make_blockscaled_trivial_tiled_mma()` for FP4 operations

Key classes:
- `Sm100BlockScaledPersistentDenseGemmKernel`: Main kernel configuration and launch logic
- Uses named barriers for synchronization (`epilog_sync_barrier`, `tmem_alloc_barrier`)
- Supports 2-CTA instructions for large tile sizes (256x64)

## Constraints and Requirements

### Matrix Size Constraints
- M must be divisible by `mma_tiler_mn[0]` (default 128)
- N must be divisible by `mma_tiler_mn[1]` (default 64)
- K must be divisible by 256
- L (batch size) is typically 1 in benchmarks

### Correctness Criteria
- Tolerance: `rtol=1e-3, atol=1e-3` (relatively loose due to FP4 precision)
- Must match reference implementation within tolerance

### Performance Ranking
- Ranked by geometric mean of benchmark results
- Grand prize: closest to speed-of-light analysis based on B200 hardware limits (1.5GHz clock)

### Speed-of-Light Targets (μs @ 1.5GHz)
```
M    N    K    L    time[μs]
256  4096 7168 1    4.708
512  4096 7168 1    8.714
256  3072 4096 1    2.125
512  3072 7168 1    6.535
```

## Development Notes

### Kernel Optimization Strategy
1. Use permuted scale factor layouts (`sfa_permuted`, `sfb1_permuted`, `sfb2_permuted`) for efficient memory access
2. Tune `_MMA_TILER_MN`, `_CLUSTER_SHAPE_MN`, and `_OCCUPANCY` for different problem sizes
3. Consider fusing both GEMMs and the SiLU + multiply epilogue into a single kernel
4. Leverage TMA for efficient global memory access
5. Use persistent kernel patterns to amortize launch overhead

### Compilation Artifacts
The repository contains pre-compiled PTX and CUBIN files for specific problem sizes:
- `cutlass_gemm_fp4_host_Ptrgmem_..._4096_7168_74.sm_100a.{ptx,cubin}`
- `cutlass_gemm_fp4_host_Ptrgmem_..._7168_16384_74.sm_100a.{ptx,cubin}`
- `cutlass_gemm_fp4_host_Ptrgmem_..._7168_2048_148.sm_100a.{ptx,cubin}`

These are auto-generated by CUTLASS and can be safely deleted (will regenerate on next run).

### Common Pitfalls
- Don't forget to handle the dual GEMM nature (two separate matrix multiplies with different B matrices)
- Scale factor indexing must match the blocked layout expected by the hardware
- The SiLU activation only applies to the first GEMM result, not the second
- Output tensor `c` is pre-allocated; write results in-place
- Ensure proper synchronization when using multi-CTA configurations

### Debugging Tips
- Use `eval.py` in test mode first to verify correctness before benchmarking
- Check scale factor layout transformation in `reference.py:create_scale_factor_tensors()`
- The `to_blocked()` helper shows the expected scale factor reordering pattern
- Set `CUDA_LAUNCH_BLOCKING=1` for more informative error messages
