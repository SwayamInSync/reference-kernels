// cuBLASLt NVFP4 GEMM Implementation
// 
// This implements NVFP4 block-scaled GEMM using cuBLASLt API.
// Key requirements from cuBLAS documentation:
// 1. A must be transposed (CUBLAS_OP_T), B must be non-transposed (CUBLAS_OP_N) - "TN" format
// 2. Scale mode: CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3
// 3. Compute type: CUBLAS_COMPUTE_32F, Scale type: CUDA_R_32F
// 4. Supported output types: CUDA_R_16F, CUDA_R_16BF, or CUDA_R_32F
//
// Scale factor layout from cuBLAS docs:
// For VEC16_UE4M3:
//   - Block size: 128 (outer) x 64 (inner) data elements
//   - Scale tile: 32 (outer) x 4 (inner) scale factors per 128x64 data block
//   - Layout: [n_row_blocks * n_col_blocks, 32, 16] where each 32x16 tile covers 128x64 data

#include <cuda.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <iostream>

// Helper macros for error checking
#define checkCublasStatus(status) do { \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << " - status: " << status << std::endl; \
        throw std::runtime_error("cuBLAS error: " + std::to_string(status)); \
    } \
} while(0)

#define checkCudaStatus(status) do { \
    if (status != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(status) << std::endl; \
        throw std::runtime_error("CUDA error: " + std::string(cudaGetErrorString(status))); \
    } \
} while(0)

// ceil division helper
__host__ __device__ inline int ceilDiv(int a, int b) {
    return (a + b - 1) / b;
}

// Kernel to convert scale factors from [rows, cols] to cuBLAS blocked format
// The blocked format is: view as [n_row_blocks, 128, n_col_blocks, 4]
//                        permute to [n_row_blocks, n_col_blocks, 128, 4]
//                        reshape to [-1, 4, 32, 4], transpose to [-1, 32, 4, 4]
//                        reshape to [-1, 32, 16] and flatten
//
// For input [rows, cols], n_row_blocks = ceil(rows/128), n_col_blocks = ceil(cols/4)
// Output size: n_row_blocks * n_col_blocks * 128 * 4 = n_row_blocks * n_col_blocks * 512
__global__ void convertScaleFactorsToBlocked(
    const __nv_fp8_e4m3* __restrict__ input,  // [rows, cols] - row-major
    __nv_fp8_e4m3* __restrict__ output,       // blocked format
    int rows,
    int cols)
{
    // Output layout: [n_row_blocks * n_col_blocks, 32, 16]
    // After the full transformation, the mapping is:
    // output[rb * n_col_blocks + cb, s32, s16] corresponds to:
    //   s4_outer = s16 / 4
    //   s4_inner = s16 % 4
    //   row = rb * 128 + s4_outer * 4 + s32
    //   col = cb * 4 + s4_inner
    // But s32 goes 0..31 for s4_outer=0, then 0..31 for s4_outer=1, etc.
    // Actually the reshape is: [-1, 4, 32, 4] -> transpose(1,2) -> [-1, 32, 4, 4] -> reshape [-1, 32, 16]
    // So s16 = s4_outer * 4 + s4_inner where s4_outer is the original axis 1 (range 0-3)
    // and s4_inner is axis 3 (range 0-3)
    // row in input = rb * 128 + s4_outer * 32 + s32
    // col in input = cb * 4 + s4_inner
    
    int n_row_blocks = ceilDiv(rows, 128);
    int n_col_blocks = ceilDiv(cols, 4);
    int total_output = n_row_blocks * n_col_blocks * 32 * 16;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_output) return;
    
    // Decode output index
    int s16 = idx % 16;
    int s32 = (idx / 16) % 32;
    int block_idx = idx / (32 * 16);
    int rb = block_idx / n_col_blocks;
    int cb = block_idx % n_col_blocks;
    
    // Map s16 back to s4_outer and s4_inner
    int s4_outer = s16 / 4;
    int s4_inner = s16 % 4;
    
    // Calculate input coordinates
    int row = rb * 128 + s4_outer * 32 + s32;
    int col = cb * 4 + s4_inner;
    
    // Read from input (with bounds check for padding)
    __nv_fp8_e4m3 val;
    if (row < rows && col < cols) {
        val = input[row * cols + col];
    } else {
        // Pad with 1.0 (scale factor of 1 means no scaling)
        // FP8 E4M3: 1.0 = 0x38 (sign=0, exp=0111, mantissa=000)
        val = *reinterpret_cast<const __nv_fp8_e4m3*>(&(uint8_t){0x38});
    }
    
    output[idx] = val;
}

// cuBLASLt NVFP4 matmul wrapper
// We compute C = A @ B^T where A is (M,K) and B is (N,K)
// cuBLAS uses column-major, so we actually compute:
//   C^T[col-major] = B @ A^T
// where C^T is (N,M) col-major = (M,N) row-major (what we want)
// For this: transa=N (B is K,N), transb=T (A is K,M -> M,K transposed)
// But wait, that doesn't work with the FP4 constraints...
//
// Alternative: We keep TN format but swap the roles:
// Compute C_colmaj = A_cublasT @ B_cublasN
// where A_cublas is our B[N,K/2] (so K,N in colmaj), transposed -> N,K
// and B_cublas is our A[M,K/2] (so K,M in colmaj), not transposed -> K,M
// Result: C_colmaj = (N,K) @ (K,M) = (N,M)
// In row-major this is [M,N] - our desired output!
void LtNvfp4Matmul(cublasLtHandle_t ltHandle,
                   int m,  // rows of logical A (our A matrix)
                   int n,  // rows of logical B (our B matrix)
                   int k,  // common dimension
                   const float alpha,
                   const __nv_fp8_e4m3 *a_scale,  // Scale factors for our A in blocked format
                   const void *A,                  // Our A: [M, K/2] FP4 packed
                   const __nv_fp8_e4m3 *b_scale,  // Scale factors for our B in blocked format
                   const void *B,                  // Our B: [N, K/2] FP4 packed
                   const float beta,
                   __half *C,                      // Output: [M, N] FP16
                   void *workspace,
                   size_t workspaceSize,
                   cudaStream_t stream) 
{
    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL;
    cublasLtMatmulPreference_t preference = NULL;

    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};

    // We use TN format but swap A and B:
    // cuBLAS_A = our B: (K, N) col-major, transposed -> (N, K)
    // cuBLAS_B = our A: (K, M) col-major, not transposed -> (K, M)
    // Result: (N, K) @ (K, M) = (N, M) col-major = [M, N] row-major
    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;

    // Create operation descriptor with FP32 compute
    checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // Set block scaling mode for A and B (swapped)
    cublasLtMatmulMatrixScale_t scaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode, sizeof(scaleMode)));

    // Set scaling factor pointers (swapped: A_scale for cuBLAS_A which is our B)
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &b_scale, sizeof(b_scale)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &a_scale, sizeof(a_scale)));

    // Matrix layouts (swapped):
    // cuBLAS_A = our B: [N, K/2] row-major = (K/2, N) col-major = (K, N) FP4 logical
    // cuBLAS_B = our A: [M, K/2] row-major = (K/2, M) col-major = (K, M) FP4 logical  
    // cuBLAS_C: (N, M) col-major = [M, N] row-major
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_4F_E2M1, k, n, k));  // cuBLAS_A = our B
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_4F_E2M1, k, m, k));  // cuBLAS_B = our A
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16F, n, m, n));      // (N, M) col-major

    // Create preference handle
    checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
    checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                           &workspaceSize, sizeof(workspaceSize)));

    // Get heuristic
    checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1,
                                                     &heuristicResult, &returnedResults));

    if (returnedResults == 0) {
        throw std::runtime_error("cuBLASLt: No algorithm found for NVFP4 GEMM configuration");
    }

    // Run matmul (note: B and A swapped in the call)
    checkCublasStatus(cublasLtMatmul(ltHandle, operationDesc, &alpha, B, Adesc, A, Bdesc, &beta, C, Cdesc, C, Cdesc,
                                     &heuristicResult.algo, workspace, workspaceSize, stream));

    // Cleanup
    if (preference) checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
    if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
    if (operationDesc) checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
}

// Main entry point
// Input:
//   A: [M, K/2, L] float4_e2m1fn_x2 (row-major, packed K)
//   B: [N, K/2, L] float4_e2m1fn_x2 (row-major, packed K)
//   SFA: [M, sf_K, L] or permuted format - scale factors for A
//   SFB: [N, sf_K, L] or permuted format - scale factors for B
//   C: [M, N, L] float16 output
//
// Note: We receive the NON-PERMUTED scale factors (sfa_ref_cpu, sfb_ref_cpu from reference.py)
torch::Tensor cublaslt_nvfp4_gemm(
    torch::Tensor A,       // [M, K/2, L] float4_e2m1fn_x2 
    torch::Tensor B,       // [N, K/2, L] float4_e2m1fn_x2
    torch::Tensor SFA,     // [M, sf_K, L] float8_e4m3 - NON-PERMUTED scale factors
    torch::Tensor SFB,     // [N, sf_K, L] float8_e4m3 - NON-PERMUTED scale factors
    torch::Tensor C,       // [M, N, L] float16 output
    float alpha,
    float beta)
{
    const int M = A.size(0);
    const int K = A.size(1) * 2;
    const int L = A.size(2);
    const int N = B.size(0);
    const int sf_K = K / 16;  // Number of scale factor columns

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Create cuBLASLt handle
    cublasLtHandle_t ltHandle;
    checkCublasStatus(cublasLtCreate(&ltHandle));

    // Allocate workspace
    size_t workspaceSize = 32 * 1024 * 1024;
    void* workspace;
    checkCudaStatus(cudaMalloc(&workspace, workspaceSize));

    // Calculate blocked scale factor sizes
    int sf_a_rows = M;
    int sf_a_cols = sf_K;
    int sf_b_rows = N;
    int sf_b_cols = sf_K;
    
    int n_row_blocks_a = ceilDiv(sf_a_rows, 128);
    int n_col_blocks_a = ceilDiv(sf_a_cols, 4);
    int blocked_size_a = n_row_blocks_a * n_col_blocks_a * 32 * 16;
    
    int n_row_blocks_b = ceilDiv(sf_b_rows, 128);
    int n_col_blocks_b = ceilDiv(sf_b_cols, 4);
    int blocked_size_b = n_row_blocks_b * n_col_blocks_b * 32 * 16;

    // Allocate blocked scale factor buffers
    __nv_fp8_e4m3 *blocked_sfa, *blocked_sfb;
    checkCudaStatus(cudaMalloc(&blocked_sfa, blocked_size_a * sizeof(__nv_fp8_e4m3)));
    checkCudaStatus(cudaMalloc(&blocked_sfb, blocked_size_b * sizeof(__nv_fp8_e4m3)));

    // Batch strides
    size_t a_batch_stride = M * (K / 2);
    size_t b_batch_stride = N * (K / 2);
    size_t c_batch_stride = M * N;
    size_t sfa_batch_stride = M * sf_K;
    size_t sfb_batch_stride = N * sf_K;

    // Process each batch
    for (int batch = 0; batch < L; ++batch) {
        // Get batch pointers for input data
        const void* a_ptr = static_cast<const char*>(A.data_ptr()) + batch * a_batch_stride;
        const void* b_ptr = static_cast<const char*>(B.data_ptr()) + batch * b_batch_stride;
        const __nv_fp8_e4m3* sfa_ptr = reinterpret_cast<const __nv_fp8_e4m3*>(
            static_cast<const char*>(SFA.data_ptr())) + batch * sfa_batch_stride;
        const __nv_fp8_e4m3* sfb_ptr = reinterpret_cast<const __nv_fp8_e4m3*>(
            static_cast<const char*>(SFB.data_ptr())) + batch * sfb_batch_stride;
        __half* c_ptr = reinterpret_cast<__half*>(C.data_ptr<at::Half>()) + batch * c_batch_stride;

        // Convert scale factors to blocked format
        int threads = 256;
        int blocks_a = (blocked_size_a + threads - 1) / threads;
        int blocks_b = (blocked_size_b + threads - 1) / threads;
        
        convertScaleFactorsToBlocked<<<blocks_a, threads, 0, stream>>>(
            sfa_ptr, blocked_sfa, sf_a_rows, sf_a_cols);
        convertScaleFactorsToBlocked<<<blocks_b, threads, 0, stream>>>(
            sfb_ptr, blocked_sfb, sf_b_rows, sf_b_cols);

        // Run cuBLASLt matmul
        LtNvfp4Matmul(
            ltHandle,
            M, N, K,
            alpha,
            blocked_sfa, a_ptr,
            blocked_sfb, b_ptr,
            beta,
            c_ptr,
            workspace,
            workspaceSize,
            stream
        );
    }

    cudaStreamSynchronize(stream);
    
    // Cleanup
    cudaFree(blocked_sfa);
    cudaFree(blocked_sfb);
    cudaFree(workspace);
    checkCublasStatus(cublasLtDestroy(ltHandle));

    return C;
}
