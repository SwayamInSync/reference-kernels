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
#include <mutex>

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

// Global cached resources for performance
static cublasLtHandle_t g_ltHandle = nullptr;
static void* g_workspace = nullptr;
static size_t g_workspaceSize = 32 * 1024 * 1024;
static std::once_flag g_init_flag;

// Initialize global resources (called once)
static void initGlobalResources() {
    checkCublasStatus(cublasLtCreate(&g_ltHandle));
    checkCudaStatus(cudaMalloc(&g_workspace, g_workspaceSize));
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

// Main entry point - accepts PRE-BLOCKED scale factors for fair comparison with CUTLASS
// Input:
//   A: [M, K/2, L] float4_e2m1fn_x2 (row-major, packed K)
//   B: [N, K/2, L] float4_e2m1fn_x2 (row-major, packed K)
//   SFA: [blocked_size_a, L] float8_e4m3 - PRE-BLOCKED scale factors for A
//   SFB: [blocked_size_b, L] float8_e4m3 - PRE-BLOCKED scale factors for B
//   C: [M, N, L] float16 output
//
// Scale factors should be pre-converted to cuBLAS blocked format using to_blocked_cublas()
torch::Tensor cublaslt_nvfp4_gemm(
    torch::Tensor A,       // [M, K/2, L] float4_e2m1fn_x2 
    torch::Tensor B,       // [N, K/2, L] float4_e2m1fn_x2
    torch::Tensor SFA,     // [blocked_size_a, L] float8_e4m3 - PRE-BLOCKED
    torch::Tensor SFB,     // [blocked_size_b, L] float8_e4m3 - PRE-BLOCKED
    torch::Tensor C,       // [M, N, L] float16 output
    float alpha,
    float beta)
{
    // Initialize global resources once
    std::call_once(g_init_flag, initGlobalResources);

    const int M = A.size(0);
    const int K = A.size(1) * 2;
    const int L = A.size(2);
    const int N = B.size(0);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Get blocked scale factor sizes from the pre-blocked tensors
    size_t blocked_size_a = SFA.size(0);
    size_t blocked_size_b = SFB.size(0);

    // Batch strides
    size_t a_batch_stride = M * (K / 2);
    size_t b_batch_stride = N * (K / 2);
    size_t c_batch_stride = M * N;

    // Process each batch
    for (int batch = 0; batch < L; ++batch) {
        // Get batch pointers for input data
        const void* a_ptr = static_cast<const char*>(A.data_ptr()) + batch * a_batch_stride;
        const void* b_ptr = static_cast<const char*>(B.data_ptr()) + batch * b_batch_stride;
        
        // Scale factors are already in blocked format - just get batch pointers
        const __nv_fp8_e4m3* sfa_ptr = reinterpret_cast<const __nv_fp8_e4m3*>(
            static_cast<const char*>(SFA.data_ptr())) + batch * blocked_size_a;
        const __nv_fp8_e4m3* sfb_ptr = reinterpret_cast<const __nv_fp8_e4m3*>(
            static_cast<const char*>(SFB.data_ptr())) + batch * blocked_size_b;
        __half* c_ptr = reinterpret_cast<__half*>(C.data_ptr<at::Half>()) + batch * c_batch_stride;

        // Run cuBLASLt matmul directly - no scale conversion needed!
        LtNvfp4Matmul(
            g_ltHandle,
            M, N, K,
            alpha,
            sfa_ptr, a_ptr,
            sfb_ptr, b_ptr,
            beta,
            c_ptr,
            g_workspace,
            g_workspaceSize,
            stream
        );
    }

    // Don't synchronize here - let PyTorch handle it
    // This allows async execution and proper benchmarking
    
    return C;
}
