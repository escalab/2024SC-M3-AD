#include "precTest.hpp"

// random float number between -1 and 1
#define randf() 2*( (double)rand() / (double)RAND_MAX )-1

// Detect errors
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
    if (stat != cudaSuccess)
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
}
#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
    if (stat != CUBLAS_STATUS_SUCCESS)
        fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
}

int main(int argc, char** argv )
{
    if ( argc != 2 ) {
        printf("usage: ./precTest [size]\n");
        return -1;
    }

    printf("SIZE: %s\n", argv[1]);

    cublasHandle_t handle;
    cublasCreate(&handle);

    // set sizes
    int M = atoi(argv[1]);
    int N = atoi(argv[1]);
    int K = atoi(argv[1]);

    double *h_A, *h_B;
    double *h_C, *h_D;
    cudaMallocHost(&h_A, M * K * sizeof(double));
    cudaMallocHost(&h_B, K * N * sizeof(double));
    cudaMallocHost(&h_C, M * N * sizeof(double));
    cudaMallocHost(&h_D, M * N * sizeof(double));

    // initialize h_A and h_B
    srand((unsigned)time(NULL));
    for (int i = 0; i < M * K; i++)
        h_A[i] = randf();
    for (int i = 0; i < K * N; i++)
        h_B[i] = randf();
    for (int i = 0; i < M * N; i++) {
        h_C[i] = randf();
        h_D[i] = 0;
    }

    // print h_A, h_B, and h_C
    // std::cout << std::endl;
    // for(int i=0; i<M; i++) {
    //     for(int j=0; j<K; j++) {
    //         std::cout << (double)h_A[i*K+j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;
    // for(int i=0; i<K; i++) {
    //     for(int j=0; j<N; j++) {
    //         std::cout << (double)h_B[i*N+j] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;
    // for(int i=0; i<M; i++) {
    //     for(int j=0; j<N; j++) {
    //         std::cout << (double)h_C[j*M+i] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;

    // copy h_A and h_B to device
    double *d_A;
    cudaMalloc(&d_A, M * K * sizeof(double));
    cudaMemcpy(d_A, h_A, M * K * sizeof(double), cudaMemcpyHostToDevice);
    
    double *d_B;
    cudaMalloc(&d_B, K * N * sizeof(double));
    cudaMemcpy(d_B, h_B, K * N * sizeof(double), cudaMemcpyHostToDevice);
    
    double *d_C;
    cudaMalloc(&d_C, M * N * sizeof(double));
    cudaMemcpy(d_C, h_C, M * N * sizeof(double), cudaMemcpyHostToDevice);

    // launch kernel
    const double alphaD = 1;
    const double betaD = 1;

/// FP64 //////////////////////////////////////////////////////////////////////
    cublasErrCheck( cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, 
                                &alphaD, 
                                d_A, CUDA_R_64F, K, 
                                d_B, CUDA_R_64F, N, 
                                &betaD, 
                                d_C, CUDA_R_64F, M, 
                                CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );

    cudaMemcpy(h_D, d_C, M * N * sizeof(double), cudaMemcpyDeviceToHost);

    // C is transposed
    // for(int i=0; i<M; i++) {
    //     for(int j=0; j<N; j++) {
    //         std::cout << (double)h_C[j*M+i] << " ";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;

/// for different data types //////////////////////////////////////////////////

    const float alphaF = 1;
    const float betaF = 1;

    float *h_A1, *h_B1;
    float *h_C1, *h_D1;
    cudaMallocHost(&h_A1, M * K * sizeof(float));
    cudaMallocHost(&h_B1, K * N * sizeof(float));
    cudaMallocHost(&h_C1, M * N * sizeof(float));
    cudaMallocHost(&h_D1, M * N * sizeof(float));

    for (int i = 0; i < M * K; i++)
        h_A1[i] = (float)h_A[i];
    for (int i = 0; i < K * N; i++)
        h_B1[i] = (float)h_B[i];
    for (int i = 0; i < M * N; i++) {
        h_C1[i] = (float)h_C[i];
        h_D1[i] = 0;
    }

    // copy h_A and h_B to device
    float *d_A1;
    cudaMalloc(&d_A1, M * K * sizeof(float));
    cudaMemcpy(d_A1, h_A1, M * K * sizeof(float), cudaMemcpyHostToDevice);
    
    float *d_B1;
    cudaMalloc(&d_B1, K * N * sizeof(float));
    cudaMemcpy(d_B1, h_B1, K * N * sizeof(float), cudaMemcpyHostToDevice);
    
    float *d_C1;
    cudaMalloc(&d_C1, M * N * sizeof(float));

    std::cout<<std::fixed;
    std::cout.precision(15);

/// FP32 = 3BP16 (Ma) = 3TF32 (cuTlass) ///////////////////////////////////////
    cudaMemcpy(d_C1, h_C1, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cublasErrCheck( cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, 
                            &alphaF, 
                            d_A1, CUDA_R_32F, K, 
                            d_B1, CUDA_R_32F, N, 
                            &betaF, 
                            d_C1, CUDA_R_32F, M, 
                            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );

    cudaMemcpy(h_D1, d_C1, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    double err = 0;
    for (int i=0; i<M*N; i++) {
        if(err < fabs((double)h_D1[i] - h_D[i]))
            err = fabs((double)h_D1[i] - h_D[i]);
    }
    std::cout << "FP32 error: \t" << err << std::endl;

    cudaMemcpy(d_C1, h_C1, M * N * sizeof(float), cudaMemcpyHostToDevice);

/// FP16 //////////////////////////////////////////////////////////////////////
    cudaMemcpy(d_C1, h_C1, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cublasErrCheck( cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, 
                            &alphaF, 
                            d_A1, CUDA_R_32F, K, 
                            d_B1, CUDA_R_32F, N, 
                            &betaF, 
                            d_C1, CUDA_R_32F, M, 
                            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );

    cudaMemcpy(h_D1, d_C1, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    err = 0;
    for (int i=0; i<M*N; i++) {
        if(err < fabs((double)h_D1[i] - h_D[i]))
            err = fabs((double)h_D1[i] - h_D[i]);
    }
    std::cout << "FP16 error: \t" << err << std::endl;

/// BP16 //////////////////////////////////////////////////////////////////////
    cudaMemcpy(d_C1, h_C1, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cublasErrCheck( cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, 
                            &alphaF, 
                            d_A1, CUDA_R_32F, K, 
                            d_B1, CUDA_R_32F, N, 
                            &betaF, 
                            d_C1, CUDA_R_32F, M, 
                            CUBLAS_COMPUTE_32F_FAST_16BF, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );

    cudaMemcpy(h_D1, d_C1, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    err = 0;
    for (int i=0; i<M*N; i++) {
        if(err < fabs((double)h_D1[i] - h_D[i]))
            err = fabs((double)h_D1[i] - h_D[i]);
    }
    std::cout << "BFP16 error: \t" << err << std::endl;

/// TF32 //////////////////////////////////////////////////////////////////////
    cudaMemcpy(d_C1, h_C1, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cublasErrCheck( cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, 
                            &alphaF, 
                            d_A1, CUDA_R_32F, K, 
                            d_B1, CUDA_R_32F, N, 
                            &betaF, 
                            d_C1, CUDA_R_32F, M, 
                            CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );

    cudaMemcpy(h_D1, d_C1, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    err = 0;
    for (int i=0; i<M*N; i++) {
        if(err < fabs((double)h_D1[i] - h_D[i]))
            err = fabs((double)h_D1[i] - h_D[i]);
    }
    std::cout << "TF32 error: \t" << err << std::endl;

    cudaFree(d_A1);
    cudaFree(d_B1);
    cudaFree(d_C1);
    cudaFreeHost(h_A1);
    cudaFreeHost(h_B1);
    cudaFreeHost(h_C1);
    cudaFreeHost(h_D1);

/// 2FP16 (Markidis) //////////////////////////////////////////////////////////

    // half *h_AH1, *h_AL1, *h_BH1, *h_BL1;
    // cudaMallocHost(&h_AH1, M * K * sizeof(half));
    // cudaMallocHost(&h_AL1, M * K * sizeof(half));
    // cudaMallocHost(&h_BH1, K * N * sizeof(half));
    // cudaMallocHost(&h_BL1, K * N * sizeof(half));
    // cudaMallocHost(&h_C1, M * N * sizeof(float));
    // cudaMallocHost(&h_D1, M * N * sizeof(float));

    // for (int i = 0; i < M * K; i++) {
    //     h_AH1[i] = (half)h_A[i];
    //     h_AL1[i] = (half)(h_A[i] - (float)(h_AH1[i]));
    // }
    // for (int i = 0; i < K * N; i++) {
    //     h_BH1[i] = (half)h_B[i];
    //     h_BL1[i] = (half)(h_B[i] - (float)(h_BH1[i]));
    // }
    // for (int i = 0; i < M * N; i++) {
    //     h_C1[i] = (float)h_C[i];
    //     h_D1[i] = 0;
    // }

    // // copy h_A and h_B to device
    // half *d_AH1;
    // cudaMalloc(&d_AH1, M * K * sizeof(half));
    // cudaMemcpy(d_AH1, h_AH1, M * K * sizeof(half), cudaMemcpyHostToDevice);

    // half *d_AL1;
    // cudaMalloc(&d_AL1, M * K * sizeof(half));
    // cudaMemcpy(d_AL1, h_AL1, M * K * sizeof(half), cudaMemcpyHostToDevice);
    
    // half *d_BH1;
    // cudaMalloc(&d_BH1, K * N * sizeof(half));
    // cudaMemcpy(d_BH1, h_BH1, K * N * sizeof(half), cudaMemcpyHostToDevice);

    // half *d_BL1;
    // cudaMalloc(&d_BL1, K * N * sizeof(half));
    // cudaMemcpy(d_BL1, h_BL1, K * N * sizeof(half), cudaMemcpyHostToDevice);
    
    // cudaMalloc(&d_C1, M * N * sizeof(float));
    // cudaMemcpy(d_C1, h_C1, M * N * sizeof(float), cudaMemcpyHostToDevice);

    // cublasErrCheck( cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, 
    //                         &alphaF, 
    //                         d_AH1, CUDA_R_16F, K, 
    //                         d_BH1, CUDA_R_16F, N, 
    //                         &betaF, 
    //                         d_C1, CUDA_R_32F, M, 
    //                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );
    // cublasErrCheck( cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, 
    //                         &alphaF, 
    //                         d_AH1, CUDA_R_16F, K, 
    //                         d_BL1, CUDA_R_16F, N, 
    //                         &betaF, 
    //                         d_C1, CUDA_R_32F, M, 
    //                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );
    // cublasErrCheck( cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, 
    //                         &alphaF, 
    //                         d_AL1, CUDA_R_16F, K, 
    //                         d_BH1, CUDA_R_16F, N, 
    //                         &betaF, 
    //                         d_C1, CUDA_R_32F, M, 
    //                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );
    // cublasErrCheck( cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, 
    //                         &alphaF, 
    //                         d_AL1, CUDA_R_16F, K, 
    //                         d_BL1, CUDA_R_16F, N, 
    //                         &betaF, 
    //                         d_C1, CUDA_R_32F, M, 
    //                         CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );

    // cudaMemcpy(h_D1, d_C1, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // err = 0;
    // for (int i=0; i<M*N; i++) {
    //     if(err < fabs((double)h_D1[i] - h_D[i]))
    //         err = fabs((double)h_D1[i] - h_D[i]);
    // }
    // std::cout << "2FP16 error: \t" << err << std::endl;

    // cudaFree(d_AH1);
    // cudaFree(d_AL1);
    // cudaFree(d_BH1);
    // cudaFree(d_BL1);
    // cudaFree(d_C1);
    // cudaFreeHost(h_AH1);
    // cudaFreeHost(h_AL1);
    // cudaFreeHost(h_BH1);
    // cudaFreeHost(h_BL1);
    // cudaFreeHost(h_C1);
    // cudaFreeHost(h_D1);

/// 2TF32 (Ma) ////////////////////////////////////////////////////////////////

    float *h_AH2, *h_AL2, *h_BH2, *h_BL2;
    cudaMallocHost(&h_AH2, M * K * sizeof(float));
    cudaMallocHost(&h_AL2, M * K * sizeof(float));
    cudaMallocHost(&h_BH2, K * N * sizeof(float));
    cudaMallocHost(&h_BL2, K * N * sizeof(float));
    cudaMallocHost(&h_C1, M * N * sizeof(float));
    cudaMallocHost(&h_D1, M * N * sizeof(float));

    for (int i = 0; i < M * K; i++) {
        h_AH2[i] = (half)h_A[i];
        h_AL2[i] = (half)(h_A[i] - h_AH2[i]);
    }
    for (int i = 0; i < K * N; i++) {
        h_BH2[i] = (half)h_B[i];
        h_BL2[i] = (half)(h_B[i] - h_BH2[i]);
    }
    for (int i = 0; i < M * N; i++) {
        h_C1[i] = (float)h_C[i];
        h_D1[i] = 0;
    }

    // copy h_A and h_B to device
    float *d_AH2;
    cudaMalloc(&d_AH2, M * K * sizeof(float));
    cudaMemcpy(d_AH2, h_AH2, M * K * sizeof(float), cudaMemcpyHostToDevice);

    float *d_AL2;
    cudaMalloc(&d_AL2, M * K * sizeof(float));
    cudaMemcpy(d_AL2, h_AL2, M * K * sizeof(float), cudaMemcpyHostToDevice);
    
    float *d_BH2;
    cudaMalloc(&d_BH2, K * N * sizeof(float));
    cudaMemcpy(d_BH2, h_BH2, K * N * sizeof(float), cudaMemcpyHostToDevice);

    float *d_BL2;
    cudaMalloc(&d_BL2, K * N * sizeof(float));
    cudaMemcpy(d_BL2, h_BL2, K * N * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaMalloc(&d_C1, M * N * sizeof(float));
    cudaMemcpy(d_C1, h_C1, M * N * sizeof(float), cudaMemcpyHostToDevice);

    cublasErrCheck( cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, 
                            &alphaF, 
                            d_AH2, CUDA_R_32F, K, 
                            d_BH2, CUDA_R_32F, N, 
                            &betaF, 
                            d_C1, CUDA_R_32F, M, 
                            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );
    cublasErrCheck( cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, 
                            &alphaF, 
                            d_AH2, CUDA_R_32F, K, 
                            d_BL2, CUDA_R_32F, N, 
                            &betaF, 
                            d_C1, CUDA_R_32F, M, 
                            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );
    cublasErrCheck( cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, 
                            &alphaF, 
                            d_AL2, CUDA_R_32F, K, 
                            d_BH2, CUDA_R_32F, N, 
                            &betaF, 
                            d_C1, CUDA_R_32F, M, 
                            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );
    cublasErrCheck( cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, 
                            &alphaF, 
                            d_AL2, CUDA_R_32F, K, 
                            d_BL2, CUDA_R_32F, N, 
                            &betaF, 
                            d_C1, CUDA_R_32F, M, 
                            CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );

    cudaMemcpy(h_D1, d_C1, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // err = 0;
    // for (int i=0; i<M*N; i++) {
    //     if(err < fabs((double)h_D1[i] - h_D[i]))
    //         err = fabs((double)h_D1[i] - h_D[i]);
    // }
    // std::cout << "2FP16 error: \t" << err << std::endl;

    cudaMemcpy(d_C1, h_C1, M * N * sizeof(float), cudaMemcpyHostToDevice);

    cublasErrCheck( cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, 
                            &alphaF, 
                            d_AH2, CUDA_R_32F, K, 
                            d_BH2, CUDA_R_32F, N, 
                            &betaF, 
                            d_C1, CUDA_R_32F, M, 
                            CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );
    cublasErrCheck( cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, 
                            &alphaF, 
                            d_AH2, CUDA_R_32F, K, 
                            d_BL2, CUDA_R_32F, N, 
                            &betaF, 
                            d_C1, CUDA_R_32F, M, 
                            CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );
    cublasErrCheck( cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, 
                            &alphaF, 
                            d_AL2, CUDA_R_32F, K, 
                            d_BH2, CUDA_R_32F, N, 
                            &betaF, 
                            d_C1, CUDA_R_32F, M, 
                            CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );
    cublasErrCheck( cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, 
                            &alphaF, 
                            d_AL2, CUDA_R_32F, K, 
                            d_BL2, CUDA_R_32F, N, 
                            &betaF, 
                            d_C1, CUDA_R_32F, M, 
                            CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP) );

    cudaMemcpy(h_D1, d_C1, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    err = 0;
    for (int i=0; i<M*N; i++) {
        if(err < fabs((double)h_D1[i] - h_D[i]))
            err = fabs((double)h_D1[i] - h_D[i]);
    }
    std::cout << "2TF32 error: \t" << err << std::endl;

    cudaFree(d_AH2);
    cudaFree(d_AL2);
    cudaFree(d_BH2);
    cudaFree(d_BL2);
    cudaFree(d_C1);
    cudaFreeHost(h_AH2);
    cudaFreeHost(h_AL2);
    cudaFreeHost(h_BH2);
    cudaFreeHost(h_BL2);
    cudaFreeHost(h_C1);
    cudaFreeHost(h_D1);

    // destroy handle
    cublasDestroy(handle);

    // Free the memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}