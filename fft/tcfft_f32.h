#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>

struct tcfft_f32_Handle
{
    int N, N_batch;
    int radices[9] = {16, 16, 16, 16, 16, 16, 16, 16, 16};
    int n_radices;
    int mergings[3] = {0, 0, 0};
    int n_mergings;
    void (*layer_0[3])(float2 *, float *, float *);
    void (*layer_1[3])(int, float2 *, float *, float *);
    float *F_real, *F_imag;
    float *F_real_tmp, *F_imag_tmp;
};

void tcfft_f32_Exec(tcfft_f32_Handle plan, float *data);
void tcfft_f32_Create(tcfft_f32_Handle *plan, int n, int n_batch);
