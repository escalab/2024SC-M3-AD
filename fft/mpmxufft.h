#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>

struct mpmxufftHandle
{
    int N, N_batch;
    int radices[9] = {16, 16, 16, 16, 16, 16, 16, 16, 16};
    int n_radices;
    int mergings[3] = {0, 0, 0};
    int n_mergings;
    void (*layer_0[3])(half2 *, half *, half *);
    void (*layer_1[3])(int, half2 *, half *, half *);
    half *F_real, *F_imag;
    half *F_real_tmp, *F_imag_tmp;
};

void mpmxufftExec(mpmxufftHandle plan, half *data);
void mpmxufftCreate(mpmxufftHandle *plan, int n, int n_batch);
