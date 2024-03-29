#include "tcfft_f32.h"
using namespace nvcuda;
const int WARP_SIZE = 32, WMMA_M = 16, WMMA_N = 16, WMMA_K = 8, CONT_SIZE = 16;

__device__ inline void
complex_mul(wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> &frag_F_real, wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> &frag_F_imag,
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major> &frag_in_real, wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major> &frag_in_imag,
            wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> &frag_out_real, wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> &frag_out_imag)
{
    wmma::fill_fragment(frag_out_real, 0.0);
    wmma::fill_fragment(frag_out_imag, 0.0);

    wmma::mma_sync(frag_out_real, frag_F_imag, frag_in_imag, frag_out_real);
    for (int i = 0; i < frag_out_real.num_elements; i++)
    {
        frag_out_real.x[i] = -frag_out_real.x[i];
    }
    wmma::mma_sync(frag_out_real, frag_F_real, frag_in_real, frag_out_real);

    wmma::mma_sync(frag_out_imag, frag_F_real, frag_in_imag, frag_out_imag);
    wmma::mma_sync(frag_out_imag, frag_F_imag, frag_in_real, frag_out_imag);
}

__device__ inline void
complex_mul_mpmxu(wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> &frag_F_real,
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major> &frag_in_real, 
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> &frag_out_real)
{
    // for (int i = 0; i<2; i++){
        wmma::fill_fragment(frag_out_real, 0.0);
        wmma::mma_sync(frag_out_real, frag_F_real, frag_in_real, frag_out_real);
        wmma::mma_sync(frag_out_real, frag_F_real, frag_in_real, frag_out_real);
    // }
    
}


// but this is not called ;(
// __device__ inline void complex_mul_acc(wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, float, wmma::row_major> &frag_F_real, wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, float, wmma::row_major> &frag_F_imag,
//                                        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, float, wmma::col_major> &frag_in_real, wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, float, wmma::col_major> &frag_in_imag,
//                                        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> &frag_out_real, wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> &frag_out_imag)
// {
//     wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_buf_real;
//     wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_buf_imag;
//     wmma::fill_fragment(frag_buf_real, 0.0);
//     wmma::fill_fragment(frag_buf_imag, 0.0);

//     //MOD
//     // wmma::mma_sync(frag_buf_real, frag_F_imag, frag_in_imag, frag_buf_real);
//     // for (int i = 0; i < frag_buf_real.num_elements; i++){
//     //     int idx = i % frag_buf_real.num_elements;
//     //     frag_buf_real.x[idx] = -frag_buf_real.x[idx];
//     // }        
//     wmma::mma_sync(frag_buf_real, frag_F_real, frag_in_real, frag_buf_real);
//     for (int i = 0; i < frag_buf_real.num_elements; i++){
//         frag_out_real.x[i] += frag_buf_real.x[i];
//     }
//     wmma::mma_sync(frag_buf_imag, frag_F_imag, frag_in_imag, frag_buf_imag);
//     for (int i = 0; i < frag_buf_imag.num_elements; i++){
//         frag_out_imag.x[i] += frag_buf_imag.x[i];
//     }
        

    // wmma::mma_sync(frag_out_imag, frag_F_real, frag_in_imag, frag_out_imag);
    // wmma::mma_sync(frag_out_imag, frag_F_imag, frag_in_real, frag_out_imag);
// }

__device__ __host__ inline float2 W_N_K(int N, int K)
{
    float2 t = {cosf(2 * M_PI * K / N), -sinf(2 * M_PI * K / N)};
    return t;
}

__device__ __host__ inline float2 W_N_K_fp32(int N, int K)
{
    float2 t = {cosf(2 * M_PI * K / N), -sinf(2 * M_PI * K / N)};
    return t;
}

__device__ inline float2 const cmul(const float2 &a, const float2 &b)
{
    return {a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x};
}

__device__ inline float2 const cmul_mixed(const float2 &a, const float2 &b)
{
    return {a.x * (b.x) - a.y * (b.y), a.x * (b.y) + a.y * (b.x)};
}

__device__ inline void swap(float &a, float &b)
{
    float tmp = a;
    a = b;
    b = tmp;
}



template <int CONT_SIZE, int NUM_WARP>
__global__ void layer_256_0_A100(float2 *in, float *F_real, float *F_imag)
{
    extern __shared__ float2 smem_in[];
    int t_block = threadIdx.x + threadIdx.y * blockDim.x;
    int block_start = blockIdx.x * 256 * CONT_SIZE;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> frag_F_real;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> frag_F_imag;
    wmma::load_matrix_sync(frag_F_real, F_real, 16);
    // wmma::load_matrix_sync(frag_F, F_real, F_imag, 16);
    wmma::load_matrix_sync(frag_F_imag, F_imag, 16);

    int raw_row = threadIdx.x % 4 * 2;
    int raw_col = threadIdx.x / 4;
    // float2 twiddle_unit = W_N_K(256, raw_col);

    /* opt test
    for (int i = 0; i < 256 * CONT_SIZE; i += NUM_WARP * 32)
    {
        int eid = i + t_block;
        smem_in[eid] = in[block_start + eid];
    }
    __syncthreads();

    for (int i = 0; i < 256 * CONT_SIZE; i += NUM_WARP * 32)
    {
        int eid = i + t_block;
        smem_in[eid / 32 * 32 + eid % 32 / 2 + eid % 32 % 2 * 16] = smem_in[eid];
    }
    __syncthreads();
    */

    for (int i = 0; i < 256 * CONT_SIZE; i += NUM_WARP * 16 * 16)
    {
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_out_real;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_out_imag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major> frag_in_real;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major> frag_in_imag;

        int warp_start = i + threadIdx.y * 256;

        for (int j = 0; j < 8; ++j)
        {
            int row = raw_row + j % 4 / 2 * 8 + j % 2;
            int col = raw_col + j / 4 * 8;
            float2 ele = in[block_start + warp_start + row + col * 16];
            // float2 ele = smem_in[warp_start + row + col * 16]; // opt test
            frag_in_real.x[8 + j] = frag_in_real.x[j] = ele.x;
            frag_in_imag.x[8 + j] = frag_in_imag.x[j] = ele.y;
        }

        complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);
        // complex_mul_mpmxu(frag_F_real, frag_in_real, frag_out_real);
        // complex_mul_mpmxu(frag_F_real, frag_in_real, frag_out_real);

        wmma::store_matrix_sync((float *)(smem_in + warp_start), frag_out_real, 16, wmma::mem_row_major);
        wmma::store_matrix_sync((float *)(smem_in + warp_start) + 256, frag_out_imag, 16, wmma::mem_row_major);

        wmma::load_matrix_sync(frag_in_real, (float *)(smem_in + warp_start), 16);
        wmma::load_matrix_sync(frag_in_imag, (float *)(smem_in + warp_start) + 256, 16);

        // float2 twiddle_factor = {1.0, 0};
        // for (int j = 0; j < 16; ++j)
        // {
        //     int row = j;
        //     int col = raw_col;
        //     float2 in_ele = {frag_in_real.x[j], frag_in_imag.x[j]};
        //     in_ele = cmul(in_ele, twiddle_factor);
        //     frag_in_real.x[j] = in_ele.x;
        //     frag_in_imag.x[j] = in_ele.y;
        //     twiddle_factor = cmul(twiddle_factor, twiddle_unit);
        // }
        for (int j = 0; j < 8; ++j)
        {
            int row = raw_row + j % 4 / 2 * 8 + j % 2;
            int col = raw_col + j / 4 * 8;
            float2 in_ele = {frag_in_real.x[j], frag_in_imag.x[j]};
            in_ele = cmul(in_ele, W_N_K(256, row * col));
            frag_in_real.x[8 + j] = frag_in_real.x[j] = in_ele.x;
            frag_in_imag.x[8 + j] = frag_in_imag.x[j] = in_ele.y;
        }

        complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);
        // complex_mul_mpmxu(frag_F_real, frag_in_real, frag_out_real);
        // complex_mul_mpmxu(frag_F_real, frag_in_real, frag_out_real);

        // int raw_row = threadIdx.x / 16 * 4 + threadIdx.x % 8 / 4 * 8 + threadIdx.x % 4;
        // raw_col = threadIdx.x % 16 / 8 * 8;
        for (int j = 0; j < 8; ++j)
        {
            int col = raw_row + j / 4 * 8 + j % 2;
            int row = raw_col + j % 4 / 2 * 8;
            in[block_start + warp_start + row * 16 + col] = {frag_out_real.x[j], frag_out_imag.x[j]};
            // smem_in[warp_start + row * 16 + col] = {frag_out_real.x[j], frag_out_imag.x[j]}; //opt test
        }
    }

    /* opt test
    __syncthreads();
    for (int i = 0; i < 256 * CONT_SIZE; i += NUM_WARP * 32)
    {
        int eid = i + t_block;
        in[block_start + eid] = smem_in[eid];
    }
    */
}



template <int CONT_SIZE, int NUM_WARP>
__global__ void layer_256_1_A100(int step, float2 *in, float *F_real, float *F_imag)
{
    extern __shared__ float2 smem_in[];
    int t_block = threadIdx.x + threadIdx.y * blockDim.x;
    int block_start = blockIdx.y * step * 256 + blockIdx.x * CONT_SIZE;

    // int b_c_col = threadIdx.x / 16 * 4 + threadIdx.x % 16 / 8 * 8 + threadIdx.x % 4;
    // int glb_col = blockIdx.x * CONT_SIZE + threadIdx.y % 2 * 16 + b_c_col;
    // float2 twiddle_unit = W_N_K(step * 16, glb_col);
    int warp_col = blockIdx.x * CONT_SIZE + threadIdx.y % 2 * 16;

    int raw_row = threadIdx.x % 4 * 2;
    int raw_col = threadIdx.x / 4;

    float2 twiddle_factor = {1.0, 0};
    float2 twiddle_unit = W_N_K(step * 16, blockIdx.x * CONT_SIZE + threadIdx.x);
    // for (int i = 0; i < 256 * CONT_SIZE; i += NUM_WARP * 32)
    // {
    //     int eid = i + t_block;
    //     smem_in[eid] = in[block_start + eid / CONT_SIZE * step + eid % CONT_SIZE];
    // }

    for (int i = 0; i < 2; ++i)
    {
        int eid = i * 512 * 8 + threadIdx.y * 512 + threadIdx.x;
        twiddle_factor = {1.0, 0};
        for (int j = 0; j < 16; ++j)
        {
            smem_in[eid] = cmul(in[block_start + eid / CONT_SIZE * step + eid % CONT_SIZE], twiddle_factor);
            eid += 32;
            twiddle_factor = cmul(twiddle_factor, twiddle_unit);
        }
    }

    __syncthreads();

    /* opt test
    for (int i = 0; i < 256 * CONT_SIZE; i += NUM_WARP * 32)
    {
        int eid = i + t_block;
        smem_in[eid / 32 * 32 + eid % 32 / 2 + eid % 32 % 2 * 16] = smem_in[eid];
    }
    __syncthreads();
    */

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> frag_F_real;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> frag_F_imag;
    wmma::load_matrix_sync(frag_F_real, F_real, 16);
    wmma::load_matrix_sync(frag_F_imag, F_imag, 16);

    float2 twiddle[8];
    // for (int j = 0; j < 8; ++j)
    // {
    //     int row = raw_row + j % 4 / 2 * 8 + j % 2;
    //     int col = raw_col + j / 4 * 8;
    //     twiddle[j] = W_N_K(step * 16, (warp_col + col) * row);
    // }

    for (int i_start = 0; i_start < 256 * CONT_SIZE; i_start += NUM_WARP * 256)
    {
        int warp_start = i_start + threadIdx.y / 2 * 512 + threadIdx.y % 2 * 16;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_out_real;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_out_imag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major> frag_in_real;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major> frag_in_imag;

        for (int j = 0; j < 8; ++j)
        {
            int row = raw_row + j % 4 / 2 * 8 + j % 2;
            int col = raw_col + j / 4 * 8;
            int eid = warp_start + row * 32 + col;
            float2 ele = smem_in[eid];
            // ele = cmul(ele, twiddle[j]);
            frag_in_real.x[8 + j] = frag_in_real.x[j] = ele.x;
            frag_in_imag.x[8 + j] = frag_in_imag.x[j] = ele.y;
        }

        // float2 twiddle_factor = {1.0, 0};
        // for (int j = 0; j < 16; ++j)
        // {
        //     int col = b_c_col;
        //     int row = j;
        //     int eid = warp_start + row * 32 + col;
        //     float2 ele = in[block_start + eid / CONT_SIZE * step + eid % CONT_SIZE];
        //     ele = cmul(ele, twiddle_factor);
        //     frag_in_real.x[j] = ele.x;
        //     frag_in_imag.x[j] = ele.y;
        //     twiddle_factor = cmul(twiddle_factor, twiddle_unit);
        // }

        complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);
        // complex_mul_mpmxu(frag_F_real, frag_in_real, frag_out_real);
        // complex_mul_mpmxu(frag_F_real, frag_in_real, frag_out_real);

        for (int j = 0; j < 8; ++j)
        {
            int col = raw_row + j / 4 * 8 + j % 2;
            int row = raw_col + j % 4 / 2 * 8;
            smem_in[warp_start + row * 32 + col] = {frag_out_real.x[j], frag_out_imag.x[j]};
        }
    }

    __syncthreads();

    // warp_col = blockIdx.x * CONT_SIZE + threadIdx.y / 2 * step + threadIdx.y % 2 * 16;
    // for (int j = 0; j < 8; ++j)
    // {
    //     int row = raw_row + j % 4 / 2 * 8 + j % 2;
    //     int col = raw_col + j / 4 * 8;
    //     twiddle[j] = W_N_K(step * 256, warp_col * row + col * row);
    // }
    // float2 twiddle_unit_2[4];
    // for (int j = 0; j < 4; ++j)
    // {
    //     int row = raw_row + j / 2 * 8 + j % 2;
    //     twiddle_unit_2[j] = W_N_K(step * 256, step * 4 * row);
    // }

    for (int i = 0; i < 2; ++i)
    {
        twiddle_unit = W_N_K(step * 256, blockIdx.x * CONT_SIZE + threadIdx.y * step + i * 8 * step + threadIdx.x);
        int eid = i * 32 * 8 + threadIdx.y * 32 + threadIdx.x;
        twiddle_factor = {1.0, 0};
        for (int j = 0; j < 16; ++j)
        {
            smem_in[eid] = cmul(smem_in[eid], twiddle_factor);
            eid += 512;
            twiddle_factor = cmul(twiddle_factor, twiddle_unit);
        }
    }

    __syncthreads();

    for (int i_start = 0; i_start < CONT_SIZE / NUM_WARP; i_start++)
    {
        int warp_start = i_start * NUM_WARP * 16 + threadIdx.y * 16;
        // int glb_col_2 = blockIdx.x * CONT_SIZE + c + threadIdx.y / 2 * step + threadIdx.y % 2 * 16 + b_c_col;
        // float2 twiddle_unit_2 = W_N_K(step * 256, glb_col_2);
        // warp_col = blockIdx.x * CONT_SIZE + i_start * step * 4 + threadIdx.y / 2 * step + threadIdx.y % 2 * 16;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_out_real;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_out_imag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major> frag_in_real;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major> frag_in_imag;
        // float2 twiddle_factor = {1.0, 0};
        // for (int j = 0; j < 16; ++j)
        // {
        //     int col = b_c_col;
        //     int row = j;
        //     float2 ele = smem_in[warp_start + row * 512 + col];
        //     ele = cmul(ele, twiddle_factor);
        //     frag_in_real.x[j] = ele.x;
        //     frag_in_imag.x[j] = ele.y;
        //     twiddle_factor = cmul(twiddle_factor, twiddle_unit_2);
        // }

        for (int j = 0; j < 8; ++j)
        {
            int row = raw_row + j % 4 / 2 * 8 + j % 2;
            int col = raw_col + j / 4 * 8;
            float2 ele = smem_in[warp_start + row * 512 + col];
            // ele = cmul(ele, twiddle[j]);
            frag_in_real.x[8 + j] = frag_in_real.x[j] = ele.x;
            frag_in_imag.x[8 + j] = frag_in_imag.x[j] = ele.y;
            // twiddle[j] = cmul(twiddle[j], twiddle_unit_2[j % 4]);
        }

        complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);
        // complex_mul_mpmxu(frag_F_real, frag_in_real, frag_out_real);
        // complex_mul_mpmxu(frag_F_real, frag_in_real, frag_out_real);

        for (int j = 0; j < 8; ++j)
        {
            int col = raw_row + j / 4 * 8 + j % 2;
            int row = raw_col + j % 4 / 2 * 8;
            smem_in[warp_start + row * 512 + col] = {frag_out_real.x[j], frag_out_imag.x[j]};
        }
    }

    __syncthreads();

    /* opt test
    for (int i = 0; i < 256 * CONT_SIZE; i += NUM_WARP * 32)
    {
        int eid = i + t_block;
        smem_in[eid / 32 * 32 + eid % 32 / 2 + eid % 32 % 2 * 16] = smem_in[eid];
    }
    __syncthreads();
    */

    for (int i = 0; i < 256 * CONT_SIZE; i += NUM_WARP * 32)
    {
        int eid = i + t_block;
        in[block_start + eid / CONT_SIZE * step + eid % CONT_SIZE] = smem_in[eid];
    }
}


template <int CONT_SIZE, int NUM_WARP>
__global__ void layer_512_0_A100(float2 *in, float *F_real, float *F_imag)
{
    extern __shared__ float2 smem_in[];
    int t_block = threadIdx.x + threadIdx.y * blockDim.x;
    int block_start = blockIdx.x * 512 * CONT_SIZE;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> frag_F_real;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> frag_F_imag;
    wmma::load_matrix_sync(frag_F_real, F_real, 16);
    wmma::load_matrix_sync(frag_F_imag, F_imag, 16);

    int raw_row = threadIdx.x % 4 * 2;
    int raw_col = threadIdx.x / 4;
    float2 twiddle_two = W_N_K(512, t_block);

    for (int i = 0; i < 512 * CONT_SIZE; i += NUM_WARP * 16 * 16)
    {
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_out_real;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_out_imag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major> frag_in_real;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major> frag_in_imag;

        int warp_start = i + threadIdx.y * 256;

        for (int j = 0; j < 8; ++j)
        {
            int row = raw_row + j % 4 / 2 * 8 + j % 2;
            int col = raw_col + j / 4 * 8;
            float2 ele = in[block_start + warp_start + row + col * 16];
            frag_in_real.x[8 + j] = frag_in_real.x[j] = ele.x;
            frag_in_imag.x[8 + j] = frag_in_imag.x[j] = ele.y;
        }

        complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);
        // complex_mul_mpmxu(frag_F_real, frag_in_real, frag_out_real);

        wmma::store_matrix_sync((float *)(smem_in + warp_start), frag_out_real, 16, wmma::mem_row_major);
        wmma::store_matrix_sync((float *)(smem_in + warp_start) + 256, frag_out_imag, 16, wmma::mem_row_major);

        wmma::load_matrix_sync(frag_in_real, (float *)(smem_in + warp_start), 16);
        wmma::load_matrix_sync(frag_in_imag, (float *)(smem_in + warp_start) + 256, 16);

        for (int j = 0; j < 8; ++j)
        {
            int row = raw_row + j % 4 / 2 * 8 + j % 2;
            int col = raw_col + j / 4 * 8;
            float2 in_ele = {frag_in_real.x[j], frag_in_imag.x[j]};
            in_ele = cmul(in_ele, W_N_K(256, row * col));
            frag_in_real.x[8 + j] = frag_in_real.x[j] = in_ele.x;
            frag_in_imag.x[8 + j] = frag_in_imag.x[j] = in_ele.y;
        }

        complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);
        // complex_mul_mpmxu(frag_F_real, frag_in_real, frag_out_real);
        // complex_mul_mpmxu(frag_F_real, frag_in_real, frag_out_real);

        for (int j = 0; j < 8; ++j)
        {
            int col = raw_row + j / 4 * 8 + j % 2;
            int row = raw_col + j % 4 / 2 * 8;
            smem_in[warp_start + row * 16 + col] = {frag_out_real.x[j], frag_out_imag.x[j]};
        }
    }

    __syncthreads();
    for (int i = 0; i < 512 * CONT_SIZE; i += NUM_WARP * 32 * 2)
    {
        int eid = i + t_block;
        float2 ele_0 = smem_in[eid];
        float2 ele_1 = cmul(smem_in[eid + 256], twiddle_two);
        // in[block_start + eid] = __fadd2(ele_0, ele_1); 
        in[block_start + eid] = {ele_0.x + ele_1.x, ele_0.y + ele_1.y};
        // in[block_start + eid + 256] = __hsub2(ele_0, ele_1);
        in[block_start + eid + 256] = {ele_0.x - ele_1.x, ele_0.y - ele_1.y};
    }
}



template <int CONT_SIZE, int NUM_WARP>
__global__ void layer_512_1_A100(int step, float2 *in, float *F_real, float *F_imag)
{
    extern __shared__ float2 smem_in[];
    int t_block = threadIdx.x + threadIdx.y * blockDim.x;
    int block_start = blockIdx.y * step * 512 + blockIdx.x * CONT_SIZE;

    // int b_c_col = threadIdx.x / 16 * 4 + threadIdx.x % 16 / 8 * 8 + threadIdx.x % 4;
    // int glb_col = blockIdx.x * CONT_SIZE + b_c_col;
    // float2 twiddle_unit = W_N_K(step * 16, glb_col);

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> frag_F_real;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> frag_F_imag;
    wmma::load_matrix_sync(frag_F_real, F_real, 16);
    wmma::load_matrix_sync(frag_F_imag, F_imag, 16);

    int raw_row = threadIdx.x % 4 * 2;
    int raw_col = threadIdx.x / 4;

    float2 twiddle_factor = {1.0, 0};
    float2 twiddle_unit = W_N_K(step * 16, blockIdx.x * CONT_SIZE + threadIdx.x % 16);

    for (int i = 0; i < 2; ++i)
    {
        int eid = i * 512 * 8 + threadIdx.y * 512 + threadIdx.x / 16 * 256 + threadIdx.x % 16;
        twiddle_factor = {1.0, 0};
        for (int j = 0; j < 16; ++j)
        {
            smem_in[eid] = cmul(in[block_start + eid / CONT_SIZE * step + eid % CONT_SIZE], twiddle_factor);
            eid += 16;
            twiddle_factor = cmul(twiddle_factor, twiddle_unit);
        }
    }

    __syncthreads();

    for (int i_start = 0; i_start < 512 * CONT_SIZE; i_start += NUM_WARP * 256)
    {
        int warp_start = i_start + threadIdx.y * 256;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_out_real;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_out_imag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major> frag_in_real;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major> frag_in_imag;

        for (int j = 0; j < 8; ++j)
        {
            int row = raw_row + j % 4 / 2 * 8 + j % 2;
            int col = raw_col + j / 4 * 8;
            int eid = warp_start + row * 16 + col;
            float2 ele = smem_in[eid];
            // ele = cmul(ele, twiddle[j]);
            frag_in_real.x[8 + j] = frag_in_real.x[j] = ele.x;
            frag_in_imag.x[8 + j] = frag_in_imag.x[j] = ele.y;
        }

        complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);
        // complex_mul_mpmxu(frag_F_real, frag_in_real, frag_out_real);
        // complex_mul_mpmxu(frag_F_real, frag_in_real, frag_out_real);

        for (int j = 0; j < 8; ++j)
        {
            int col = raw_row + j / 4 * 8 + j % 2;
            int row = raw_col + j % 4 / 2 * 8;
            smem_in[warp_start + row * 16 + col] = {frag_out_real.x[j], frag_out_imag.x[j]};
        }
    }

    __syncthreads();

    for (int i = 0; i < 2; ++i)
    {
        twiddle_unit = W_N_K(step * 256, blockIdx.x * CONT_SIZE + threadIdx.y * step * 2 + threadIdx.x / 16 * step + threadIdx.x % 16);
        int eid = i * 16 * 16 * 16 + threadIdx.y * 32 + threadIdx.x;
        twiddle_factor = {1.0, 0};
        for (int j = 0; j < 16; ++j)
        {
            smem_in[eid] = cmul(smem_in[eid], twiddle_factor);
            eid += 256;
            twiddle_factor = cmul(twiddle_factor, twiddle_unit);
        }
    }

    __syncthreads();

    for (int i_start = 0; i_start < 4; i_start++)
    {
        int warp_start = i_start % 2 * NUM_WARP * 16 + i_start / 2 * 256 * CONT_SIZE + threadIdx.y * 16;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_out_real;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_out_imag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major> frag_in_real;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major> frag_in_imag;

        for (int j = 0; j < 8; ++j)
        {
            int row = raw_row + j % 4 / 2 * 8 + j % 2;
            int col = raw_col + j / 4 * 8;
            float2 ele = smem_in[warp_start + row * 256 + col];
            frag_in_real.x[8 + j] = frag_in_real.x[j] = ele.x;
            frag_in_imag.x[8 + j] = frag_in_imag.x[j] = ele.y;
        }

        complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);
        // complex_mul_mpmxu(frag_F_real, frag_in_real, frag_out_real);
        // complex_mul_mpmxu(frag_F_real, frag_in_real, frag_out_real);

        for (int j = 0; j < 8; ++j)
        {
            int col = raw_row + j / 4 * 8 + j % 2;
            int row = raw_col + j % 4 / 2 * 8;
            smem_in[warp_start + row * 256 + col] = {frag_out_real.x[j], frag_out_imag.x[j]};
        }
    }

    __syncthreads();

    float2 twiddle_unit_2 = W_N_K(step * 512, 256 / CONT_SIZE * step);
    twiddle_factor = W_N_K(step * 512, t_block / CONT_SIZE * step + t_block % CONT_SIZE);
    for (int i = 0; i < 256 * CONT_SIZE; i += NUM_WARP * 32)
    {
        int eid = i + t_block;
        float2 ele_0 = smem_in[eid];
        float2 ele_1 = cmul(smem_in[eid + 256 * CONT_SIZE], twiddle_factor);
        // in[block_start + eid / CONT_SIZE * step + eid % CONT_SIZE] = __hadd2(ele_0, ele_1);
        in[block_start + eid / CONT_SIZE * step + eid % CONT_SIZE] = {ele_0.x + ele_1.x, ele_0.y + ele_1.y};
        eid += 256 * CONT_SIZE;
        // in[block_start + eid / CONT_SIZE * step + eid % CONT_SIZE] = __hsub2(ele_0, ele_1);
        in[block_start + eid / CONT_SIZE * step + eid % CONT_SIZE] = {ele_0.x - ele_1.x, ele_0.y - ele_1.y};
        twiddle_factor = cmul(twiddle_factor, twiddle_unit_2);
    }
}



template <int CONT_SIZE, int NUM_WARP>
__global__ void layer_1024_0_A100(float2 *in, float *F_real, float *F_imag)
{
    extern __shared__ float2 smem_in[];
    int t_block = threadIdx.x + threadIdx.y * blockDim.x;
    int block_start = blockIdx.x * 1024 * CONT_SIZE;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> frag_F_real;
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::row_major> frag_F_imag;
    wmma::load_matrix_sync(frag_F_real, F_real, 16);
    wmma::load_matrix_sync(frag_F_imag, F_imag, 16);

    int raw_row = threadIdx.x % 4 * 2;
    int raw_col = threadIdx.x / 4;

    for (int i = 0; i < 1024 * CONT_SIZE; i += NUM_WARP * 16 * 16)
    {
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_out_real;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_out_imag;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major> frag_in_real;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32, wmma::col_major> frag_in_imag;

        int warp_start = i + threadIdx.y * 256;

        for (int j = 0; j < 8; ++j)
        {
            int row = raw_row + j % 4 / 2 * 8 + j % 2;
            int col = raw_col + j / 4 * 8;
            float2 ele = in[block_start + warp_start + row + col * 16];
            frag_in_real.x[8 + j] = frag_in_real.x[j] = ele.x;
            frag_in_imag.x[8 + j] = frag_in_imag.x[j] = ele.y;
        }

        complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);
        // complex_mul_mpmxu(frag_F_real, frag_in_real, frag_out_real);
        // complex_mul_mpmxu(frag_F_real, frag_in_real, frag_out_real);

        wmma::store_matrix_sync((float *)(smem_in + warp_start), frag_out_real, 16, wmma::mem_row_major);
        wmma::store_matrix_sync((float *)(smem_in + warp_start) + 256, frag_out_imag, 16, wmma::mem_row_major);

        wmma::load_matrix_sync(frag_in_real, (float *)(smem_in + warp_start), 16);
        wmma::load_matrix_sync(frag_in_imag, (float *)(smem_in + warp_start) + 256, 16);

        for (int j = 0; j < 8; ++j)
        {
            int row = raw_row + j % 4 / 2 * 8 + j % 2;
            int col = raw_col + j / 4 * 8;
            float2 in_ele = {frag_in_real.x[j], frag_in_imag.x[j]};
            in_ele = cmul(in_ele, W_N_K(256, row * col));
            frag_in_real.x[8 + j] = frag_in_real.x[j] = in_ele.x;
            frag_in_imag.x[8 + j] = frag_in_imag.x[j] = in_ele.y;
        }

        complex_mul(frag_F_real, frag_F_imag, frag_in_real, frag_in_imag, frag_out_real, frag_out_imag);
        // complex_mul_mpmxu(frag_F_real, frag_in_real, frag_out_real);
        // complex_mul_mpmxu(frag_F_real, frag_in_real, frag_out_real);

        for (int j = 0; j < 8; ++j)
        {
            int col = raw_row + j / 4 * 8 + j % 2;
            int row = raw_col + j % 4 / 2 * 8;
            smem_in[warp_start + row * 16 + col] = {frag_out_real.x[j], frag_out_imag.x[j]};
        }
    }

    float2 twiddle_1024_1 = W_N_K(1024, t_block);
    float2 twiddle_1024_2 = cmul(twiddle_1024_1, twiddle_1024_1);
    float2 twiddle_1024_3 = cmul(twiddle_1024_2, twiddle_1024_1);

    __syncthreads();
    for (int i = 0; i < 1024 * CONT_SIZE; i += NUM_WARP * 32 * 4)
    {
        int eid = i + t_block;
        float2 ele0 = smem_in[eid];
        float2 ele1 = cmul(smem_in[eid + 256], twiddle_1024_1);
        float2 ele2 = cmul(smem_in[eid + 512], twiddle_1024_2);
        float2 ele3 = cmul(smem_in[eid + 768], twiddle_1024_3);
        in[block_start + eid] = {ele0.x + ele1.x + ele2.x + ele3.x, ele0.y + ele1.y + ele2.y + ele3.y};
        // in[block_start + eid + 256] = {ele0. + float2({ele1.y, -ele1.x}) - ele2 + float2({-ele3.y, ele3.x}), };
        in[block_start + eid + 256] = {ele0.x + ele1.y - ele2.x -ele3.y, ele0.y - ele1.x - ele2.y + ele3.x};
        in[block_start + eid + 512] = {ele0.x - ele1.x - ele2.x - ele3.x, ele0.y - ele1.y - ele2.y - ele3.y};
        // in[block_start + eid + 768] = ele0 + float2({-ele1.y, ele1.x}) - ele2 + float2({ele3.y, -ele3.x});
        in[block_start + eid + 768] = {ele0.x - ele1.y - ele2.x + ele3.y, ele0.y + ele1.x - ele2.y -ele3.x};
    }
}

void tcfft_f32_Exec(tcfft_f32_Handle plan, float *data)
{
    const int num_warp = 8;
    const int n_cont[3] = {32, 16, 8};

    int step = 1;
    int RADIX = 1;
    dim3 threads, blocks;

    // V100
    switch (plan.mergings[0])
    {
    case 0:
        RADIX = 256;
        break;

    case 1:
        RADIX = 512;
        break;

    case 2:
        RADIX = 1024;
        break;

    default:
        break;
    }
    threads = {32, num_warp};
    cudaFuncSetAttribute(plan.layer_0[plan.mergings[0]], cudaFuncAttributeMaxDynamicSharedMemorySize, RADIX * sizeof(float2) * n_cont[plan.mergings[0]]);
    plan.layer_0[plan.mergings[0]]<<<plan.N * plan.N_batch / n_cont[plan.mergings[0]] / RADIX, threads, RADIX * sizeof(float2) * n_cont[plan.mergings[0]]>>>((float2 *)data, plan.F_real, plan.F_imag);
    step *= RADIX;

    for (int i = 1; i < plan.n_mergings; ++i)
    {
        switch (plan.mergings[i])
        {
        case 0:
            RADIX = 256;
            break;

        case 1:
            RADIX = 512;
            break;

        case 2:
            RADIX = 1024;
            break;

        default:
            break;
        }
        blocks = {step / n_cont[plan.mergings[i]], plan.N_batch * plan.N / step / RADIX};
        cudaFuncSetAttribute(plan.layer_1[plan.mergings[i]], cudaFuncAttributeMaxDynamicSharedMemorySize, RADIX * sizeof(float2) * n_cont[plan.mergings[i]]);
        plan.layer_1[plan.mergings[i]]<<<blocks, threads, RADIX * sizeof(float2) * n_cont[plan.mergings[i]]>>>(step, (float2 *)data, plan.F_real, plan.F_imag);
        step *= RADIX;
    }
}

void tcfft_f32_Create(tcfft_f32_Handle *plan, int n, int n_batch)
{
    plan->N = n;
    plan->N_batch = n_batch;
    // setup functions
    const int num_warp = 8;
    const int n_cont_256 = 32;
    const int n_cont_512 = 16;
    const int n_cont_1024 = 8;
    // printf("mpmxu called\n");
    // plan->layer_0[0] = layer_256_0<n_cont_256, num_warp>;
    // plan->layer_0[1] = layer_512_0<n_cont_512, num_warp>;
    // plan->layer_0[2] = layer_1024_0<n_cont_1024, num_warp>;
    // plan->layer_1[0] = layer_256_1<n_cont_256, num_warp>;
    // plan->layer_1[1] = layer_512_1<n_cont_512, num_warp>;

    plan->layer_0[0] = layer_256_0_A100<n_cont_256, num_warp>;
    plan->layer_0[1] = layer_512_0_A100<n_cont_512, num_warp>;
    plan->layer_0[2] = layer_1024_0_A100<n_cont_1024, num_warp>;
    plan->layer_1[0] = layer_256_1_A100<n_cont_256, num_warp>;
    plan->layer_1[1] = layer_512_1_A100<n_cont_512, num_warp>;
    // radices
    switch (n)
    {
    case 256:
        plan->n_radices = 2;
        plan->n_mergings = 1;
        break;

    case 512:
        plan->n_radices = 3;
        plan->radices[2] = 2;
        plan->n_mergings = 1;
        plan->mergings[0] = 1;
        break;

    case 1024:
        plan->n_radices = 3;
        plan->radices[2] = 4;
        plan->n_mergings = 1;
        plan->mergings[0] = 2;
        break;

    case 131072:
        plan->n_radices = 5;
        plan->radices[2] = 2;
        plan->n_mergings = 2;
        plan->mergings[0] = 1;
        break;

    case 262144:
        plan->n_radices = 6;
        plan->radices[2] = 2;
        plan->radices[5] = 2;
        plan->n_mergings = 2;
        plan->mergings[0] = 1;
        plan->mergings[1] = 1;
        break;

    case 524288:
        plan->n_radices = 6;
        plan->radices[2] = 4;
        plan->radices[5] = 2;
        plan->n_mergings = 2;
        plan->mergings[0] = 2;
        plan->mergings[1] = 1;
        break;

    case 16777216:
        plan->n_radices = 6;
        plan->n_mergings = 3;
        break;

    case 33554432:
        plan->n_radices = 7;
        plan->radices[2] = 2;
        plan->n_mergings = 3;
        plan->mergings[0] = 1;
        break;

    case 67108864:
        plan->n_radices = 8;
        plan->radices[2] = 2;
        plan->radices[5] = 2;
        plan->n_mergings = 3;
        plan->mergings[0] = 1;
        plan->mergings[1] = 1;
        break;

    case 134217728:
        plan->n_radices = 9;
        plan->radices[2] = 2;
        plan->radices[5] = 2;
        plan->radices[8] = 2;
        plan->n_mergings = 3;
        plan->mergings[0] = 1;
        plan->mergings[1] = 1;
        plan->mergings[2] = 1;
        break;

    default:
        break;
    }
    // F
    plan->F_real_tmp = (float *)malloc(sizeof(float) * 256);
    plan->F_imag_tmp = (float *)malloc(sizeof(float) * 256);
#pragma omp parallel for
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 16; ++j)
        {
            plan->F_real_tmp[16 * i + j] = cosf(2 * M_PI * i * j / 16);
            plan->F_imag_tmp[16 * i + j] = -sinf(2 * M_PI * i * j / 16);
        }
    cudaMalloc(&plan->F_real, sizeof(float) * 256);
    cudaMemcpy(plan->F_real, plan->F_real_tmp, sizeof(float) * 256, cudaMemcpyHostToDevice);
    cudaMalloc(&plan->F_imag, sizeof(float) * 256);
    cudaMemcpy(plan->F_imag, plan->F_imag_tmp, sizeof(float) * 256, cudaMemcpyHostToDevice);
}