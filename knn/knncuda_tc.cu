#include <stdio.h>
#include <cuda.h>
// #include <cublas.h>
#include <cublas_v2.h>

#define BLOCK_DIM 16



/**
 * For each reference point (i.e. each column) finds the k-th smallest distances
 * of the distance matrix and their respective indexes and gathers them at the top
 * of the 2 arrays.
 *
 * Since we only need to locate the k smallest distances, sorting the entire array
 * would not be very efficient if k is relatively small. Instead, we perform a
 * simple insertion sort by eventually inserting a given distance in the first
 * k values.
 *
 * @param dist         distance matrix
 * @param dist_pitch   pitch of the distance matrix given in number of columns
 * @param index        index matrix
 * @param index_pitch  pitch of the index matrix given in number of columns
 * @param width        width of the distance matrix and of the index matrix
 * @param height       height of the distance matrix
 * @param k            number of values to find
 */
__global__ void modified_insertion_sort(float * dist,
                                        int     dist_pitch,
                                        int *   index,
                                        int     index_pitch,
                                        int     width,
                                        int     height,
                                        int     k){

    // Column position
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;

    // Do nothing if we are out of bounds
    if (xIndex < width) {

        // Pointer shift
        float * p_dist  = dist  + xIndex;
        int *   p_index = index + xIndex;

        // Initialise the first index
        p_index[0] = 0;

        // Go through all points
        for (int i=1; i<height; ++i) {

            // Store current distance and associated index
            float curr_dist = p_dist[i*dist_pitch];
            int   curr_index  = i;

            // Skip the current value if its index is >= k and if it's higher the k-th slready sorted mallest value
            if (i >= k && curr_dist >= p_dist[(k-1)*dist_pitch]) {
                continue;
            }

            // Shift values (and indexes) higher that the current distance to the right
            int j = min(i, k-1);
            while (j > 0 && p_dist[(j-1)*dist_pitch] > curr_dist) {
                p_dist[j*dist_pitch]   = p_dist[(j-1)*dist_pitch];
                p_index[j*index_pitch] = p_index[(j-1)*index_pitch];
                --j;
            }

            // Write the current distance and index at their position
            p_dist[j*dist_pitch]   = curr_dist;
            p_index[j*index_pitch] = curr_index; 
        }
    }
}



/**
 * Computes the squared norm of each column of the input array.
 *
 * @param array   input array
 * @param width   number of columns of `array` = number of points
 * @param pitch   pitch of `array` in number of columns
 * @param height  number of rows of `array` = dimension of the points
 * @param norm    output array containing the squared norm values
 */
__global__ void compute_squared_norm(float * array, int width, int pitch, int height, float * norm){
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (xIndex<width){
        float sum = 0.f;
        for (int i=0; i<height; i++){
            float val = array[i*pitch+xIndex];
            sum += val*val;
        }
        norm[xIndex] = sum;
    }
}


/**
 * Add the reference points norm (column vector) to each colum of the input array.
 *
 * @param array   input array
 * @param width   number of columns of `array` = number of points
 * @param pitch   pitch of `array` in number of columns
 * @param height  number of rows of `array` = dimension of the points
 * @param norm    reference points norm stored as a column vector
 */
__global__ void add_reference_points_norm(float * array, int width, int pitch, int height, float * norm){
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int xIndex = blockIdx.x * blockDim.x + tx;
    unsigned int yIndex = blockIdx.y * blockDim.y + ty;
    __shared__ float shared_vec[16];
    if (tx==0 && yIndex<height)
        shared_vec[ty] = norm[yIndex];
    __syncthreads();
    if (xIndex<width && yIndex<height)
        array[yIndex*pitch+xIndex] += shared_vec[ty];
}


/**
 * Adds the query points norm (row vector) to the k first lines of the input
 * array and computes the square root of the resulting values.
 *
 * @param array   input array
 * @param width   number of columns of `array` = number of points
 * @param pitch   pitch of `array` in number of columns
 * @param k       number of neighbors to consider
 * @param norm     query points norm stored as a row vector
 */
__global__ void add_query_points_norm_and_sqrt(float * array, int width, int pitch, int k, float * norm){
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    if (xIndex<width && yIndex<k)
        array[yIndex*pitch + xIndex] = sqrt(array[yIndex*pitch + xIndex] + norm[xIndex]);
}


bool knn_cublas(const float * ref,
                int           ref_nb,
                const float * query,
                int           query_nb,
                int           dim, 
                int           k, 
                float *       knn_dist,
                int *         knn_index) {

    // Constants
    const unsigned int size_of_float = sizeof(float);
    const unsigned int size_of_int   = sizeof(int);

    // Return variables
    cudaError_t  err0, err1, err2, err3, err4, err5;

    // Check that we have at least one CUDA device 
    int nb_devices;
    err0 = cudaGetDeviceCount(&nb_devices);
    if (err0 != cudaSuccess || nb_devices == 0) {
        printf("ERROR: No CUDA device found\n");
        return false;
    }

    // Select the first CUDA device as default
    err0 = cudaSetDevice(0);
    if (err0 != cudaSuccess) {
        printf("ERROR: Cannot set the chosen CUDA device\n");
        return false;
    }

    // Initialize CUBLAS
    // cublasInit();
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);
    cublasSetMathMode(cublasHandle, CUBLAS_DEFAULT_MATH);
    // cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));

    // Allocate global memory
    float * ref_dev        = NULL;
    float * query_dev      = NULL;
    float * dist_dev       = NULL;
    int   * index_dev      = NULL;
    float * ref_norm_dev   = NULL;
    float * query_norm_dev = NULL;
    size_t  ref_pitch_in_bytes;
    size_t  query_pitch_in_bytes;
    size_t  dist_pitch_in_bytes;
    size_t  index_pitch_in_bytes;
    err0 = cudaMallocPitch((void**)&ref_dev,   &ref_pitch_in_bytes,   ref_nb   * size_of_float, dim);
    err1 = cudaMallocPitch((void**)&query_dev, &query_pitch_in_bytes, query_nb * size_of_float, dim);
    err2 = cudaMallocPitch((void**)&dist_dev,  &dist_pitch_in_bytes,  query_nb * size_of_float, ref_nb);
    err3 = cudaMallocPitch((void**)&index_dev, &index_pitch_in_bytes, query_nb * size_of_int,   k);
    err4 = cudaMalloc((void**)&ref_norm_dev,   ref_nb   * size_of_float);
    err5 = cudaMalloc((void**)&query_norm_dev, query_nb * size_of_float);
    if (err0 != cudaSuccess || err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess || err4 != cudaSuccess || err5 != cudaSuccess) {
        printf("ERROR: Memory allocation error\n");
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        cudaFree(ref_norm_dev);
        cudaFree(query_norm_dev);
        // cublasShutdown();
        return false;
    }

    // Deduce pitch values
    size_t ref_pitch   = ref_pitch_in_bytes   / size_of_float;
    size_t query_pitch = query_pitch_in_bytes / size_of_float;
    size_t dist_pitch  = dist_pitch_in_bytes  / size_of_float;
    size_t index_pitch = index_pitch_in_bytes / size_of_int;

    // Check pitch values
    if (query_pitch != dist_pitch || query_pitch != index_pitch) {
        printf("ERROR: Invalid pitch value\n");
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        cudaFree(ref_norm_dev);
        cudaFree(query_norm_dev);
        // cublasShutdown();
        return false; 
    }

    // Copy reference and query data from the host to the device
    err0 = cudaMemcpy2D(ref_dev,   ref_pitch_in_bytes,   ref,   ref_nb * size_of_float,   ref_nb * size_of_float,   dim, cudaMemcpyHostToDevice);
    err1 = cudaMemcpy2D(query_dev, query_pitch_in_bytes, query, query_nb * size_of_float, query_nb * size_of_float, dim, cudaMemcpyHostToDevice);
    if (err0 != cudaSuccess || err1 != cudaSuccess) {
        printf("ERROR: Unable to copy data from host to device\n");
        printf("%s\n", cudaGetErrorString(err0));
        printf("%s\n", cudaGetErrorString(err1));
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        cudaFree(ref_norm_dev);
        cudaFree(query_norm_dev);
        // cublasShutdown();
        return false; 
    }

    // Compute the squared norm of the reference points
    dim3 block0(256, 1, 1);
    dim3 grid0(ref_nb / 256, 1, 1);
    if (ref_nb % 256 != 0) grid0.x += 1;
    compute_squared_norm<<<grid0, block0>>>(ref_dev, ref_nb, ref_pitch, dim, ref_norm_dev);
    if (cudaGetLastError() != cudaSuccess) {
        printf("ERROR: Unable to execute kernel\n");
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        cudaFree(ref_norm_dev);
        cudaFree(query_norm_dev);
        // cublasShutdown();
        return false;
    }

    // Compute the squared norm of the query points
    dim3 block1(256, 1, 1);
    dim3 grid1(query_nb / 256, 1, 1);
    if (query_nb % 256 != 0) grid1.x += 1;
    compute_squared_norm<<<grid1, block1>>>(query_dev, query_nb, query_pitch, dim, query_norm_dev);
    if (cudaGetLastError() != cudaSuccess) {
        printf("ERROR: Unable to execute kernel\n");
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        cudaFree(ref_norm_dev);
        cudaFree(query_norm_dev);
        // cublasShutdown();
        return false;
    }

    // Computation of query*transpose(reference)
    // cublasSgemm('n', 't', (int)query_pitch, (int)ref_pitch, dim, (float)-2.0, query_dev, query_pitch, ref_dev, ref_pitch, (float)0.0, dist_dev, query_pitch);
    float alpha = -2.0f;
    float beta = 0.0f;
    // cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, 
    //     (int)query_pitch, (int)ref_pitch, dim, 
    //     &alpha,
    //     query_dev, query_pitch,
    //     ref_dev, ref_pitch,
    //     &beta, 
    //     dist_dev,query_pitch);

    cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, 
        (int)query_pitch, (int)ref_pitch, (int)dim, 
        &alpha,
        query_dev, CUDA_R_32F,query_pitch,
        ref_dev, CUDA_R_32F,ref_pitch,
        &beta,
        dist_dev, CUDA_R_32F, query_pitch,
        CUBLAS_COMPUTE_32F_FAST_TF32,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP 
    );
    cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, 
        (int)query_pitch, (int)ref_pitch, (int)dim, 
        &alpha,
        query_dev, CUDA_R_32F,query_pitch,
        ref_dev, CUDA_R_32F,ref_pitch,
        &beta,
        dist_dev, CUDA_R_32F, query_pitch,
        CUBLAS_COMPUTE_32F_FAST_TF32,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP 
    );
    if (cudaGetLastError() != cudaSuccess) {
        printf("ERROR: Unable to execute kernel\n");
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        cudaFree(ref_norm_dev);
        cudaFree(query_norm_dev);
        cublasDestroy(cublasHandle);
        // cublasShutdown();
        return false;
    }

    // Add reference points norm
    dim3 block2(16, 16, 1);
    dim3 grid2(query_nb / 16, ref_nb / 16, 1);
    if (query_nb % 16 != 0) grid2.x += 1;
    if (ref_nb   % 16 != 0) grid2.y += 1;
    add_reference_points_norm<<<grid2, block2>>>(dist_dev, query_nb, dist_pitch, ref_nb, ref_norm_dev);
    if (cudaGetLastError() != cudaSuccess) {
        printf("ERROR: Unable to execute kernel\n");
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        cudaFree(ref_norm_dev);
        cudaFree(query_norm_dev);
        cublasDestroy(cublasHandle);
        // cublasShutdown();
        return false;
    }

    // Sort each column
    modified_insertion_sort<<<grid1, block1>>>(dist_dev, dist_pitch, index_dev, index_pitch, query_nb, ref_nb, k);
    if (cudaGetLastError() != cudaSuccess) {
        printf("ERROR: Unable to execute kernel\n");
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        cudaFree(ref_norm_dev);
        cudaFree(query_norm_dev);
        cublasDestroy(cublasHandle);
        // cublasShutdown();
        return false;
    }

    // Add query norm and compute the square root of the of the k first elements
    dim3 block3(16, 16, 1);
    dim3 grid3(query_nb / 16, k / 16, 1);
    if (query_nb % 16 != 0) grid3.x += 1;
    if (k        % 16 != 0) grid3.y += 1;
    add_query_points_norm_and_sqrt<<<grid3, block3>>>(dist_dev, query_nb, dist_pitch, k, query_norm_dev);
    if (cudaGetLastError() != cudaSuccess) {
        printf("ERROR: Unable to execute kernel\n");
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        cudaFree(ref_norm_dev);
        cudaFree(query_norm_dev);
        cublasDestroy(cublasHandle);
        // cublasShutdown();
        return false;
    }

    // Copy k smallest distances / indexes from the device to the host
    err0 = cudaMemcpy2D(knn_dist,  query_nb * size_of_float, dist_dev,  dist_pitch_in_bytes,  query_nb * size_of_float, k, cudaMemcpyDeviceToHost);
    err1 = cudaMemcpy2D(knn_index, query_nb * size_of_int,   index_dev, index_pitch_in_bytes, query_nb * size_of_int,   k, cudaMemcpyDeviceToHost);
    if (err0 != cudaSuccess || err1 != cudaSuccess) {
        printf("ERROR: Unable to copy data from device to host\n");
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        cudaFree(ref_norm_dev);
        cudaFree(query_norm_dev);
        cublasDestroy(cublasHandle);
        // cublasShutdown();
        return false; 
    }

    // Memory clean-up and CUBLAS shutdown
    cudaFree(ref_dev);
    cudaFree(query_dev);
    cudaFree(dist_dev);
    cudaFree(index_dev);
    cudaFree(ref_norm_dev);
    cudaFree(query_norm_dev);
    cublasDestroy(cublasHandle);
    // cublasShutdown();

    return true;
}
