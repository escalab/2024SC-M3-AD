#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <iostream>

#include "knncuda_simt.h"


/**
 * Initializes randomly the reference and query points.
 *
 * @param ref        refence points
 * @param ref_nb     number of reference points
 * @param query      query points
 * @param query_nb   number of query points
 * @param dim        dimension of points
 */
void initialize_data(float * ref,
                     int     ref_nb,
                     float * query,
                     int     query_nb,
                     int     dim) {

    // Initialize random number generator
    // srand(time(NULL));
    srand(1024);

    // Generate random reference points
    for (int i=0; i<ref_nb*dim; ++i) {
        ref[i] = 10. * (float)(rand() / (double)RAND_MAX);
    }

    // Generate random query points
    for (int i=0; i<query_nb*dim; ++i) {
        query[i] = 10. * (float)(rand() / (double)RAND_MAX);
        // printf("query %d: %f\n", i, query[i]);
    }
}



/**
 * Test an input k-NN function implementation by verifying that its output
 * results (distances and corresponding indexes) are similar to the expected
 * results (ground truth).
 *
 * Since the k-NN computation might end-up in slightly different results
 * compared to the expected one depending on the considered implementation,
 * the verification consists in making sure that the accuracy is high enough.
 *
 * The tested function is ran several times in order to have a better estimate
 * of the processing time.
 *
 * @param ref            reference points
 * @param ref_nb         number of reference points
 * @param query          query points
 * @param query_nb       number of query points
 * @param dim            dimension of reference and query points
 * @param k              number of neighbors to consider
 * @param gt_knn_dist    ground truth distances
 * @param gt_knn_index   ground truth indexes
 * @param knn            function to test
 * @param name           name of the function to test (for display purpose)
 * @param nb_iterations  number of iterations
 * return false in case of problem, true otherwise
 */
bool test(const float * ref,
          int           ref_nb,
          const float * query,
          int           query_nb,
          int           dim,
          int           k,
          float *       gt_knn_dist,
          int *         gt_knn_index,
          bool (*knn)(const float *, int, const float *, int, int, int, float *, int *),
          const char *  name,
          int           nb_iterations) {

    // Parameters
    const float precision    = 0.001f; // distance error max
    const float min_accuracy = 0.999f; // percentage of correct values required

    // Allocate memory for computed k-NN neighbors
    float * test_knn_dist  = (float*) malloc(query_nb * k * sizeof(float));
    int   * test_knn_index = (int*)   malloc(query_nb * k * sizeof(int));

    // Allocation check
    if (!test_knn_dist || !test_knn_index) {
        printf("ALLOCATION ERROR\n");
        free(test_knn_dist);
        free(test_knn_index);
        return false;
    }

    // Start timer
    struct timeval tic;
    gettimeofday(&tic, NULL);

    // Compute k-NN several times
    for (int i=0; i<nb_iterations; ++i) {
        if (!knn(ref, ref_nb, query, query_nb, dim, k, test_knn_dist, test_knn_index)) {
            free(test_knn_dist);
            free(test_knn_index);
            return false;
        }
    }

    // Stop timer
    struct timeval toc;
    gettimeofday(&toc, NULL);

    // Elapsed time in ms
    double elapsed_time = toc.tv_sec - tic.tv_sec;
    elapsed_time += (toc.tv_usec - tic.tv_usec) / 1000000.;

    if  (nb_iterations > 1){
        std::cout << ref_nb << "," << query_nb << "," << dim <<"," << k << "," << (elapsed_time/nb_iterations)*1000 << "\n";
    }

    // Free memory
    free(test_knn_dist);
    free(test_knn_index);

    return true;
}


/**
 * 1. Create the synthetic data (reference and query points).
 * 2. Compute the ground truth.
 * 3. Test the different implementation of the k-NN algorithm.
 */
int main(int argc, char **argv) {

    if (argc < 5){
        printf("usage: ./test ref_nb query_nb dim k\n");
        exit(0);
    }
    const int ref_nb   = atoi(argv[1]);
    const int query_nb = atoi(argv[2]);
    const int dim      = atoi(argv[3]);
    const int k        = atoi(argv[4]);

    // Display


    // Sanity check
    if (ref_nb<k) {
        printf("Error: k value is larger that the number of reference points\n");
        return EXIT_FAILURE;
    }

    // Allocate input points and output k-NN distances / indexes
    float * ref        = (float*) malloc(ref_nb   * dim * sizeof(float));
    float * query      = (float*) malloc(query_nb * dim * sizeof(float));
    float * knn_dist   = (float*) malloc(query_nb * k   * sizeof(float));
    int   * knn_index  = (int*)   malloc(query_nb * k   * sizeof(int));

    // Allocation checks
    if (!ref || !query || !knn_dist || !knn_index) {
        printf("Error: Memory allocation error\n"); 
        free(ref);
	    free(query);
	    free(knn_dist);
	    free(knn_index);
        return EXIT_FAILURE;
    }

    // Initialize reference and query points with random values
    initialize_data(ref, ref_nb, query, query_nb, dim);
    test(ref, ref_nb, query, query_nb, dim, k, knn_dist, knn_index, &knn_cublas,       "knn_cublas",       1);

    double t;
    // Test all k-NN functions
    // printf("warmup(ignore)\n");
    test(ref, ref_nb, query, query_nb, dim, k, knn_dist, knn_index, &knn_cublas,       "knn_cublas",       100);

    // Deallocate memory 
    free(ref);
    free(query);
    free(knn_dist);
    free(knn_index);

    return EXIT_SUCCESS;
}
