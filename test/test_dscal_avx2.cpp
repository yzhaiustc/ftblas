#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/utils.h"
#include "../include/ftblas.h"

// linking with Intel oneMKL
#include "mkl.h"

int main(int argc, char* argv[])
{
    int inc_x = 1;

    double *vec_x, *vec_x_result;
    double a = 2.0;
    double t0, t1;
    double elapsed_time;

    int SIZE[21];
    for (int i = 0; i < 21; i++) {
        SIZE[i] = (int)(1e6 + i * 1e5);
    }

    int upper_limit = (sizeof(SIZE) / sizeof(int));
    int max_size = SIZE[upper_limit - 1];
    const int TEST_COUNT = 20;
    vec_x = (double *)malloc(sizeof(double) * max_size * 1);
    vec_x_result = (double *)malloc(sizeof(double) * max_size * 1);
    
    for (int i_count = 0; i_count < upper_limit; i_count++) {
        int m = SIZE[i_count];
        
        randomize_matrix(vec_x, max_size, 1);
        //Initialization x_result with x
        for(int i = 0; i <m; i++){
            vec_x_result[i] = vec_x[i];
        }

        printf("\nTesting M = %d:\n",m);

        cblas_dscal(m, a, vec_x_result, inc_x);
        ori_dscal(m, a, vec_x, inc_x);

        double threshold = 1e-3;

        for(int i = 0; i < m; i++){
            if(fabs(vec_x[i] - vec_x_result[i]) >= threshold){
                printf("Failed to pass the correctness verification against Intel oneMKL. Exited.\n");
                exit(-1);
            }
        }
        printf("Passed the sanity check, start benchmarking the performance.\n");
        
        t0 = get_sec();
        
        for (int t_count = 0; t_count < TEST_COUNT; t_count++) {
            // we dont need the result so we don't take the return val here.
            ori_dscal(m, a, vec_x, inc_x);
        }

        t1 = get_sec();
        elapsed_time = t1 - t0;

        printf("Average elasped time: %f second, performance: %f GFLOPS.\n", \
            elapsed_time/TEST_COUNT, 2.*1e-9*TEST_COUNT * m / elapsed_time);
    }

    free(vec_x);

    return 0;
}