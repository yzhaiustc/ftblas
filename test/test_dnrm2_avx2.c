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

    double *vec_x;
    double t0, t1, res_baseline, res_ori;
    double elapsed_time;

    int SIZE[21];
    for (int i = 0; i < 21; i++) {
        SIZE[i] = (int)(1e6 + i * 1e5);
    }

    int upper_limit = (sizeof(SIZE) / sizeof(int));
    int max_size = SIZE[upper_limit - 1];
    const int TEST_COUNT = 20;
    vec_x = (double *)malloc(sizeof(double) * max_size * 1);
    randomize_matrix(vec_x, max_size, 1);
    
    for (int i_count = 0; i_count < upper_limit; i_count++) {
        int m = SIZE[i_count];
        
        printf("\nTesting M = %d:\n",m);
        
        res_baseline = cblas_dnrm2(m, vec_x, inc_x);
        res_ori = ori_dnrm2(m, vec_x, inc_x);

        double diff = res_baseline - res_ori;
        
        if (fabs(diff) > 1e-3) {
            printf("Failed to pass the correctness verification against Intel oneMKL. Exited.\n");
            exit(-1);
        }else{
            printf("Passed the sanity check, start benchmarking the performance.\n");
        }
        
        t0 = get_sec();
        
        for (int t_count = 0; t_count < TEST_COUNT; t_count++) {
            // we dont need the result so we don't take the return val here.
            ori_dnrm2(m, vec_x, inc_x);
        }

        t1 = get_sec();
        elapsed_time = t1 - t0;

        printf("Average elasped time: %f second, performance: %f GFLOPS.\n", \
            elapsed_time/TEST_COUNT, 2.*1e-9*TEST_COUNT * m / elapsed_time);
    }

    free(vec_x);

    return 0;
}