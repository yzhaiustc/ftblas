#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include "../include/utils.h"
#include "../include/cblas.h"

void REF_DAXPY(long int n, double alpha, double *x, long int inc_x, double *y, long int inc_y);
void REF_DSCAL(long int n, double alpha, double *x, long int inc_x);

int main(int argc, char* argv[])
{
    int inc_x = 1;
    int inc_y = 1;

    double *vec_x;
    double *vec_y;
    double *vec_x_result;
    double *vec_y_result;
    
    double res_baseline, res;

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
    vec_y = (double *)malloc(sizeof(double) * max_size * 1);
    vec_x_result = (double *)malloc(sizeof(double) * max_size * 1);
    vec_y_result = (double *)malloc(sizeof(double) * max_size * 1);
    
    for (int i_count = 0; i_count < upper_limit; i_count++) {
        int m = SIZE[i_count];
        //test framework has some problems when m is bigger than 1900000
        if(m>1800000) break;

#ifdef FT_ENABLED
        printf("\nFault-tolerant version:\n");
#else
        printf("\nNon-fault-tolerant version:\n");
#endif
        printf("Testing M = %d:\n",m);

        randomize_matrix(vec_x, m, 1);
        randomize_matrix(vec_y, m, 1);
        copy_matrix(vec_x, vec_x_result, m);
        copy_matrix(vec_y, vec_y_result, m);
        
        double alpha = (double)(rand() % 100) + 0.01 * (rand() % 100);
        //REF_DAXPY(m, alpha, vec_x_result, inc_x, vec_y_result, inc_y);
        //cblas_daxpy(m, alpha, vec_x, inc_x, vec_y, inc_y);
        //REF_DSCAL(m, alpha, vec_x_result, inc_x);
        //cblas_dscal(m, alpha, vec_x, inc_x);
       
        //test code here
        res = cblas_dsdot(m, vec_x, inc_x, vec_y, inc_y);
        printf("res = %f\n", res);
        /*
        double diff = 0.0;

        for(int i = 0; i <m; i+=inc_y){
            //diff = std::max(diff, vec_y[i] - vec_y_result[i]);
            diff = std::max(diff, vec_x_result[i] - vec_x[i]);
        }
        
        printf("vec_x[1]=%f, vec_x_result[1]=%f, vec_y[1]=%f, vec_y_result[1]=%f\n",vec_x[1], vec_x_result[1], vec_y[1], vec_y_result[1]);

        if (fabs(diff) > 1e-3) {
            printf("Failed to pass the correctness verification. Exited.\n");
            exit(-1);
        }else{
            printf("Passed the sanity check, start benchmarking the performance.\n");
        }
        
        t0 = get_sec();
        
        for (int t_count = 0; t_count < TEST_COUNT; t_count++) {
            // we dont need the result so we don't take the return val here.
            cblas_daxpy(m, alpha, vec_x, inc_x, vec_y, inc_y);
        }

        t1 = get_sec();
        elapsed_time = t1 - t0;

        printf("Average elasped time: %f second, performance: %f GFLOPS.\n", \
            elapsed_time/TEST_COUNT, 2.*1e-9*TEST_COUNT * m / elapsed_time);
        */
    }
    free(vec_x);
    free(vec_y);
    free(vec_x_result);
    free(vec_y_result);

    return 0;
}

void REF_DAXPY(long int n, double alpha, double *x, long int inc_x, double *y, long int inc_y) {
    for (int cnt_x = 0, cnt_y = 0; cnt_x < n && cnt_y < n; cnt_x += inc_x, cnt_y += inc_y) {
        y[cnt_y] += alpha*x[cnt_x];
    }
}

void REF_DSCAL(long int n, double alpha, double *x, long int inc_x){
    for (int cnt_x = 0; cnt_x < n; cnt_x += inc_x) {
        x[cnt_x] = alpha*x[cnt_x];
    }
}