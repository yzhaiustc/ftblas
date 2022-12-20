#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/utils.h"
#include "../include/cblas.h"

double REF_DDOT(long int n, double *x, long int inc_x, double *y, long int inc_y);

double REF(long int n, double *x, long int inc_x){
    double res =0.0;
    for (int cnt_x = 0; cnt_x < n; cnt_x += inc_x) {
        res += fabs(x[cnt_x]);
    }
    return res;
}

int main(int argc, char* argv[])
{
    int inc_x = 1;

    double *vec_x;
    double *vec_y;
    
    double res_baseline, res_ori;

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
    randomize_matrix(vec_x, max_size, 1);
    randomize_matrix(vec_y, max_size, 1);
    
    for (int i_count = 0; i_count < upper_limit; i_count++) {
        int m = SIZE[i_count];

#ifdef FT_ENABLED
        printf("\nFault-tolerant version:\n");
#else
        printf("\nNon-fault-tolerant version:\n");
#endif
        printf("Testing M = %d:\n",m);
        
        /*
        //test for ddot
        res_baseline = REF_DDOT(m, vec_x, inc_x, vec_y, inc_x);
        res_ori = cblas_ddot(m, vec_x, inc_x, vec_y, inc_x);
        */

        //test if the function can run and accuracy
        //dasum segmentfalut
        
        /*
        //daxpy, ddot etc test code

        //use for daxpy test, remember to delete when push
        int inc_y = 1;
        printf("test run\n");
        double alpha = 2.0;
        printf("before computing x[1]=%f\n, y[1]=%f\n", vec_x[999999], vec_y[999999]);
        cblas_daxpy(m, alpha, vec_x, inc_x, vec_y, inc_y);
        printf("after computing x[1]=%f\n, y[1]=%f\n", vec_x[999999], vec_y[999999]);
        REF(m, alpha, vec_x, inc_x, vec_y, inc_y);
        printf("after baseline computing x[1]=%f\n, y[1]=%f\n", vec_x[999999], vec_y[999999]);
        */
        
        res_baseline = REF(m, vec_x, inc_x);
        res_ori = cblas_dasum(m, vec_x, inc_x);
        

        //test code end here

        double diff = res_baseline - res_ori;
        printf("res_baseline=%f, res_ori=%f\n",res_baseline, res_ori);

        if (fabs(diff) > 1e-3) {
            printf("Failed to pass the correctness verification. Exited.\n");
            exit(-1);
        }else{
            printf("Passed the sanity check, start benchmarking the performance.\n");
        }
        
        t0 = get_sec();
        
        for (int t_count = 0; t_count < TEST_COUNT; t_count++) {
            // we dont need the result so we don't take the return val here.
            cblas_ddot(m, vec_x, inc_x, vec_y, inc_x);
        }

        t1 = get_sec();
        elapsed_time = t1 - t0;

        printf("Average elasped time: %f second, performance: %f GFLOPS.\n", \
            elapsed_time/TEST_COUNT, 2.*1e-9*TEST_COUNT * m / elapsed_time);
    }

    free(vec_x);
    free(vec_y);

    return 0;
}

double REF_DDOT(long int n, double *x, long int inc_x, double *y, long int inc_y) {
    double result = 0.;
    for (int cnt_x = 0, cnt_y = 0; cnt_x < n && cnt_y < n; cnt_x += inc_x, cnt_y += inc_y) {
        result += (x[cnt_x] * y[cnt_y]);
    }
    return result;
}