#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/utils.h"
#include "../include/cblas.h"

void REF_DGEMM(int m, int n, int k, double *a, double *b, double *c);

int main(int argc, char* argv[])
{
    int inc_x = 1;

    double *a, *b, *c, *c_ref;

    double t0, t1;
    double elapsed_time;

    int SIZE[21];
    for (int i = 0; i < 21; i++) {
        SIZE[i] = (int)(16 + i * 16);
    }

    int upper_limit = (sizeof(SIZE) / sizeof(int));
    int max_size = SIZE[upper_limit - 1];
    
    const int TEST_COUNT = 20;
    
    a = (double *)malloc(sizeof(double) * max_size * max_size);
    b = (double *)malloc(sizeof(double) * max_size * max_size);
    c = (double *)malloc(sizeof(double) * max_size * max_size);
    c_ref = (double *)malloc(sizeof(double) * max_size * max_size);
    randomize_matrix(a, max_size, max_size);
    randomize_matrix(b, max_size, max_size);
    randomize_matrix(c, max_size, max_size);
    copy_matrix(c, c_ref, max_size * max_size);
    
    for (int i_count = 0; i_count < upper_limit; i_count++) {
        int m, n, k;
        double alpha, beta;
        m = n = k = SIZE[i_count];
        alpha = 1.;
        beta = 0.;
#ifdef FT_ENABLED
        printf("\nFault-tolerant version:\n");
#else
        printf("\nNon-fault-tolerant version:\n");
#endif
        printf("Testing M = %d:\n",m);
        
        REF_DGEMM(m, n, k, a, b, c_ref);
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, a, m, b, k, beta, c, m);

        bool is_verified = verify_matrix(c_ref, c, m * n);

        if (!is_verified) {
            exit(-1);
        }
        
        t0 = get_sec();
        
        for (int t_count = 0; t_count < TEST_COUNT; t_count++) {
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1., a, m, b, k, 1., c, m);
        }

        t1 = get_sec();
        elapsed_time = t1 - t0;

        printf("Average elasped time: %f second, performance: %f GFLOPS.\n", \
            elapsed_time/TEST_COUNT, 2.*1e-9*TEST_COUNT * m * n * k / elapsed_time);
    }

    free(a);
    free(b);
    free(c);
    free(c_ref);

    return 0;
}

void REF_DGEMM(int m, int n, int k, double *a, double *b, double *c) {
    int lda = m, ldb = k, ldc = m;

#define a(i, j) a[(i) + (j) * lda]
#define b(i, j) b[(i) + (j) * ldb]
#define c(i, j) c[(i) + (j) * ldc]

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            double result = 0.;
            for (int p = 0; p < k; p++) {
                result += a(i, p) * b(p, j);
            }
            c(i, j) = result;
        }
    }

}