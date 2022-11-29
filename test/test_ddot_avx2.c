#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/utils.h"
#include "../include/ftblas.h"

int main(int argc, char* argv[])
{
    int m = 20000;

    if (argc != 2) 
    {
        printf("Please input m:\n");
        printf("Wrong input, exit.\n");
        exit(-1);
    }

    m = atoi(argv[1]);

    if (m == 0) {
        printf("vector length cannot be ZERO, return.\n");
        return -1;
    }

//    m = 2 << m;
    printf("m = %d\n", m);

    double *vec_x, *vec_y;
    double *vec_a, *vec_b;
    double *vec_m, *vec_n;
    double t0, t1, tmp1 = 0.0, tmp2 = 0.0, tmp3 = 0.0;

    vec_x = (double *)malloc(sizeof(double) * m * 1);
    vec_y = (double *)malloc(sizeof(double) * m * 1);
    vec_a = (double *)malloc(sizeof(double) * m * 1);
    vec_b = (double *)malloc(sizeof(double) * m * 1);
    vec_m = (double *)malloc(sizeof(double) * m * 1);
    vec_n = (double *)malloc(sizeof(double) * m * 1);

    randomize_matrix(vec_x, m, 1);
    randomize_matrix(vec_y, m, 1);
    copy_matrix(vec_x, vec_a, m);
    copy_matrix(vec_y, vec_b, m);
    copy_matrix(vec_x, vec_m, m);
    copy_matrix(vec_y, vec_n, m);

    print_vector(vec_a,m);
    print_vector(vec_b,m);

    printf("\n################# FT #################\n");
    fflush(stdout);

    t0 = get_sec();
    tmp2 = ft_ddot(m, vec_a, 1, vec_b, 1);
    t1 = get_sec();
    printf("FT-BLAS: Elapsed time: %8.6fs, Perf: %8.6f \n", t1-t0, \
        2 * (1 / 1000.) * (m / 1000.) * (1 / 1000.) / (t1 - t0)); 
    fflush(stdout);


    printf("\n################# CBLAS #################\n");
    fflush(stdout);

    t0 = get_sec();
    tmp1 = cblas_ddot(m, vec_x, 1, vec_y, 1);
    t1 = get_sec();
    printf("CBLAS : Elapsed time: %8.6fs, Perf: %8.6f \n", t1-t0, \
        2 * (1 / 1000.) * (m / 1000.) * (1 / 1000.) / (t1 - t0));    
    fflush(stdout);

    double diff = fabs(tmp1 - tmp2);
    if (diff > 1e-2)
    {
        printf("error!, diff = %f, mkl:%f, ftblas:%f\n", diff,tmp1,tmp2);
    }
    else
    {
        printf("correct!\n");
    }
    

    free(vec_x);
    free(vec_y);
    free(vec_a);
    free(vec_b);
    free(vec_m);
    free(vec_n);

    return 0;
}