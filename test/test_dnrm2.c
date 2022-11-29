#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/utils.h"
#include "../include/ftblas.h"


int main(int argc, char* argv[])
{
    int m = 20000;
    int inc_x = 1;
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

    if (inc_x <= 0) {
        printf("increment of vector should be positive, return.\n");
        return -1;
    }

    double *vec_x, *vec_y;
    double *vec_a, *vec_b;
    double *vec_m, *vec_n;
    double t0, t1, res_ft, res_ori;

    vec_x = (double *)malloc(sizeof(double) * m * 1);
    vec_y = (double *)malloc(sizeof(double) * m * 1);
    vec_a = (double *)malloc(sizeof(double) * m * 1);
    vec_b = (double *)malloc(sizeof(double) * m * 1);
    vec_m = (double *)malloc(sizeof(double) * m * 1);
    vec_n = (double *)malloc(sizeof(double) * m * 1);

    randomize_matrix(vec_x, m, 1);
    //randomize_matrix(vec_y, m, 1);
    //copy_matrix(vec_y, vec_n, m);
    //print_vector(vec_x, m);
/*
    printf("\n################# BLAS #################\n");
    fflush(stdout);
    t0 = get_sec();
    res_ori = cblas_dnrm2(m, vec_x, inc_x);
    t1 = get_sec();
    printf("CBLAS : Elapsed time: %8.6fs, Perf: %8.6f \n", t1-t0, \
        2 * (1 / 1000. / inc_x) * (m / 1000.) * (1 / 1000.) / (t1 - t0)) ;    
    fflush(stdout);
*/
    printf("\n################# FT #################\n");
    fflush(stdout);
    t0 = get_sec();
    res_ft = ft_dnrm2(m, vec_x, inc_x);
    t1 = get_sec();
    printf("ft-blas : Elapsed time: %8.6fs, Perf: %8.6f \n", t1-t0, \
        2 * (1 / 1000. / inc_x) * (m / 1000.) * (1 / 1000.) / (t1 - t0)); 
    fflush(stdout);

    //print_vector(vec_x, m);


    //print_vector(vec_a, m);

    

    free(vec_x);
    free(vec_y);
    free(vec_a);
    free(vec_b);
    free(vec_m);
    free(vec_n);

    return 0;
}