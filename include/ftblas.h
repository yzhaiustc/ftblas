/***
 * 
*/

#ifndef _FTBLAS_H_
#define _FTBLAS_H_

#include <math.h>
#include <sys/time.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

void ftblas_daxpy_ori(const long int n, const double alpha, const double *x, const long int inc_x, const double *y, const long int inc_y);
void ftblas_daxpy_ft(const long int n, const double alpha, const double *x, const long int inc_x, const double *y, const long int inc_y);

double ftblas_ddot_ori(const long int n, const double *x, const long int inc_x, const double *y, const long int inc_y);
double ftblas_ddot_ft(const long int n, const double *x, const long int inc_x, const double *y, const long int inc_y);

double ftblas_dnrm2_ori(const long int n, const double *x, const long int inc_x);
double ftblas_dnrm2_ft(const long int n, const double *x, const long int inc_x);

void ftblas_dgemm_ft(int m, int n, int k, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc);
void ftblas_dgemm_ori(int m, int n, int k, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc);

#endif