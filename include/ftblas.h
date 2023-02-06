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

double ftblas_dasum_ori(const long int n, const double *x, const long int inc_x);
double ftblas_dasum_ft(const long int n, const double *x, const long int inc_x);

void ftblas_daxpy_ori(const long int n, const double alpha, const double *x, const long int inc_x, const double *y, const long int inc_y);
void ftblas_daxpy_ft(const long int n, const double alpha, const double *x, const long int inc_x, const double *y, const long int inc_y);

void ftblas_dcopy(const int n, const double *x, const int inc_x, const double *y, const int inc_y);

double ftblas_ddot_ori(const long int n, const double *x, const long int inc_x, const double *y, const long int inc_y);
double ftblas_ddot_ft(const long int n, const double *x, const long int inc_x, const double *y, const long int inc_y);

double ftblas_dnrm2_ori(const long int n, const double *x, const long int inc_x);
double ftblas_dnrm2_ft(const long int n, const double *x, const long int inc_x);

void ftblas_drot_ori(const long int n, const double *x, const long int inc_x, const double *y, const long int inc_y, const double c, const double s);
void ftblas_drot_ft(const long int n, const double *x, const long int inc_x, const double *y, const long int inc_y, const double c, const double s);

void ftblas_drotm_ori(const long int n, const double *x, const long int inc_x, const double *y, const long int inc_y, const double *param);
void ftblas_drotm_ft(const long int n, const double *x, const long int inc_x, const double *y, const long int inc_y, const double *param);

void ftblas_dscal_ori(const long int n, const double alpha, const double *x, const long int inc_x);
void ftblas_dscal_ft(const long int n, const double alpha, const double *x, const long int inc_x);

double ftblas_dsdot_ori(const long int n, const double *x, const long int inc_x, const double *y, const long int inc_y);
double ftblas_dsdot_ft(const long int n, const double *x, const long int inc_x, const double *y, const long int inc_y);

void ftblas_dswap(const int n, const double *x, const int inc_x, const double *y, const int inc_y);

long int ftblas_idamax_ft(const long int n, const double* x, const long int inc_x);
long int ftblas_idamax_ori(const long int n, const double* x, const long int inc_x);

float ftblas_sasum_ft(const long int n, const float *x, const long int inc_x);
float ftblas_sasum_ori(const long int n, const float *x, const long int inc_x);

void ftblas_saxpy_ft(const long int n, const float alpha, const float *x, const long int inc_x, const float *y, const long int inc_y);
void ftblas_saxpy_ori(const long int n, const float alpha, const float *x, const long int inc_x, const float *y, const long int inc_y);

void ftblas_scopy(const int n, const float *x, const int inc_x, const float *y, const int inc_y);

float ftblas_sdot_ft(const long int n, const float *x, const long int inc_x, const float *y, const long int inc_y);
float ftblas_sdot_ori(const long int n, const float *x, const long int inc_x, const float *y, const long int inc_y);

void ftblas_dgemm_ft(int m, int n, int k, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc);
void ftblas_dgemm_ori(int m, int n, int k, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc);

void ftblas_dsymm_ft(int m, int n, int k, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc);
void ftblas_dsymm_ori(int m, int n, int k, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc);

void ftblas_dtrmm_ft(int m, int n, int k, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc);
void ftblas_dtrmm_ori(int m, int n, int k, double alpha, double *A, int lda, double *B, int ldb, double beta, double *C, int ldc);

void ftblas_dtrsm_ft(int m, int n, double alpha, double *A, int lda, double *B, int ldb);
void ftblas_dtrsm_ori(int m, int n, double alpha, double *A, int lda, double *B, int ldb);

void ftblas_dgemv_n_ft(double *A, double *x, double *y, int m, int n, int lda);
void ftblas_dgemv_t_ft(double *A, double *x, double *y, int m, int n, int lda);

void ftblas_dgemv_n_ori(double *A, double *x, double *y, int m, int n, int lda);
void ftblas_dgemv_t_ori(double *A, double *x, double *y, int m, int n, int lda);

void ftblas_dger_ft(long lda, long m, long n, double alpha, double *A, double *x, double *y);
void ftblas_dger_ori(long lda, long m, long n, double alpha, double *A, double *x, double *y);

void ftblas_dtrsv_low_col_ft(double *A, int LDA, double *b, int n);
void ftblas_dtrsv_low_row_ft(double *A, int LDA, double *b, int n);
void ftblas_dtrsv_low_col_ori(double *A, int LDA, double *b, int n);
void ftblas_dtrsv_low_row_ori(double *A, int LDA, double *b, int n);

void ftblas_dsyr_upp_row(int n, double alpha, double *x, int incx, double *a, int lda);
void ftblas_dsyr_low_row(int n, double alpha, double *x, int incx, double *a, int lda);

void ftblas_dsbmv_upp(int n, int k, double alpha, double *a, int lda, double *x, int incx, double beta, double *y, int incy);
void ftblas_dsbmv_low(int n, int k, double alpha, double *a, int lda, double *x, int incx, double beta, double *y, int incy);

void ftblas_dspr_low(int n, double alpha, double *x, int incx, double *ap);
void ftblas_dspr_upp(int n, double alpha, double *x, int incx, double *ap);

void ftblas_dspmv_low(int n, double alpha, double *ap, double *x, int incx, double beta, double *y, int incy);
void ftblas_dspmv_upp(int n, double alpha, double *ap, double *x, int incx, double beta, double *y, int incy);

#endif