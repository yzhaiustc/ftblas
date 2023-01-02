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

double ftblas_ddot_ori(const long int n, const double *x, const long int inc_x, const double *y, const long int inc_y);
double ftblas_ddot_ft(const long int n, const double *x, const long int inc_x, const double *y, const long int inc_y);

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

#endif