#include "../include/ftblas.h"

void ftblas_dsyr_upp_row(int n, double alpha, double *x, int incx, double *a, int lda)
{
    int i, j;
    for (i = 0; i < n; i++)
    {
        for (j = i; j < n; j++)
        {
            a[i * lda + j] += alpha * x[i * incx] * x[j * incx];
        }
    }
}

void ftblas_dsyr_low_row(int n, double alpha, double *x, int incx, double *a, int lda)
{
    int i, j;
    for (i = 0; i < n; i++)
    {
        for (j = 0; j <= i; j++)
        {
            a[i * lda + j] += alpha * x[i * incx] * x[j * incx];
        }
    }
}