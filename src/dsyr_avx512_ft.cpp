#include "../include/ftblas.h"

void ftblas_dsyr_upp_row(int n, double alpha, double *x, int incx, double *a, int lda)
{

    double *X;
    X = x;

    if (incx != 1)
    {
        X = (double *)malloc(n * sizeof(double));
        ftblas_dcopy(n, x, incx, X, 1);
    }

    int i;
    for (i = 0; i < n; i++)
    {
#ifdef FT_ENABLED
        ftblas_daxpy_ft(n - i, alpha * X[i], X + i, 1, a + (i * lda + i), 1);
#else
        ftblas_daxpy_ori(n - i, alpha * X[i], X + i, 1, a + (i * lda + i), 1);
#endif
    }

    if (incx != 1)
        free(X);
}

void ftblas_dsyr_low_row(int n, double alpha, double *x, int incx, double *a, int lda)
{
    double *X;
    X = x;

    if (incx != 1)
    {
        X = (double *)malloc(n * sizeof(double));
        ftblas_dcopy(n, x, incx, X, 1);
    }

    int i;
    for (i = 0; i < n; i++)
    {
#ifdef FT_ENABLED
        ftblas_daxpy_ft(i + 1, alpha * X[i], X, 1, a + i * lda, 1);
#else
        ftblas_daxpy_ori(i + 1, alpha * X[i], X, 1, a + i * lda, 1);
#endif
    }

    if (incx != 1)
        free(X);
}