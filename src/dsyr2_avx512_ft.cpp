#include "../include/ftblas.h"

void ftblas_dsyr2_low(int n, double alpha, double *x, int incx, double *y, int incy, double *a, int lda)
{
    double *X;
    double *Y;
    X = x;
    Y = y;

    if (incx != 1)
    {
        X = (double *)malloc(n * sizeof(double));
        ftblas_dcopy(n, x, incx, X, 1);
    }

    if (incy != 1)
    {
        Y = (double *)malloc(n * sizeof(double));
        ftblas_dcopy(n, y, incy, Y, 1);
    }

    int i;
    for (i = 0; i < n; i++)
    {
#ifdef FT_ENABLED
        ftblas_daxpy_ft(i + 1, alpha * X[i], Y, 1, a + i * lda, 1);
        ftblas_daxpy_ft(i + 1, alpha * Y[i], X, 1, a + i * lda, 1);
#else
        ftblas_daxpy_ori(i + 1, alpha * X[i], Y, 1, a + i * lda, 1);
        ftblas_daxpy_ori(i + 1, alpha * Y[i], X, 1, a + i * lda, 1);
#endif
    }

    if (incx != 1)
        free(X);

    if (incy != 1)
        free(Y);

}

void ftblas_dsyr2_upp(int n, double alpha, double *x, int incx, double *y, int incy, double *a, int lda)
{

    double *X;
    double *Y;
    X = x;
    X = y;

    if (incx != 1)
    {
        X = (double *)malloc(n * sizeof(double));
        ftblas_dcopy(n, x, incx, X, 1);
    }

    if (incy != 1)
    {
        Y = (double *)malloc(n * sizeof(double));
        ftblas_dcopy(n, y, incy, Y, 1);
    }

    int i;
    for (i = 0; i < n; i++)
    {
#ifdef FT_ENABLED
        ftblas_daxpy_ft(n - i, alpha * X[i], Y + i, 1, a + (i * lda + i), 1);
        ftblas_daxpy_ft(n - i, alpha * Y[i], X + i, 1, a + (i * lda + i), 1);
#else
        ftblas_daxpy_ori(n - i, alpha * X[i], Y + i, 1, a + (i * lda + i), 1);
        ftblas_daxpy_ori(n - i, alpha * Y[i], X + i, 1, a + (i * lda + i), 1);
#endif
    }

    if (incx != 1)
        free(X);
        
    if (incy != 1)
        free(Y);
}
