#include "../include/ftblas.h"

void ftblas_dspr2_low(int n, double alpha, double *x, int incx, double *y, int incy, double *ap)
{
    double *X;
    double *Y;
    double *AP;
    X = x;
    Y = y;
    AP = ap;

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
        ftblas_daxpy_ft(i + 1, alpha * X[i], Y, 1, AP, 1);
        ftblas_daxpy_ft(i + 1, alpha * Y[i], X, 1, AP, 1);
#else
        ftblas_daxpy_ori(i + 1, alpha * X[i], Y, 1, AP, 1);
        ftblas_daxpy_ori(i + 1, alpha * Y[i], X, 1, AP, 1);
#endif
        AP += i + 1;
    }

    if (incx != 1)
        free(X);

    if (incy != 1)
        free(Y);
}

void ftblas_dspr2_upp(int n, double alpha, double *x, int incx, double *y, int incy, double *ap)
{
    double *X;
    double *Y;
    double *AP;
    X = x;
    Y = y;
    AP = ap;

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
        ftblas_daxpy_ft(n - i, alpha * X[i], Y + i, 1, AP, 1);
        ftblas_daxpy_ft(n - i, alpha * Y[i], X + i, 1, AP, 1);
#else
        ftblas_daxpy_ori(n - i, alpha * X[i], Y + i, 1, AP, 1);
        ftblas_daxpy_ori(n - i, alpha * Y[i], X + i, 1, AP, 1);
#endif
        AP += n - i;
    }

    if (incx != 1)
        free(X);

    if (incy != 1)
        free(Y);
}