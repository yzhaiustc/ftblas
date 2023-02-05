#include "../include/ftblas.h"

void ftblas_dspmv_low(int n, double alpha, double *ap, double *x, int incx, double beta, double *y, int incy)
{
    double *X;
    double *Y;
    double *AP;
    X = x;
    X = y;
    AP = ap;

    if (incx != 1)
    {
        X = (double *)malloc(n * sizeof(double));
        ftblas_dcopy(n, x, incx, X, 1);
    }

    if (incy != 1)
    {
        Y = (double *)malloc(n * sizeof(double));
        ftblas_dcopy(n, y, incx, Y, 1);
    }

    int i, length;
    for (i = 0; i < n; i++)
    {
#ifdef FT_ENABLED
        if (i > 0)  Y[i] += alpha * ftblas_ddot_ft(i, AP, 1, X, 1);
#else
        if (i > 0)  Y[i] += alpha * ftblas_ddot_ori(i, AP, 1, X, 1);
#endif

#ifdef FT_ENABLED
        ftblas_daxpy_ft(i + 1, alpha * X[i], AP, 1, Y, 1);
#else
        ftblas_daxpy_ori(i + 1, alpha * X[i], AP, 1, Y, 1);
#endif

        AP += i + 1;
    }

    if (incx != 1)
        free(X);

    if (incy != 1)
        free(Y);
}

void ftblas_dspmv_upp(int n, double alpha, double *ap, double *x, int incx, double beta, double *y, int incy)
{
    double *X;
    double *Y;
    double *AP;
    X = x;
    X = y;
    AP = ap;

    if (incx != 1)
    {
        X = (double *)malloc(n * sizeof(double));
        ftblas_dcopy(n, x, incx, X, 1);
    }

    if (incy != 1)
    {
        Y = (double *)malloc(n * sizeof(double));
        ftblas_dcopy(n, y, incx, Y, 1);
    }

    int i, length;
    for (i = 0; i < n; i++)
    {

#ifdef FT_ENABLED
        Y[i] += alpha * ftblas_ddot_ft(n - i, AP + i, 1, X + i, 1);
#else
        Y[i] += alpha * ftblas_ddot_ori(n - i, AP + i, 1, X + i, 1);
#endif

#ifdef FT_ENABLED
        if (n - i > 1)
            ftblas_daxpy_ft(n - i - 1, alpha * X[i], AP + i + 1, 1, Y + i + 1, 1);
#else
        if (n - i > 1)
            ftblas_daxpy_ori(n - i - 1, alpha * X[i], AP + i + 1, 1, Y + i + 1, 1);
#endif

        AP += n - i - 1;
    }

    if (incx != 1)
        free(X);

    if (incy != 1)
        free(Y);
}