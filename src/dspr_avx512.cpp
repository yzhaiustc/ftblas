#include "../include/ftblas.h"

void ftblas_dspr_low(int n, double alpha, double *x, int incx, double *ap)
{
    double *X;
    double *AP;
    X = x;
    AP = ap;

    if (incx != 1)
    {
        X = (double *)malloc(n * sizeof(double));
        ftblas_dcopy(n, x, incx, X, 1);
    }

    int i;
    for (i = 0; i < n; i++)
    {
#ifdef FT_ENABLED
        ftblas_daxpy_ft(i + 1, alpha * X[i], X, 1, AP, 1);
#else
        ftblas_daxpy_ori(i + 1, alpha * X[i], X, 1, AP, 1);
#endif
        AP += i + 1;
    }

    if (incx != 1)
        free(X);
}

void ftblas_dspr_upp(int n, double alpha, double *x, int incx, double *ap)
{
    double *X;
    double *AP;
    X = x;
    AP = ap;

    if (incx != 1)
    {
        X = (double *)malloc(n * sizeof(double));
        ftblas_dcopy(n, x, incx, X, 1);
    }

    int i;
    for (i = 0; i < n; i++)
    {
#ifdef FT_ENABLED
        ftblas_daxpy_ft(n - i, alpha * X[i], X + i, 1, AP, 1);
#else
        ftblas_daxpy_ori(n - i, alpha * X[i], X + i, 1, AP, 1);
#endif
        AP += n - i;
    }

    if (incx != 1)
        free(X);
}