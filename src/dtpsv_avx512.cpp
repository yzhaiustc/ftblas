#include "../include/ftblas.h"
#include "../include/params.h"

void ftblas_dtpsv_low(int uplo, int trans, int unit, int n, double *a, double *x, int incx)
{
    double *X, *A;
    X = x;
    A = a;

    if (incx != 1)
    {
        X = (double *)malloc(n * sizeof(double));
        ftblas_dcopy(n, x, incx, X, 1);
    }

    int i;
    for (i = 0; i < n; i++)
    {
        if (trans)
        {
            if (i > 0)
            {
#ifdef FT_ENABLED
                X[i] -= ftblas_ddot_ft(i, A, 1, X, 1);
#else
                X[i] -= ftblas_ddot_ori(i, A, 1, X, 1);
#endif
            }
        }

        if (!unit)
        {
            if (!trans)
            {
                X[i] /= A[0];
            }
            else
            {
                X[i] /= A[i];
            }
        }

        if (!trans)
        {
            if (i < n - 1)
            {
#ifdef FT_ENABLED
                ftblas_daxpy_ft(n - i - 1, -X[i], A + 1, 1, X + i + 1, 1);
#else
                ftblas_daxpy_ori(n - i - 1, -X[i], A + 1, 1, X + i + 1, 1);
#endif
            }
        }

        if (!trans)
        {
            A += (n - i);
        }
        else
        {
            A += (i + 1);
        }
    }

    if (incx != 1)
    {
        ftblas_dcopy(n, X, 1, x, incx);
        free(X);
    }
}

void ftblas_dtpsv_upp(int uplo, int trans, int unit, int n, double *a, double *x, int incx)
{
    double *X, *A;
    X = x;
    A = a;

    if (incx != 1)
    {
        X = (double *)malloc(n * sizeof(double));
        ftblas_dcopy(n, x, incx, X, 1);
    }

    int i;
    A += (n + 1) * n / 2 - 1;
    for (i = 0; i < n; i++)
    {
        if (trans)
        {
            if (i > 0)
            {
#ifdef FT_ENABLED
                X[n - i - 1] -= ftblas_ddot_ft(i, A + 1, 1, X + n - i, 1);
#else
                X[n - i - 1] -= ftblas_ddot_ori(i, A + 1, 1, X + n - i, 1);
#endif
            }
        }

        if (!unit)
        {

            X[n - i - 1] /= A[0];
        }

        if (!trans)
        {
            if (i < n - 1)
            {
#ifdef FT_ENABLED
                ftblas_daxpy_ft(n - i - 1, -X[n - i - 1], A - (n - i - 1), 1, X, 1);
#else
                ftblas_daxpy_ori(n - i - 1, -X[n - i - 1], A - (n - i - 1), 1, X, 1);
#endif
            }
        }

        if (!trans)
        {
            A -= (n - i);
        }
        else
        {
            A -= (i + 2);
        }
    }

    if (incx != 1)
    {
        ftblas_dcopy(n, X, 1, x, incx);
        free(X);
    }
}