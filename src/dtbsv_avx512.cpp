#include "../include/ftblas.h"

void ftblas_dtbsv_low(int uplo, int trans, int unit, int n, int k, double *a, int lda, double *x, int incx)
{
    double *X, *A;
    X = x;
    A = a;

    if (incx != 1)
    {
        X = (double *)malloc(n * sizeof(double));
        ftblas_dcopy(n, x, incx, X, 1);
    }

    int i, length;
    for (i = 0; i < n; i++)
    {
        if (trans)
        {
            length = i;
            if (length > k)
                length = k;
            if (length > 0)
            {
#ifdef FT_ENABLED
                X[i] -= ftblas_ddot_ft(length, A + k - length, 1, X + i - length, 1);
#else
                X[i] -= ftblas_ddot_ori(length, A + k - length, 1, X + i - length, 1);
#endif
            }
        }

        if (!unit)
        {
            if (trans)
            {
                X[i] /= A[k];
            }
            else
            {
                X[i] /= A[0];
            }
        }

        if (!trans)
        {
            length = n - i - 1;
            if (length > k)
                length = k;

            if (length > 0)
            {
#ifdef FT_ENABLED
                ftblas_daxpy_ft(length, -X[i], A + 1, 1, X + i + 1, 1);
#else
                ftblas_daxpy_ori(length, -X[i], A + 1, 1, X + i + 1, 1);
#endif
            }
        }

        A += lda;
    }

    if (incx != 1)
    {
        ftblas_dcopy(n, X, 1, x, incx);
        free(X);
    }
}

void ftblas_dtbsv_upp(int uplo, int trans, int unit, int n, int k, double *a, int lda, double *x, int incx)
{
    double *X, *A;
    X = x;
    A = a;

    if (incx != 1)
    {
        X = (double *)malloc(n * sizeof(double));
        ftblas_dcopy(n, x, incx, X, 1);
    }

    int i, length;
    A += (n - 1) * lda;

    for (i = n - 1; i >= 0; i--)
    {
        if (trans)
        {
            length = n - i - 1;
            if (length > k)
                length = k;

            if (length > 0)
            {
#ifdef FT_ENABLED
                X[i] -= ftblas_ddot_ft(length, A + 1, 1, X + i + 1, 1);
#else
                X[i] -= ftblas_ddot_ori(length, A + 1, 1, X + i + 1, 1);
#endif
            }
        }

        if (!unit)
        {
            if (trans)
            {
                X[i] /= A[0];
            }
            else
            {
                X[i] /= A[k];
            }
        }

        if (!trans)
        {
            length = i;
            if (length > k)
                length = k;

            if (length > 0)

            {
#ifdef FT_ENABLED
                ftblas_daxpy_ori(length, -X[i], A + k - length, 1, X + i - length, 1);
#else
                ftblas_daxpy_ori(length, -X[i], A + k - length, 1, X + i - length, 1);
#endif
            }
        }
        A -= lda;
    }

    if (incx != 1)
    {
        ftblas_dcopy(n, X, 1, x, incx);
        free(X);
    }
}