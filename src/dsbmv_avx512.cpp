#include "../include/ftblas.h"

void ftblas_dbmv_low(int n, int k, double alpha, double *a, int lda, double *x,
                     int incx, double beta, double *y, int incy)
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
        ftblas_dcopy(n, y, incx, Y, 1);
    }

    int i, length;
    for (i = 0; i < n; i++)
    {
        length = i;
        if (length > k)
            length = k;

#ifdef FT_ENABLED
        ftblas_daxpy_ft(length + 1, alpha * X[i], (a + i * lda) + k - i, 1, Y + i - length, 1);
#else
        ftblas_daxpy_ori(length + 1, alpha * X[i], (a + i * lda) + k - i, 1, Y + i - length, 1);
#endif

#ifdef FT_ENABLED
        Y[i] += alpha * ftblas_ddot_ft(length, (a + i * lda) + k - length, 1, X + i - length, 1);
#else
        Y[i] += alpha * ftblas_ddot_ori(length, (a + i * lda) + k - length, 1, X + i - length, 1);
#endif
    }

    if (incx != 1)
        free(X);

    if (incy != 1)
        free(Y);
}

void ftblas_dbmv_upp(int n, int k, double alpha, double *a, int lda, double *x,
                     int incx, double beta, double *y, int incy)
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
        ftblas_dcopy(n, y, incx, Y, 1);
    }

    int i, length;
    for (i = 0; i < n; i++)
    {
        length = k;
        if (n - 1 - i < k)
            length = n - i - 1;

#ifdef FT_ENABLED
        ftblas_daxpy_ft(length + 1, alpha * X[i], (a + i * lda), 1, Y + i, 1);
#else
        ftblas_daxpy_ori(length + 1, alpha * X[i], (a + i * lda), 1, Y + i, 1);
#endif

#ifdef FT_ENABLED
        Y[i] += alpha * ftblas_ddot_ft(length, (a + i * lda) + 1, 1, X + i + 1, 1);
#else
        Y[i] += alpha * ftblas_ddot_ori(length, (a + i * lda) + 1, 1, X + i + 1, 1);
#endif
    }

    if (incx != 1)
        free(X);

    if (incy != 1)
        free(Y);
}
