#include "../include/ftblas.h"
#include "../include/params.h"

void ftblas_dtrmv_low(int uplo, int trans, int unit, int n, double *a, int lda, double *x, int incx)
{
    double *X, *A1, *X1;
    X = x;

    if (incx != 1)
    {
        X = (double *)malloc(n * sizeof(double));
        ftblas_dcopy(n, x, incx, X, 1);
    }

    int i, is, min_i;
    for (is = n; is > 0; is -= DTB_ENTRIES)
    {
        min_i = is < DTB_ENTRIES ? is : DTB_ENTRIES;

        if (!trans)
        {
            if (n - is > 0)
            {
#ifdef FT_ENABLED
                ftblas_dgemv_n_ft(a + is + (is - min_i) * lda, X + is - min_i, X + is, n - is, min_i, lda);
#else
                ftblas_dgemv_n_ori(a + is + (is - min_i) * lda, X + is - min_i, X + is, n - is, min_i, lda);
#endif
            }
        }

        for (i = 0; i < min_i; i++)
        {
            A1 = a + (is - i - 1) * lda + (is - i - 1);
            X1 = X + (is - i - 1);
            if (!trans)
            {
                if (i > 0)
                {
#ifdef FT_ENABLED
                    ftblas_daxpy_ft(i, X1[0], A1 + 1, 1, X1 + 1, 1);
#else
                    ftblas_daxpy_ori(i, X1[0], A1 + 1, 1, X1 + 1, 1);
#endif
                }
            }
            if (!unit)
            {
                X1[0] *= A1[0];
            }
            if (trans)
            {
                if (i < min_i - 1)
                {
#ifdef FT_ENABLED
                    X1[0] += ftblas_ddot_ft(min_i - i - 1, A1 - (min_i - i - 1), 1, X1 - (min_i - i - 1), 1);
#else
                    X1[0] += ftblas_ddot_ori(min_i - i - 1, A1 - (min_i - i - 1), 1, X1 - (min_i - i - 1), 1);
#endif
                }
            }
        }

        if (trans)
        {
            if (is - min_i > 0)
            {
#ifdef FT_ENABLED
                ftblas_dgemv_t_ft(a + (is - min_i) * lda, X, X + (is - min_i), is - min_i, min_i, lda);
#else
                ftblas_dgemv_t_ori(a + (is - min_i) * lda, X, X + (is - min_i), is - min_i, min_i, lda);
#endif
            }
        }
    }

    if (incx != 1)
    {
        ftblas_dcopy(n, X, 1, x, incx);
        free(X);
    }
}

void ftblas_dtrmv_upp(int uplo, int trans, int unit, int n, double *a, int lda, double *x, int incx)
{
    double *X, *A1, *X1;
    X = x;

    if (incx != 1)
    {
        X = (double *)malloc(n * sizeof(double));
        ftblas_dcopy(n, x, incx, X, 1);
    }

    int i, is, min_i;
    for (is = 0; is < n; is += DTB_ENTRIES)
    {
        min_i = n - is < DTB_ENTRIES ? n - is : DTB_ENTRIES;

        if (!trans)
        {
            if (is > 0)
            {
#ifdef FT_ENABLED
                ftblas_dgemv_n_ft(a + is * lda, X + is, X, is, min_i, lda);
#else
                ftblas_dgemv_n_ori(a + is * lda, X + is, X, is, min_i, lda);
#endif
            }
        }

        for (i = 0; i < min_i; i++)
        {
            A1 = a + (is + i) * lda + is;
            X1 = X + is;
            if (!trans)
            {
                if (i > 0)
                {
#ifdef FT_ENABLED
                    ftblas_daxpy_ft(i, X1[i], A1, 1, X1, 1);
#else
                    ftblas_daxpy_ori(i, X1[i], A1, 1, X1, 1);
#endif
                }
            }
            if (!unit)
            {
                X1[i] *= A1[i];
            }
            if (trans)
            {
                if (i < min_i - 1)
                {
#ifdef FT_ENABLED
                    X1[i] += ftblas_ddot_ft(min_i - i - 1, A1 + i + 1, 1, X1 + i + 1, 1);
#else
                    X1[i] += ftblas_ddot_ori(min_i - i - 1, A1 + i + 1, 1, X1 + i + 1, 1);
#endif
                }
            }
        }

        if (trans)
        {
            if (n - is > min_i)
            {
#ifdef FT_ENABLED
                ftblas_dgemv_t_ft(a + is * lda + is + min_i, X + is + min_i, X + is, n - is - min_i, min_i, lda);
#else
                ftblas_dgemv_t_ori(a + is * lda + is + min_i, X + is + min_i, X + is, n - is - min_i, min_i, lda);
#endif
            }
        }
    }

    if (incx != 1)
    {
        ftblas_dcopy(n, X, 1, x, incx);
        free(X);
    }
}