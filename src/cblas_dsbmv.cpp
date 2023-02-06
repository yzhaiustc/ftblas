#include "../include/ftblas.h"
#include "../include/cblas.h"

void cblas_dsbmv_compute(const CBLAS_UPLO Uplo, const FTBLAS_INT n, const FTBLAS_INT k, const double alpha, const double *a, const FTBLAS_INT lda, const double *x,
                        const int incx, const double beta, const double *y, const int incy);

void cblas_dsbmv(const CBLAS_UPLO uplo, const FTBLAS_INT n, const FTBLAS_INT k, const double alpha, const double *a, const FTBLAS_INT lda, const double *x,
                const int incx, const double beta, const double *y, const int incy)
{

    cblas_dsbmv_compute(uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

void cblas_dsbmv_compute(const CBLAS_UPLO Uplo, const FTBLAS_INT n, const FTBLAS_INT k, const double alpha, const double *a, const FTBLAS_INT lda, const double *x,
                        const int incx, const double beta, const double *y, const int incy)
{
    int uplo;
    if (Uplo == CblasUpper)
        uplo = 0;
    if (Uplo == CblasLower)
        uplo = 1;

    if (uplo == 1)
    {
#ifdef FT_ENABLED
        ftblas_dsbmv_upp(n, k, alpha, (double *)a, lda, (double *)x, incx, beta, (double *)y, incy);
#else
        ftblas_dsbmv_upp(n, k, alpha, (double *)a, lda, (double *)x, incx, beta, (double *)y, incy);
#endif
    }
    else
    {
#ifdef FT_ENABLED
        ftblas_dsbmv_low(n, k, alpha, (double *)a, lda, (double *)x, incx, beta, (double *)y, incy);
#else
        ftblas_dsbmv_low(n, k, alpha, (double *)a, lda, (double *)x, incx, beta, (double *)y, incy);
#endif
    }
}