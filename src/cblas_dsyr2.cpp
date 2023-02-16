#include "../include/ftblas.h"
#include "../include/cblas.h"

void dsyr2_compute(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, FTBLAS_INT n, double alpha,
                   double *x, int incx, double *y, int incy, double *a, FTBLAS_INT lda);

void cblas_dsyr2(const CBLAS_ORDER order, const CBLAS_UPLO uplo, const FTBLAS_INT n, const double alpha,
                 const double *x, const int incx, const double *y, const int incy, const double *a, const FTBLAS_INT lda)
{
    dsyr2_compute(order, uplo, n, alpha, (double *)x, incx, (double *)y, incy, (double *)a, lda);
}

void dsyr2_compute(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, FTBLAS_INT n, double alpha,
                   double *x, int incx, double *y, int incy, double *a, FTBLAS_INT lda)
{

    int uplo;
    if (order == CblasColMajor)
    {
        if (Uplo == CblasUpper)
            uplo = 0;
        if (Uplo == CblasLower)
            uplo = 1;
    }

    if (order == CblasRowMajor)
    {
        if (Uplo == CblasUpper)
            uplo = 1;
        if (Uplo == CblasLower)
            uplo = 0;
    }

    if (uplo == 1)
    {
        ftblas_dsyr2_upp(n, alpha, x, incx, y, incy, a, lda);
    }
    else
    {
        ftblas_dsyr2_low(n, alpha, x, incx, y, incy, a, lda);
    }
}
