#include "../include/ftblas.h"
#include "../include/cblas.h"

void cblas_dspr2_compute(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const FTBLAS_INT n, const double alpha,
                        const double *x, const int incx, const double *y, const int incy, const double *a);

void cblas_dspr2(const CBLAS_ORDER order, const CBLAS_UPLO uplo, const FTBLAS_INT n, const double alpha,
                const double *x, const int incx, const double *y, const int incy, const double *ap)
{

    cblas_dspr2_compute(order, uplo, n, alpha, x, incx, y, incy, ap);
}

void cblas_dspr2_compute(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, const FTBLAS_INT n, const double alpha,
                        const double *x, const int incx, const double *y, const int incy, const double *ap)
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
        ftblas_dspr2_upp(n, alpha, (double *)x, incx, (double *)y, incy, (double *)ap);
    }
    else
    {
        ftblas_dspr2_low(n, alpha, (double *)x, incx, (double *)y, incy, (double *)ap);
    }
}