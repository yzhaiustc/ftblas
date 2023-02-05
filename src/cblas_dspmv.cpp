#include "../include/ftblas.h"
#include "../include/cblas.h"

void cblas_dspmv_compute(const CBLAS_UPLO Uplo, const FTBLAS_INT n, const double alpha, const double *ap, const double *x,
                        const int incx, const double beta, const double *y, const int incy);

void cblas_dspmv(const CBLAS_UPLO uplo, const FTBLAS_INT n, const double alpha, const double *ap, const double *x,
                const int incx, const double beta, const double *y, const int incy)
{

    cblas_dspmv_compute(uplo, n, alpha, ap, x, incx, beta, y, incy);
}

void cblas_dspmv_compute(const CBLAS_UPLO Uplo, const FTBLAS_INT n, const double alpha, const double *ap, const double *x,
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
        ftblas_dspmv_upp(n, alpha, (double *)ap, (double *)x, incx, beta, (double *)y, incy);
#else
        ftblas_dspmv_upp(n, alpha, (double *)ap, (double *)x, incx, beta, (double *)y, incy);
#endif
    }
    else
    {
#ifdef FT_ENABLED
        ftblas_dspmv_low(n, alpha, (double *)ap, (double *)x, incx, beta, (double *)y, incy);
#else
        ftblas_dspmv_low(n, alpha, (double *)ap, (double *)x, incx, beta, (double *)y, incy);
#endif
    }
}