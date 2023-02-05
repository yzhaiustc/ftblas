#include "../include/ftblas.h"
#include "../include/cblas.h"

void cblas_dspr_compute(const CBLAS_UPLO Uplo, const FTBLAS_INT n, const double alpha, const double *x, const int incx, const double *a);

void cblas_dspr(const CBLAS_UPLO uplo, const FTBLAS_INT n, const double alpha, const double *x, const int incx, const double *ap)
{

    cblas_dspr_compute(uplo, n, alpha, x, incx, ap);
}

void cblas_dspr_compute(const CBLAS_UPLO Uplo, const FTBLAS_INT n, const double alpha, const double *x, const int incx, const double *ap)
{
    int uplo;
    if (Uplo == CblasUpper)
        uplo = 0;
    if (Uplo == CblasLower)
        uplo = 1;

    if (uplo == 1)
    {
#ifdef FT_ENABLED
        ftblas_dspr_upp(n, alpha, (double *)x, incx, (double *)ap);
#else
        ftblas_dspr_upp(n, alpha, (double *)x, incx, (double *)ap);
#endif
    }
    else
    {
#ifdef FT_ENABLED
        ftblas_dspr_low(n, alpha, (double *)x, incx, (double *)ap);
#else
        ftblas_dspr_low(n, alpha, (double *)x, incx, (double *)ap);
#endif
    }
}