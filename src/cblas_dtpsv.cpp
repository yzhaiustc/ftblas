#include "../include/ftblas.h"
#include "../include/cblas.h"

void cblas_dtpsv_compute(const CBLAS_ORDER layout, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans, const CBLAS_DIAG Diag,
                         FTBLAS_INT n, double *ap, double *x, int incx);

void cblas_dtpsv(const CBLAS_ORDER layout, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans, const CBLAS_DIAG Diag,
                 const FTBLAS_INT n, const double *ap, const double *x, const int incx)
{
    cblas_dtpsv_compute(layout, Uplo, Trans, Diag, n, (double *)ap, (double *)x, incx);
}

void cblas_dtpsv_compute(const CBLAS_ORDER layout, const CBLAS_UPLO Uplo, const CBLAS_TRANSPOSE Trans, const CBLAS_DIAG Diag,
                         FTBLAS_INT n, double *ap, double *x, int incx)
{
    int trans, uplo, unit, info;
    if (layout == CblasColMajor)
    {
        if (Uplo == CblasUpper)
            uplo = 0;
        if (Uplo == CblasLower)
            uplo = 1;

        if (Trans == CblasNoTrans)
            trans = 0;
        if (Trans == CblasTrans)
            trans = 1;

        if (Diag == CblasUnit)
            unit = 0;
        if (Diag == CblasNonUnit)
            unit = 1;
    }

    if (layout == CblasRowMajor)
    {
        if (Uplo == CblasUpper)
            uplo = 1;
        if (Uplo == CblasLower)
            uplo = 0;

        if (Trans == CblasNoTrans)
            trans = 1;
        if (Trans == CblasTrans)
            trans = 0;

        if (Diag == CblasUnit)
            unit = 0;
        if (Diag == CblasNonUnit)
            unit = 1;
    }

    if (uplo == trans)
    {
        ftblas_dtpsv_upp(uplo, trans, unit, n, ap, x, incx);
    }
    else
    {
        ftblas_dtpsv_low(uplo, trans, unit, n, ap, x, incx);
    }
}