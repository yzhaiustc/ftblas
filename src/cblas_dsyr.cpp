#include "../include/ftblas.h"
#include "../include/cblas.h"

void dsyr_compute(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, FTBLAS_INT n, double alpha, double *x,
                  int incx, double *a, FTBLAS_INT lda);

void cblas_dsyr(const CBLAS_ORDER order, const CBLAS_UPLO uplo, const FTBLAS_INT n, const double alpha, const double *x,
                const int incx, const double *a, const FTBLAS_INT lda)
{
    dsyr_compute(order, uplo, n, alpha, (double *)x, incx, (double *)a, lda);
}

void dsyr_compute(const CBLAS_ORDER order, const CBLAS_UPLO Uplo, FTBLAS_INT n, double alpha, double *x,
                  int incx, double *a, FTBLAS_INT lda)
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

// TODO: Check for "if (incx==1) else buffer_x" required?
    if (uplo == 1) 
    {
        ftblas_dsyr_upp_row(n, alpha, x, incx, a, lda);
    } else
    {
        ftblas_dsyr_low_row(n, alpha, x, incx, a, lda);
    }
}
