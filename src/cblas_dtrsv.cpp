#include "../include/ftblas.h"
#include "../include/cblas.h"
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

void dtrsv_compute(const CBLAS_ORDER order, const CBLAS_UPLO Uplo,
       const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
       int n, double  *a, int lda, double  *x, int incx);

void cblas_dtrsv(const CBLAS_ORDER order, const CBLAS_UPLO Uplo,
       const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
       const int n, const double  *a, const int lda, double  *x, const int incx)
{
    dtrsv_compute(order, Uplo, TransA, Diag, n, (double *)a, lda, x, incx);
}

void dtrsv_compute(const CBLAS_ORDER order, const CBLAS_UPLO Uplo,
       const CBLAS_TRANSPOSE TransA, const CBLAS_DIAG Diag,
       int n, double  *a, int lda, double  *x, int incx)
{
    int trans, uplo, unit, info;
    info  =  0;
  if (order == CblasColMajor) {
    if (Uplo == CblasUpper)         uplo  = 0;
    if (Uplo == CblasLower)         uplo  = 1;

    if (TransA == CblasNoTrans)     trans = 0;
    if (TransA == CblasTrans)       trans = 1;

    if (Diag == CblasUnit)          unit  = 0;
    if (Diag == CblasNonUnit)       unit  = 1;

    info = -1;

    if (incx == 0)          info =  8;
    if (lda  < MAX(1, n))   info =  6;
    if (n < 0)              info =  4;
    if (unit  < 0)          info =  3;
    if (trans < 0)          info =  2;
    if (uplo  < 0)          info =  1;
  }

  if (order == CblasRowMajor) {
    if (Uplo == CblasUpper)         uplo  = 1;
    if (Uplo == CblasLower)         uplo  = 0;

    if (TransA == CblasNoTrans)     trans = 1;
    if (TransA == CblasTrans)       trans = 0;


    if (Diag == CblasUnit)          unit  = 0;
    if (Diag == CblasNonUnit)       unit  = 1;

    info = -1;

    if (incx == 0)          info =  8;
    if (lda  < MAX(1, n))   info =  6;
    if (n < 0)              info =  4;
    if (unit  < 0)          info =  3;
    if (trans < 0)          info =  2;
    if (uplo  < 0)          info =  1;
  }

    if (info >= 0)
    {
        printf("illegal input, error code %d.\n", info);
        return;
    }

    if (n == 0) return;
    if (Diag!=CblasNonUnit||Uplo!=CblasLower||TransA!=CblasNoTrans) {
        printf("To be implemented.\n");
        return;
    }
    if (order==CblasColMajor){
        if (incx==1) {
#ifdef FT_ENABLED
            ftblas_dtrsv_low_col_ft(a, lda, x, n);
#else
            ftblas_dtrsv_low_col_ori(a, lda, x, n);
#endif
        }
        else{
            double *buffer_x=(double*)malloc(n*sizeof(double));
            int i,j;
            for (i=0,j=0;i<n;i++) {
                buffer_x[i]=x[j];
                j+=incx;
            }
#ifdef FT_ENABLED
            ftblas_dtrsv_low_col_ft(a, lda, buffer_x, n);
#else
            ftblas_dtrsv_low_col_ori(a, lda, buffer_x, n);
#endif
            for (i=0,j=0;i<n;i++) {
                x[j]=buffer_x[i];
                j+=incx;
            }
            free(buffer_x);
        }
    }else{
        if (incx==1) {
#ifdef FT_ENABLED
            ftblas_dtrsv_low_row_ft(a, lda, x, n);
#else
            ftblas_dtrsv_low_row_ori(a, lda, x, n);
#endif
        }
        else{
            double *buffer_x=(double*)malloc(n*sizeof(double));
            int i,j;
            for (i=0,j=0;i<n;i++) {
                buffer_x[i]=x[j];
                j+=incx;
            }
#ifdef FT_ENABLED
            ftblas_dtrsv_low_row_ft(a, lda, buffer_x, n);
#else
            ftblas_dtrsv_low_row_ori(a, lda, buffer_x, n);
#endif
            for (i=0,j=0;i<n;i++) {
                x[j]=buffer_x[i];
                j+=incx;
            }
            free(buffer_x);
        }
    }
    return;
}