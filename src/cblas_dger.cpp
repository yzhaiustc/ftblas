#include "../include/ftblas.h"
#include "../include/cblas.h"
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

void dger_compute(const CBLAS_ORDER order, \
       int m, int n, \
       double alpha, \
       double *x, int incx, \
       double *y, int incy, \
       double  *a, int lda);

void cblas_dger(const CBLAS_ORDER order, \
       const int m, const int n, \
       const double alpha, \
       const double *x, const int incx, \
       const double *y, const int incy, \
       double  *a, const int lda)
{
       dger_compute(order, m, n, alpha, (double *)x, incx, (double *)y, incy, a, lda);

}

void dger_compute(const CBLAS_ORDER order, \
       int m, int n, \
       double alpha, \
       double *x, int incx, \
       double *y, int incy, \
       double  *a, int lda)

{
    double *buffer;
    int info, t, lenx=m, leny=n;
    if (order == CblasColMajor)
    {
        info = -1;
        t = n;
        n = m;
        m = t;

        t = lenx;
        lenx = leny;
        leny = t;

        t = incx;
        incx = incy;
        incy = t;

        buffer = x;
        x = y;
        y = buffer;
        if (lda < MAX(1, n))
            info = 9;
        if (incy == 0)
            info = 7;
        if (incx == 0)
            info = 5;
        if (n < 0)
            info = 2;
        if (m < 0)
            info = 1;
    }

    if (order == CblasRowMajor)
    {
        info = -1;

        if (lda < MAX(1, n))
            info = 9;
        if (incy == 0)
            info = 7;
        if (incx == 0)
            info = 5;
        if (n < 0)
            info = 2;
        if (m < 0)
            info = 1;
    }
    if (info >= 0)
    {
        printf("illegal input, error code %d.\n", info);
        return;
    }
    if (m == 0 || n == 0) return;
    if (alpha == 0.) return;
    if (incx==1&&incy==1) {
        if (alpha==1.0) {
#ifdef FT_ENABLED
            ftblas_dger_ft(lda, m, n, alpha, a, x, y);
#else
            ftblas_dger_ori(lda, m, n, alpha, a, x, y);
#endif
        }
        else{
            double *buffer_x;
            buffer_x=(double*)malloc(lenx*sizeof(double));
            int i;
            for (i=0;i<lenx;i++) {
                buffer_x[i]=alpha*x[i];
            }
#ifdef FT_ENABLED
            ftblas_dger_ft(lda, m, n, alpha, a, buffer_x, y);
#else
            ftblas_dger_ori(lda, m, n, alpha, a, buffer_x, y);
#endif
            free(buffer_x);
        }
    }
    else{
        double *buffer_x, *buffer_y;
        buffer_x=(double*)malloc(lenx*sizeof(double));
        buffer_y=(double*)malloc(leny*sizeof(double));
        int i,j;
        for (i=0,j=0;i<lenx;i++) {
            buffer_x[i]=x[j];
            j+=incx;
        }
        for (i=0,j=0;i<leny;i++) {
            buffer_y[i]=y[j];
            j+=incy;
        }
#ifdef FT_ENABLED
        ftblas_dger_ft(lda, m, n, alpha, a, buffer_x, buffer_y);
#else
        ftblas_dger_ori(lda, m, n, alpha, a, buffer_x, buffer_y);
#endif
        free(buffer_x);free(buffer_y);
    }
}