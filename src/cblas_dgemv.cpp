#include "../include/ftblas.h"
#include "../include/cblas.h"

#define MAX(a, b) (((a) > (b)) ? (a) : (b))

void cblas_dgemv(const CBLAS_ORDER order, const CBLAS_TRANSPOSE TransA,\
                 const int m, const int n, \
                 const double alpha, const double *a, const int lda, \
                 const double *x, const int incx, \
                 const double beta, double *y, const int incy)
{
    int trans, info;
    if (order == CblasColMajor)
    {
        if (TransA == CblasNoTrans)
            trans = 0;
        if (TransA == CblasTrans)
            trans = 1;

        info = -1;

        if (incy == 0)
            info = 11;
        if (incx == 0)
            info = 8;
        if (lda < MAX(1, m))
            info = 6;
        if (n < 0)
            info = 3;
        if (m < 0)
            info = 2;
        if (trans < 0)
            info = 1;
    }

    if (order == CblasRowMajor)
    {
        if (TransA == CblasNoTrans)
            trans = 1;
        if (TransA == CblasTrans)
            trans = 0;

        info = -1;

        if (incy == 0)
            info = 11;
        if (incx == 0)
            info = 8;
        if (lda < MAX(1, n))
            info = 6;
        if (n < 0)
            info = 3;
        if (m < 0)
            info = 2;
        if (trans < 0)
            info = 1;
    }

    if (info >= 0)
    {
        printf("illegal input, error code %d.\n", info);
        return;
    }

    if ((m == 0) || (n == 0))
        return;

    if (alpha == 0.0)
        return;
    if (incx == 1 && incy == 1)
    {
        if (alpha == 1. && beta == 1.)
        {
            if (trans == 1)
            {
#ifdef FT_ENABLED
                ftblas_dgemv_t_ft((double *)a, (double *)x, y, m, n, lda);
#else
                ftblas_dgemv_t_ori((double *)a, (double *)x, y, m, n, lda);
#endif
            }
            else
            {
#ifdef FT_ENABLED
                ftblas_dgemv_n_ft((double *)a, (double *)x, y, m, n, lda);
#else
                ftblas_dgemv_n_ori((double *)a, (double *)x, y, m, n, lda);
#endif
            }
        }
        else
        {
            int len_x, len_y;
            if (trans)
            {
                len_x = n;
                len_y = m;
            }
            else
            {
                len_x = m;
                len_y = n;
            }
            if (alpha != 1. && beta == 1.)
            {
                double *buffer_x = (double *)malloc(len_x * sizeof(double));
                int i;
                for (i = 0; i < len_x; i++)
                    buffer_x[i] = alpha * x[i];
                if (trans)
                {
#ifdef FT_ENABLED
                    ftblas_dgemv_t_ft((double *)a, buffer_x, y, m, n, lda);
#else
                    ftblas_dgemv_t_ori((double *)a, buffer_x, y, m, n, lda);
#endif
                }
                else
                {
#ifdef FT_ENABLED
                    ftblas_dgemv_n_ft((double *)a, buffer_x, y, m, n, lda);
#else
                    ftblas_dgemv_n_ori((double *)a, buffer_x, y, m, n, lda);
#endif
                }
                free(buffer_x);
            }
            else
            {
                if (beta != 1. && alpha == 1.)
                {
                    double *buffer_y = (double *)malloc(len_y * sizeof(double));
                    int i;
                    for (i = 0; i < len_y; i++)
                        buffer_y[i] = 0.;
                    if (trans)
                    {
#ifdef FT_ENABLED
                        ftblas_dgemv_t_ft((double *)a, (double *)x, buffer_y, m, n, lda);
#else
                        ftblas_dgemv_t_ori((double *)a, (double *)x, buffer_y, m, n, lda);
#endif
                    }
                    else
                    {
#ifdef FT_ENABLED
                        ftblas_dgemv_n_ft((double *)a, (double *)x, buffer_y, m, n, lda);
#else
                        ftblas_dgemv_n_ori((double *)a, (double *)x, buffer_y, m, n, lda);
#endif
                    }
                    for (i = 0; i < len_y; i++)
                        y[i] = beta * y[i] + buffer_y[i];
                    free(buffer_y);
                }
                else
                {
                    if (beta==0.){
                        double *buffer_x = (double *)malloc(len_x * sizeof(double));
                        double *buffer_y = (double *)malloc(len_y * sizeof(double));
                        int i;
                        for (i = 0; i < len_x; i++)
                            buffer_x[i] = alpha * x[i];
                        for (i = 0; i < len_y; i++)
                            buffer_y[i] = 0.;
                        if (trans)
                        {
#ifdef FT_ENABLED
                            ftblas_dgemv_t_ft((double *)a, buffer_x, buffer_y, m, n, lda);
#else
                            ftblas_dgemv_t_ori((double *)a, buffer_x, buffer_y, m, n, lda);
#endif
                        }
                        else
                        {
#ifdef FT_ENABLED
                            ftblas_dgemv_n_ft((double *)a, buffer_x, buffer_y, m, n, lda);
#else
                            ftblas_dgemv_n_ori((double *)a, buffer_x, buffer_y, m, n, lda);
#endif
                        }
                        for (i = 0; i < len_y; i++)
                            y[i] = beta * y[i] + buffer_y[i];
                        free(buffer_x);
                        free(buffer_y);
                    }else{
                        double *buffer_x = (double *)malloc(len_x * sizeof(double));
                        // alpha/=beta;
                        int i;
                        for (i = 0; i < len_x; i++)
                            buffer_x[i] = (alpha / beta) * x[i];
                        if (trans)
                        {
#ifdef FT_ENABLED
                            ftblas_dgemv_t_ft((double *)a, buffer_x, y, m, n, lda);
#else
                            ftblas_dgemv_t_ori((double *)a, buffer_x, y, m, n, lda);
#endif
                        }
                        else
                        {
#ifdef FT_ENABLED
                            ftblas_dgemv_n_ft((double *)a, buffer_x, y, m, n, lda);
#else
                            ftblas_dgemv_n_ori((double *)a, buffer_x, y, m, n, lda);
#endif
                        }
                        for (i = 0; i < len_y; i++)
                            y[i] = beta * y[i];
                        free(buffer_x);
                    }

                }
            }
        }
    }
    else
    {
        int len_x, len_y;
        if (trans)
        {
            len_x = n;
            len_y = m;
        }
        else
        {
            len_x = m;
            len_y = n;
        }
        double *buffer_x = (double *)malloc(len_x * sizeof(double));
        double *buffer_y = (double *)malloc(len_y * sizeof(double));
        int i, j;
        for (i = 0, j=0; i < len_x; i++)
        {
            buffer_x[i] = alpha * x[j];
            j+=incx;
        }
        for (i = 0; i < len_y; i++)
            buffer_y[i] = 0.;
        if (trans)
        {
#ifdef FT_ENABLED
            ftblas_dgemv_t_ft((double *)a, buffer_x, buffer_y, m, n, lda);
#else
            ftblas_dgemv_t_ori((double *)a, buffer_x, buffer_y, m, n, lda);
#endif
        }
        else
        {
#ifdef FT_ENABLED
            ftblas_dgemv_n_ft((double *)a, buffer_x, buffer_y, m, n, lda);
#else
            ftblas_dgemv_n_ori((double *)a, buffer_x, buffer_y, m, n, lda);
#endif
        }
        for (i = 0, j=0; i < len_y; i++){
            y[j] = beta * y[j] + buffer_y[i];
            j+=incy;
        }
            
        free(buffer_x);
        free(buffer_y);
    }

    return;
}