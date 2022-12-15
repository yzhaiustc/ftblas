#include "../include/ftblas.h"
#include "../include/cblas.h"

void cblas_dgemm(const CBLAS_ORDER Layout, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb, \
    const FTBLAS_INT m, const FTBLAS_INT n, const FTBLAS_INT k, const double alpha, \
    const double *a, const FTBLAS_INT lda, const double *b, const FTBLAS_INT ldb,\
    const double beta, double *c, const FTBLAS_INT ldc) {
        // currently we only support colmajor & non transposed.
        bool case1 = (Layout==CblasColMajor) && ((transa==CblasNoTrans) && (transb==CblasNoTrans));
        
        if (!case1) {
            printf("This is not supported yet, return.\n");
            exit(-1);
        }
        
        if (lda < m || ldb < k || ldc < m) {
            printf("illegal input.\n");
            exit(-2);
        }

        if (alpha==0.) {
            return;
        }

#ifdef FT_ENABLED
        ftblas_dgemm_ft(m,n,k,alpha,(double *)a,lda,(double *)b,ldb,beta,(double *)c,ldc);
#else
        ftblas_dgemm_ori(m,n,k,alpha,(double *)a,lda,(double *)b,ldb,beta,(double *)c,ldc);
#endif
}