#include "../include/ftblas.h"
#include "../include/cblas.h"

void cblas_dtrsm(const CBLAS_ORDER Layout, const CBLAS_SIDE side, const CBLAS_UPLO uplo, \
    const CBLAS_TRANSPOSE transa, const CBLAS_DIAG diag, \
    const FTBLAS_INT m, const FTBLAS_INT n, const double alpha, \
    const double *a, const FTBLAS_INT lda, const double *b, const FTBLAS_INT ldb) {
        // currently we only support colmajor & non transposed.
        bool case1 = (Layout==CblasColMajor) && ((side==CblasLeft) && (uplo==CblasLower) && (transa == CblasNoTrans));

        if (!case1) {
            printf("This is not supported yet, return.\n");
            exit(-1);
        }
        
        if (lda < m || ldb < m) {
            printf("illegal input.\n");
            exit(-2);
        }

        if (alpha==0.) {
            return;
        }

#ifdef FT_ENABLED
        ftblas_dtrsm_ft(m,n,alpha,(double *)a,lda,(double *)b,ldb);
#else
        ftblas_dtrsm_ori(m,n,alpha,(double *)a,lda,(double *)b,ldb);
#endif
}