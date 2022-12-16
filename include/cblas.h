#ifndef _CBLAS_H_
#define _CBLAS_H_

#ifndef FTBLAS_INT
  #define FTBLAS_INT int
#endif

enum CBLAS_ORDER     {CblasRowMajor=101, CblasColMajor=102};
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};
enum CBLAS_UPLO      {CblasUpper=121, CblasLower=122};
enum CBLAS_DIAG      {CblasNonUnit=131, CblasUnit=132};
enum CBLAS_SIDE      {CblasLeft=141, CblasRight=142};

double cblas_ddot(const FTBLAS_INT n, const double *x, const FTBLAS_INT inc_x, const double *y, const FTBLAS_INT inc_y);
void cblas_dgemm(const CBLAS_ORDER Layout, const CBLAS_TRANSPOSE transa, const CBLAS_TRANSPOSE transb, \
    const FTBLAS_INT m, const FTBLAS_INT n, const FTBLAS_INT k, const double alpha, \
    const double *a, const FTBLAS_INT lda, const double *b, const FTBLAS_INT ldb,\
    const double beta, double *c, const FTBLAS_INT ldc);

void cblas_dsymm(const CBLAS_ORDER Layout, const CBLAS_SIDE side, const CBLAS_UPLO uplo, \
    const FTBLAS_INT m, const FTBLAS_INT n, const FTBLAS_INT k, const double alpha, \
    const double *a, const FTBLAS_INT lda, const double *b, const FTBLAS_INT ldb,\
    const double beta, double *c, const FTBLAS_INT ldc);

#endif