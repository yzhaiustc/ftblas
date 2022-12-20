#include "../include/ftblas.h"
#include "../include/cblas.h"

double cblas_daxpy(const FTBLAS_INT n, const double alpha, const double *x, const FTBLAS_INT inc_x, const double *y, const FTBLAS_INT inc_y) {
#ifdef FT_ENABLED
    ftblas_daxpy_ft(n, alpha, x, inc_x, y, inc_y);
#else
    ftblas_daxpy_ori(n, x, alpha, inc_x, y, inc_y);
#endif
}