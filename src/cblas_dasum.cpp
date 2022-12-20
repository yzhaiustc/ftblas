#include "../include/ftblas.h"
#include "../include/cblas.h"

double cblas_dasum(const FTBLAS_INT n, const double *x, const FTBLAS_INT inc_x) {
#ifdef FT_ENABLED
    ftblas_dasum_ft(n, x, inc_x);
#else
    ftblas_dasum_ori(n, x, inc_x);
#endif
}