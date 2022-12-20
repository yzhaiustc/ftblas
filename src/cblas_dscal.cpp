#include "../include/ftblas.h"
#include "../include/cblas.h"

double cblas_dscal(const FTBLAS_INT n, const double alpha, const double *x, const FTBLAS_INT inc_x) {
#ifdef FT_ENABLED
    ftblas_dscal_ft(n, alpha, x, inc_x);
#else
    ftblas_dscal_ori(n, alpha, x, inc_x);
#endif
}