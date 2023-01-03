#include "../include/ftblas.h"
#include "../include/cblas.h"

double cblas_dnrm2(const FTBLAS_INT n, const double *x, const FTBLAS_INT inc_x) {
#ifdef FT_ENABLED
    ftblas_dnrm2_ft(n, x, inc_x);
#else
    ftblas_dnrm2_ori(n, x, inc_x);
#endif
}