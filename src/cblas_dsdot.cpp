#include "../include/ftblas.h"
#include "../include/cblas.h"

double cblas_dsdot(const FTBLAS_INT n, const double *x, const FTBLAS_INT inc_x, const double *y, const FTBLAS_INT inc_y) {
#ifdef FT_ENABLED
    ftblas_dsdot_ft(n, x, inc_x, y, inc_y);
#else
    ftblas_dsdot_ori(n, x, inc_x, y, inc_y);
#endif
}