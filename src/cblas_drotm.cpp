#include "../include/ftblas.h"
#include "../include/cblas.h"

void cblas_drotm(const FTBLAS_INT n, const double *x, const FTBLAS_INT inc_x, const double *y, const FTBLAS_INT inc_y, const double *param) {
#ifdef FT_ENABLED
    ftblas_drotm_ft(n, x, inc_x, y, inc_y, param);
#else
    ftblas_drotm_ori(n, x, inc_x, y, inc_y, param);
#endif
}