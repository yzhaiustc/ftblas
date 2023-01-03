#include "../include/ftblas.h"
#include "../include/cblas.h"

void cblas_drot(const FTBLAS_INT n, const double *x, const FTBLAS_INT inc_x, const double *y, const FTBLAS_INT inc_y, const double c, const double s) {
#ifdef FT_ENABLED
    ftblas_drot_ft(n, x, inc_x, y, inc_y, c, s);
#else
    ftblas_drot_ori(n, x, inc_x, y, inc_y, c, s);
#endif
}