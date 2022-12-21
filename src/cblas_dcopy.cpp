#include "../include/ftblas.h"
#include "../include/cblas.h"

double cblas_dcopy(const FTBLAS_INT n, const double *x, const FTBLAS_INT inc_x, const double *y, const FTBLAS_INT inc_y) {
    ftblas_dcopy(n, x, inc_x, y, inc_y);
}