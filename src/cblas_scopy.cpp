#include "../include/ftblas.h"
#include "../include/cblas.h"

void cblas_scopy(const FTBLAS_INT n, const float *x, const FTBLAS_INT inc_x, const float *y, const FTBLAS_INT inc_y) {
    ftblas_scopy(n, x, inc_x, y, inc_y);
}