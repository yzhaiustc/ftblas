#include "../include/ftblas.h"
#include "../include/cblas.h"

void cblas_saxpy(const FTBLAS_INT n, const float alpha, const float *x, const FTBLAS_INT inc_x, const float *y, const FTBLAS_INT inc_y) {
#ifdef FT_ENABLED
    ftblas_saxpy_ft(n, alpha, x, inc_x, y, inc_y);
#else
    ftblas_saxpy_ori(n, alpha, x, inc_x, y, inc_y);
#endif
}