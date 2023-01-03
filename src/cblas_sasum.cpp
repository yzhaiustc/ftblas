#include "../include/ftblas.h"
#include "../include/cblas.h"

float cblas_sasum(const FTBLAS_INT n, const float *x, const FTBLAS_INT inc_x) {
#ifdef FT_ENABLED
    ftblas_sasum_ft(n, x, inc_x);
#else
    ftblas_sasum_ori(n, x, inc_x);
#endif
}