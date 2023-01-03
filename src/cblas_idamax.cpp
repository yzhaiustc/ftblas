#include "../include/ftblas.h"
#include "../include/cblas.h"

FTBLAS_INT cblas_idamax(const FTBLAS_INT n, const double *x, const FTBLAS_INT inc_x) {
#ifdef FT_ENABLED
    ftblas_idamax_ft(n, x, inc_x);
#else
    ftblas_idamax_ori(n,  x, inc_x);
#endif
}