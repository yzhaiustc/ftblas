#include "../include/ftblas.h"
#include "../include/cblas.h"

double cblas_ddot(long int n, double *x, long int inc_x, double *y, long int inc_y) {
#ifdef FT_ENABLED
        return ftblas_ddot_ft(n, x, inc_x, y, inc_y);
#else
		return ftblas_ddot_ori(n, x, inc_x, y, inc_y);
#endif
}