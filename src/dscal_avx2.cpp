#include <stdio.h>
#include <stdlib.h>

void ori_dscal_compute(long int n, double a, double *x, long int inc_x)
{
	for(int i = 0; i < n; i++){
        x[i*inc_x] = a * x[i*inc_x];
    }
}

// a driver layer useful for threaded version
void ori_dscal(long int n, double a, double *x, long int inc_x)
{
	ori_dscal_compute(n, a, x, inc_x);
}