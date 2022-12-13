#include <stdio.h>
#include <stdlib.h>

void ori_daxpy_compute(long int n, double a, double *x, long int inc_x, double *y, long int inc_y)
{
	for(int i = 0; i < n; i++){
        y[i*inc_y] += a * x[i*inc_x];
    }
}

// a driver layer useful for threaded version
void ori_daxpy(long int n, double a, double *x, long int inc_x, double *y, long int inc_y)
{
	ori_daxpy_compute(n, a, x, inc_x, y, inc_y);
}