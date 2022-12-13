/***
 * 
*/

#ifndef _FTBLAS_H_
#define _FTBLAS_H_

#include <math.h>
#include <sys/time.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// double ft_dnrm2(long int n, double *x, long int inc_x);
// double ori_dnrm2(long int n, double *x, long int inc_x);
// void ori_daxpy(long int n, double a, double *x, long int inc_x, double *y, long int inc_y);
// void ori_dscal(long int n, double a, double *x, long int inc_x);

// double ft_ddot(long int n, double *x, long int inc_x, double *y, long int inc_y);
double cblas_ddot(long int n, double *x, long int inc_x, double *y, long int inc_y);

#endif