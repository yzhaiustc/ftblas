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

double ftblas_ddot_ori(long int n, double *x, long int inc_x, double *y, long int inc_y);
double ftblas_ddot_ft(long int n, double *x, long int inc_x, double *y, long int inc_y);

#endif