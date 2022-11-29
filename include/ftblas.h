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

double ft_dnrm2(long int n, double *x, long int inc_x);
double ori_dnrm2(long int n, double *x, long int inc_x);

#endif