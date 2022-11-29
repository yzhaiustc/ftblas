#ifndef _UTIL_H_
#define _UTIL_H_

void randomize_matrix(double *A, int m, int n);
double get_sec();
void print_matrix(const double *A, int m, int n);
void print_vector(double *vec, int n);
void copy_matrix(double *src, double *dest, int n);
void verify_matrix(double *mat1, double *mat2, int n);

#endif // _UTIL_H_