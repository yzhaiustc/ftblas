#include <stdio.h>
#include <stdlib.h>
#include "immintrin.h"

void ori_dscal_compute(long int n, double scalar, double *x, long int inc_x)
{
    if (inc_x != 1) {
        for(int i = 0; i < n; i++){
            x[i*inc_x] = scalar * x[i*inc_x];
        }

        return;
    }

    // when inc_x == 1, we do this branch - the optimized one.
    int n16 = n & -16;

    __m256d scalar_tile = _mm256_set1_pd(scalar);
    
    double * curr_x_ptr = x;
    
    for (int i = 0; i < n16; i += 16) {

        __m256d x_tile_1 = _mm256_loadu_pd(curr_x_ptr);
        __m256d x_tile_2 = _mm256_loadu_pd(curr_x_ptr + 4);
        __m256d x_tile_3 = _mm256_loadu_pd(curr_x_ptr + 8);
        __m256d x_tile_4 = _mm256_loadu_pd(curr_x_ptr + 12);
        
        _mm256_storeu_pd(curr_x_ptr, x_tile_1 * scalar_tile);
        _mm256_storeu_pd(curr_x_ptr + 4, x_tile_2 * scalar_tile);
        _mm256_storeu_pd(curr_x_ptr + 8, x_tile_3 * scalar_tile);
        _mm256_storeu_pd(curr_x_ptr + 12, x_tile_4 * scalar_tile);

        curr_x_ptr += 16;
    }

    for (int i = n16; i < n; i++) {
        *curr_x_ptr = scalar * (*curr_x_ptr);
        curr_x_ptr++;
    }

}

// a driver layer useful for threaded version
void ori_dscal(long int n, double a, double *x, long int inc_x)
{
    ori_dscal_compute(n, a, x, inc_x);
}