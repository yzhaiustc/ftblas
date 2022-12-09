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
    long int n16 = n & -16;

    double *ptr_scalar = &scalar;
    
    double * curr_x_ptr = x;

	__asm__ __volatile__(\
        "vbroadcastsd (%1), %%ymm0;"\
        "1:\n\t"\
        "vmovupd (%2), %%ymm1;"\
        "vmovupd 32(%2), %%ymm2;"\
        "vmovupd 64(%2), %%ymm3;"\
        "vmovupd 96(%2), %%ymm4;"\
        "vmulpd %%ymm0, %%ymm1, %%ymm1;"\
        "vmulpd %%ymm0, %%ymm2, %%ymm2;"\
        "vmulpd %%ymm0, %%ymm3, %%ymm3;"\
        "vmulpd %%ymm0, %%ymm4, %%ymm4;"\
        "vmovupd %%ymm1, (%2);"\
        "vmovupd %%ymm2, 32(%2);"\
        "vmovupd %%ymm3, 64(%2);"\
        "vmovupd %%ymm4, 96(%2);"\
        "addq $128, %2;"\
		"subq $16,%0;"\
		"jnz 1b;"\
		"vzeroupper;"\
        :"+r"(n16),"+r"(ptr_scalar),"+r"(curr_x_ptr)\
        :\
        :"cc","memory","ymm0","ymm1","ymm2","ymm3","ymm4","ymm5","ymm6","ymm7","ymm8","ymm9","ymm10","ymm11","ymm12","ymm13","ymm14","ymm15"\
    );\



    for (int i = (n & -16); i < n; i++) {
        *curr_x_ptr = scalar * (*curr_x_ptr);
        curr_x_ptr++;
    }

}

// a driver layer useful for threaded version
void ori_dscal(long int n, double a, double *x, long int inc_x)
{
    ori_dscal_compute(n, a, x, inc_x);
}