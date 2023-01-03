#include "../include/ftblas.h"

static void dgemv_t_kernel(long int n, double *a1, double *a2, double *a3, double *a4, double *x, double *y)
{
    register long int i = 0;
    
    __asm__ __volatile__(
        "vxorps		%%zmm4 , %%zmm4, %%zmm4                 \n\t"
        "vxorps		%%zmm5 , %%zmm5, %%zmm5                 \n\t"
        "vxorps		%%zmm6 , %%zmm6, %%zmm6                 \n\t"
        "vxorps		%%zmm7 , %%zmm7, %%zmm7                 \n\t"

        ".p2align 4                          \n\t"
        "1:                     			 \n\t"
        "vmovups      (%2,%0,8), %%zmm12                    \n\t" // LX
        "vfmadd231pd  (%4,%0,8), %%zmm12, %%zmm4            \n\t" // F1
        "vfmadd231pd  (%5,%0,8), %%zmm12, %%zmm5            \n\t" // F2
        "vfmadd231pd  (%6,%0,8), %%zmm12, %%zmm6            \n\t" // F3
        "vfmadd231pd  (%7,%0,8), %%zmm12, %%zmm7            \n\t" // F4

        "addq       $8 , %0                                 \n\t"
        "subq       $8 , %1                                 \n\t"
        "jnz        1b                                      \n\t"

        "vextractf64x4     $0, %%zmm4 , %%ymm16             \n\t"
        "vextractf64x4     $0, %%zmm5 , %%ymm17             \n\t"
        "vextractf64x4     $0, %%zmm6 , %%ymm18             \n\t"
        "vextractf64x4     $0, %%zmm7 , %%ymm19             \n\t"
        
        "vextractf64x4     $1, %%zmm4 , %%ymm4              \n\t"
        "vextractf64x4     $1, %%zmm5 , %%ymm5              \n\t"
        "vextractf64x4     $1, %%zmm6 , %%ymm6              \n\t"
        "vextractf64x4     $1, %%zmm7 , %%ymm7              \n\t"

        "vaddpd       %%ymm4 , %%ymm16, %%ymm4              \n\t"
        "vaddpd       %%ymm5 , %%ymm17, %%ymm5              \n\t"
        "vaddpd       %%ymm6 , %%ymm18, %%ymm6              \n\t"
        "vaddpd       %%ymm7 , %%ymm19, %%ymm7              \n\t"

        "vextractf128   $1 , %%ymm4, %%xmm12	      \n\t"
        "vextractf128   $1 , %%ymm5, %%xmm13	      \n\t"
        "vextractf128   $1 , %%ymm6, %%xmm14	      \n\t"
        "vextractf128   $1 , %%ymm7, %%xmm15	      \n\t"

        "vaddpd		%%xmm4, %%xmm12, %%xmm4       \n\t"
        "vaddpd		%%xmm5, %%xmm13, %%xmm5       \n\t"
        "vaddpd		%%xmm6, %%xmm14, %%xmm6       \n\t"
        "vaddpd		%%xmm7, %%xmm15, %%xmm7       \n\t"

        "vhaddpd        %%xmm4, %%xmm4, %%xmm4  \n\t"
        "vhaddpd        %%xmm5, %%xmm5, %%xmm5  \n\t"
        "vhaddpd        %%xmm6, %%xmm6, %%xmm6  \n\t"
        "vhaddpd        %%xmm7, %%xmm7, %%xmm7  \n\t"

        "vmovsd         %%xmm4,    (%3)         \n\t"
        "vmovsd         %%xmm5,   8(%3)         \n\t"
        "vmovsd         %%xmm6,  16(%3)         \n\t"
        "vmovsd         %%xmm7,  24(%3)         \n\t"

        "vzeroupper			 \n\t"

        :
            "+r" (i),	// 0	
            "+r" (n)  	// 1
        :
            "r" (x),    // 2
            "r" (y),    // 3
            "r" (a1),   // 4
            "r" (a2),   // 5
            "r" (a3),   // 6
            "r" (a4)    // 7
        : "cc", 
        "%xmm4", "%xmm5", "%xmm6", "%xmm7",
        "%xmm12", "%xmm13", "%xmm14", "%xmm15",
        "memory"
    );
}

void ftblas_dgemv_t_ori(double *A, double *x, double *y, int m, int n, int lda)
{
    int m4 = m & -4;
    int n4 = n & -4;
    int n8 = n & -8;
    int i, j;
    double y_buffer[4];

    for (i = 0; i < m4; i += 4)
    {
        register double y0, y1, y2, y3;
        y0 = y[i];
        y1 = y[i + 1];
        y2 = y[i + 2];
        y3 = y[i + 3];

        if (n8)
        {
            double *a1 = A + i * lda;
            double *a2 = A + (i + 1) * lda;
            double *a3 = A + (i + 2) * lda;
            double *a4 = A + (i + 3) * lda;
            dgemv_t_kernel(n8, a1, a2, a3, a4, x, y_buffer);
            y0 += y_buffer[0];
            y1 += y_buffer[1];
            y2 += y_buffer[2];
            y3 += y_buffer[3];
            
        }

        j = n8;

        while (j < n)
        {
            y0 += A[i * lda + j] * x[j];
            y1 += A[(i+1) * lda + j] * x[j];
            y2 += A[(i+2) * lda + j] * x[j];
            y3 += A[(i+3) * lda + j] * x[j];
            j++;
        }

        y[i] = y0;
        y[i+1] = y1;
        y[i+2] = y2;
        y[i+3] = y3;
    }

    while (i < m)
    {
        for (j = 0; j < n; j++)
        {
            y[i] += A[i * lda + j] * x[j];
        }
        i++;
    }

}