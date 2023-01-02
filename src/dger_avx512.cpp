#include "../include/ftblas.h"

static void dger_kernel_sp(long int n, double *alpha, double *a1, double *a2, double *a3, double *a4, double *x, double *y)
{
    register long int i = 0;

    __asm__ __volatile__(
        "vbroadcastsd       (%3) , %%zmm4                    \n\t" // LX1
        "vbroadcastsd      8(%3) , %%zmm5                    \n\t" // LX2
        "vbroadcastsd     16(%3) , %%zmm6                    \n\t" // LX3
        "vbroadcastsd     24(%3) , %%zmm7                    \n\t" // LX4

        "vbroadcastsd       (%2) , %%zmm3                    \n\t" // alpha

        // Prologue
        "vmovups        (%4,%0,8), %%zmm20                   \n\t" // LY

        "vmovups        (%5,%0,8), %%zmm16                   \n\t" // LA1

        "vmulpd           %%zmm3 , %%zmm20, %%zmm21          \n\t" // MY
        "vmovups        (%6,%0,8), %%zmm17                   \n\t" // LA2

        "vfmadd231pd      %%zmm21, %%zmm4 , %%zmm16          \n\t" // F1
        "vmovups        (%7,%0,8), %%zmm18                   \n\t" // LA3

        "addq       $8 , %0                                  \n\t"
        "subq       $8 , %1                                  \n\t"
        "jz         2f                                       \n\t"

        ".p2align 4                                          \n\t"
        "1:                                                  \n\t"
        "prefetchw      64(%8,%0,8)                          \n\t"
        "vmovups       %%zmm16, -64(%5,%0,8)                 \n\t" // SA1
        "vfmadd231pd      %%zmm21, %%zmm5 , %%zmm17          \n\t" // F2
        "vmovups     -64(%8,%0,8), %%zmm19                   \n\t" // LA4
        "vmovups        (%4,%0,8), %%zmm20                   \n\t" // LY

        "prefetchw      128(%5,%0,8)                          \n\t"
        "vmovups       %%zmm17, -64(%6,%0,8)                 \n\t" // SA2
        "vfmadd231pd      %%zmm21, %%zmm6 , %%zmm18          \n\t" // F3
        "vmovups        (%5,%0,8), %%zmm16                   \n\t" // LA1

        "prefetchw      128(%6,%0,8)                          \n\t"
        "vmovups       %%zmm18, -64(%7,%0,8)                 \n\t" // SA3
        "vfmadd231pd      %%zmm21, %%zmm7 , %%zmm19          \n\t" // F4
        "vmulpd           %%zmm3 , %%zmm20, %%zmm21          \n\t" // MY
        "vmovups        (%6,%0,8), %%zmm17                   \n\t" // LA2

        "prefetchw      128(%7,%0,8)                          \n\t"
        "vmovups       %%zmm19, -64(%8,%0,8)                 \n\t" // SA4
        "vfmadd231pd      %%zmm21, %%zmm4 , %%zmm16          \n\t" // F1
        "vmovups        (%7,%0,8), %%zmm18                   \n\t" // LA3

        "addq       $8 , %0                                  \n\t"
        "subq       $8 , %1                                  \n\t"
        "jnz        1b                                       \n\t"

        // Epilogue
        "2:                                                  \n\t"
        "vmovups       %%zmm16, -64(%5,%0,8)                 \n\t" // SA1
        "vfmadd231pd      %%zmm21, %%zmm5 , %%zmm17          \n\t" // F2
        "vmovups     -64(%8,%0,8), %%zmm19                   \n\t" // LA4

        "vmovups       %%zmm17, -64(%6,%0,8)                 \n\t" // SA2
        "vfmadd231pd      %%zmm21, %%zmm6 , %%zmm18          \n\t" // F3

        "vmovups       %%zmm18, -64(%7,%0,8)                 \n\t" // SA3
        "vfmadd231pd      %%zmm21, %%zmm7 , %%zmm19          \n\t" // F4

        "vmovups       %%zmm19, -64(%8,%0,8)                 \n\t" // SA4
        
        // reduction - nothing to reduce
        "3:                                                  \n\t"
        "vzeroupper                                          \n\t"
        :
            "+r" (i),    // 0   
            "+r" (n)     // 1
        :
            "r" (alpha), // 2
            "r" (x),     // 3
            "r" (y),     // 4
            "r" (a1),    // 5
            "r" (a2),    // 6
            "r" (a3),    // 7
            "r" (a4)     // 8
        : "cc", 
        "%xmm4", "%xmm5", "%xmm6", "%xmm7",
        "%xmm12", "%xmm13", "%xmm14", "%xmm15",
        "memory"
    );
}

void ftblas_dger_ori(long lda, long m, long n, double alpha, double *A, double *x, double *y)
{
    int i, j;
    int m4 = m & -4;
    int n8 = n & -8;
    double *ptr_a1, *ptr_a2, *ptr_a3, *ptr_a4, *ptr_ai, *ptr_xi;

    if (n8 == n)
    {
        for (i = 0; i < m4; i += 4)
        {
            ptr_ai = A + i * lda;
            ptr_xi = x + i;
            if (n8)
            {
                ptr_a1 = ptr_ai;
                ptr_a2 = ptr_ai + lda;
                ptr_a3 = ptr_ai + lda * 2;
                ptr_a4 = ptr_ai + lda * 3;
                dger_kernel_sp(n8, &alpha, ptr_a1, ptr_a2, ptr_a3, ptr_a4, ptr_xi, y);
            }
            
        }
    }
    else
    {
        for (i = 0; i < m4; i += 4)
        {
            ptr_ai = A + i * lda;
            ptr_xi = x + i;
            if (n8)
            {
                ptr_a1 = ptr_ai;
                ptr_a2 = ptr_ai + lda;
                ptr_a3 = ptr_ai + lda * 2;
                ptr_a4 = ptr_ai + lda * 3;
                dger_kernel_sp(n8, &alpha, ptr_a1, ptr_a2, ptr_a3, ptr_a4, ptr_xi, y);
            }

            j = n8;

            register double x1 = x[i];
            register double x2 = x[i + 1];
            register double x3 = x[i + 2];
            register double x4 = x[i + 3];
            

            while (j < n)
            {
                register int in = i * lda + j;
                register double yj = alpha * y[j];
                register int n2 = lda + lda;
                register int in2 = in + lda;
                A[in] += x1 * yj;
                A[in2] += x2 * yj;
                A[in + n2] += x3 * yj;
                A[in2 + n2] += x4 * yj;
                j++;
            }
            
        }
    }
    


    if (m4 == m) return;

    for (i = m4; i < m; i++)
    {
        register double x1 = x[i];

        for (j = 0; j < n; j++)
        {
            A[i * lda + j] += alpha * x1 * y[j];
        }
    }
}