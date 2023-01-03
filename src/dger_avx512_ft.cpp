#include "../include/ftblas.h"

static int dger_ft_kernel_sp(long int n, double *alpha, double *a1, double *a2, double *a3, double *a4, double *x, double *y, long int *status)
{
    register long int i = 0;
    register long int err = 0;
    __asm__ __volatile__(
        "vbroadcastsd       (%4) , %%zmm4                    \n\t" // LX1
        "vbroadcastsd      8(%4) , %%zmm5                    \n\t" // LX2
        "vbroadcastsd     16(%4) , %%zmm6                    \n\t" // LX3
        "vbroadcastsd     24(%4) , %%zmm7                    \n\t" // LX4

        "kxnorw             %%k4 , %%k4 , %%k4               \n\t"

        // Prologue
        "vmovups        (%6,%0,8), %%zmm16                   \n\t" // LA1

        "vmovups        (%5,%0,8), %%zmm20                   \n\t" // LY
        "vmovups          %%zmm16, %%zmm12                   \n\t" // BA1
        "vmovups          %%zmm16, %%zmm26                   \n\t" // BB1
        "vmovups        (%7,%0,8), %%zmm17                   \n\t" // LA2

        "vfmadd231pd      %%zmm20, %%zmm4 , %%zmm16          \n\t" // F1
        "vfmadd231pd      %%zmm20, %%zmm4 , %%zmm12          \n\t" // F1 - dup - 16 - 12
        "vmovups          %%zmm17, %%zmm13                   \n\t" // BA2
        "vmovups          %%zmm17, %%zmm27                   \n\t" // BB2
        "vmovups        (%8,%0,8), %%zmm18                   \n\t" // LA3

        "vpcmpeqd         %%zmm12, %%zmm16, %%k0             \n\t" // C1
        "vfmadd231pd      %%zmm20, %%zmm5 , %%zmm17          \n\t" // F2
        "vfmadd231pd      %%zmm20, %%zmm5 , %%zmm13          \n\t" // F2 - dup - 17 - 13
        "vmovups          %%zmm18, %%zmm14                   \n\t" // BA3
        "vmovups          %%zmm18, %%zmm28                   \n\t" // BB3
        "vmovups        (%9,%0,8), %%zmm19                   \n\t" // LA4

        "addq       $8 , %0                                  \n\t"
        "subq       $8 , %1                                  \n\t"
        "jz         2f                                       \n\t"

        ".p2align 4                                          \n\t"
        "1:                                                  \n\t"
        "vmovups          %%zmm26, %%zmm30                   \n\t" // CHPT-1
        "vmovups       %%zmm16, -64(%6,%0,8)                 \n\t" // SA1
        "vpcmpeqd         %%zmm13, %%zmm17, %%k1             \n\t" // C2
        "vfmadd231pd      %%zmm20, %%zmm6 , %%zmm18          \n\t" // F3
        "vfmadd231pd      %%zmm20, %%zmm6 , %%zmm14          \n\t" // F3 - dup - 18 - 14
        "vmovups          %%zmm19, %%zmm15                   \n\t" // BA4
        "vmovups          %%zmm19, %%zmm29                   \n\t" // BB4
        "vmovups        (%6,%0,8), %%zmm16                   \n\t" // LA1

        "kandw            %%k0   , %%k1   , %%k0             \n\t" // red - 0 - 1 -> 0
        "vmovups          %%zmm27, %%zmm31                   \n\t" // CHPT-2
        "vmovups       %%zmm17, -64(%7,%0,8)                 \n\t" // SA2
        "vpcmpeqd         %%zmm14, %%zmm18, %%k2             \n\t" // C3
        "vfmadd231pd      %%zmm20, %%zmm7 , %%zmm19          \n\t" // F4
        "vfmadd231pd      %%zmm20, %%zmm7 , %%zmm15          \n\t" // F4 - dup - 19 - 15
        "vmovups        (%5,%0,8), %%zmm20                   \n\t" // LY
        "vmovups          %%zmm16, %%zmm12                   \n\t" // BA1
        "vmovups          %%zmm16, %%zmm26                   \n\t" // BB1
        "vmovups        (%7,%0,8), %%zmm17                   \n\t" // LA2

        "vmovups          %%zmm28, %%zmm24                   \n\t" // CHPT-3
        "vmovups       %%zmm18, -64(%8,%0,8)                 \n\t" // SA3
        "kandw            %%k0   , %%k2   , %%k0             \n\t" // red - 0 - 2 -> 0
        "vpcmpeqd         %%zmm15, %%zmm19, %%k3             \n\t" // C4
        "vfmadd231pd      %%zmm20, %%zmm4 , %%zmm16          \n\t" // F1
        "vfmadd231pd      %%zmm20, %%zmm4 , %%zmm12          \n\t" // F1 - dup - 16 - 12
        "vmovups          %%zmm17, %%zmm13                   \n\t" // BA2
        "kandw            %%k0   , %%k3   , %%k3             \n\t" // red - 0 - 3 -> 3
        "vmovups          %%zmm17, %%zmm27                   \n\t" // BB2
        "vmovups        (%8,%0,8), %%zmm18                   \n\t" // LA3
        "kxorw            %%k3   , %%k4   , %%k3             \n\t"
        "vmovups          %%zmm29, %%zmm25                   \n\t" // CHPT-4
        "vmovups       %%zmm19, -64(%9,%0,8)                 \n\t" // SA4
        "vpcmpeqd         %%zmm12, %%zmm16, %%k0             \n\t" // C1
        "vfmadd231pd      %%zmm20, %%zmm5 , %%zmm17          \n\t" // F2
        "vfmadd231pd      %%zmm20, %%zmm5 , %%zmm13          \n\t" // F2 - dup - 17 - 13
        "vmovups          %%zmm18, %%zmm14                   \n\t" // BA3
        "vmovups          %%zmm18, %%zmm28                   \n\t" // BB3
        "vmovups        (%9,%0,8), %%zmm19                   \n\t" // LA4

        "addq       $8 , %0                                  \n\t"
        "ktestw    %%k3, %%k4                                \n\t"
        "jnz        3f                                       \n\t"

        "subq       $8 , %1                                  \n\t"
        "jnz        1b                                       \n\t"

        // Epilogue
        "2:                                                  \n\t"
        "vmovups       %%zmm16, -64(%6,%0,8)                 \n\t" // SA1
        "vpcmpeqd         %%zmm13, %%zmm17, %%k1             \n\t" // C2
        "vfmadd231pd      %%zmm20, %%zmm6 , %%zmm18          \n\t" // F3
        "vfmadd231pd      %%zmm20, %%zmm6 , %%zmm14          \n\t" // F3 - dup - 18 - 14
        "vmovups          %%zmm19, %%zmm25                   \n\t" // BA4
        "vmovups          %%zmm19, %%zmm29                   \n\t" // BB4

         "kandw            %%k0   , %%k1   , %%k0             \n\t" // red - 0 - 1 -> 0
        "vmovups       %%zmm17, -64(%7,%0,8)                 \n\t" // SA2
        "vpcmpeqd         %%zmm14, %%zmm18, %%k2             \n\t" // C3
        "vfmadd231pd      %%zmm20, %%zmm7 , %%zmm19          \n\t" // F4
        "vfmadd231pd      %%zmm20, %%zmm7 , %%zmm15          \n\t" // F4 - dup - 19 - 15

        "vmovups       %%zmm18, -64(%8,%0,8)                 \n\t" // SA3
        "kandw            %%k0   , %%k2   , %%k0             \n\t" // red - 0 - 2 -> 0
        "vpcmpeqd         %%zmm15, %%zmm19, %%k3             \n\t" // C4

        "vmovups       %%zmm19, -64(%9,%0,8)                 \n\t" // SA4
        "kandw            %%k0   , %%k3   , %%k0             \n\t" // red - 0 - 3 -> 0
        "kxorw            %%k0   , %%k4   , %%k4             \n\t"
        "ktestw    %%k0, %%k4                                \n\t"
        "jnz        4f                                       \n\t"
        
        "jmp        6f                                       \n\t"

        // error handler to loop body
        "3:                                                  \n\t"
        "addq       $1 , %2                                  \n\t"
        "subq       $8 , %0                                  \n\t"
        "vmovups          %%zmm30, %%zmm16                   \n\t" // LA1

        "vmovups     -64(%5,%0,8), %%zmm20                   \n\t" // LY
        "vmovups          %%zmm16, %%zmm12                   \n\t" // BA1
        "vmovups          %%zmm16, %%zmm26                   \n\t" // BB1
        "vmovups        %%zmm31, %%zmm17                   \n\t" // LA2

        "vfmadd231pd      %%zmm20, %%zmm4 , %%zmm16          \n\t" // F1
        "vfmadd231pd      %%zmm20, %%zmm4 , %%zmm12          \n\t" // F1 - dup - 16 - 12
        "vmovups          %%zmm17, %%zmm13                   \n\t" // BA2
        "vmovups          %%zmm17, %%zmm27                   \n\t" // BB2
        "vmovups          %%zmm24, %%zmm18                   \n\t" // LA3

        "vpcmpeqd         %%zmm12, %%zmm16, %%k0             \n\t" // C1
        "vfmadd231pd      %%zmm20, %%zmm5 , %%zmm17          \n\t" // F2
        "vfmadd231pd      %%zmm20, %%zmm5 , %%zmm13          \n\t" // F2 - dup - 17 - 13
        "vmovups          %%zmm18, %%zmm14                   \n\t" // BA3
        "vmovups          %%zmm18, %%zmm28                   \n\t" // BB3
        "vmovups          %%zmm25, %%zmm19                   \n\t" // LA4

        "vmovups          %%zmm26, %%zmm30                   \n\t" // CHPT-1
        "vmovups       %%zmm16, -64(%6,%0,8)                 \n\t" // SA1
        "vpcmpeqd         %%zmm13, %%zmm17, %%k1             \n\t" // C2
        "vfmadd231pd      %%zmm20, %%zmm6 , %%zmm18          \n\t" // F3
        "vfmadd231pd      %%zmm20, %%zmm6 , %%zmm14          \n\t" // F3 - dup - 18 - 14
        "vmovups          %%zmm19, %%zmm15                   \n\t" // BA4
        "vmovups          %%zmm19, %%zmm29                   \n\t" // BB4
        "vmovups        (%6,%0,8), %%zmm16                   \n\t" // LA1

        "kandw            %%k0   , %%k1   , %%k0             \n\t" // red - 0 - 1 -> 0
        "vmovups          %%zmm27, %%zmm31                   \n\t" // CHPT-2
        "vmovups       %%zmm17, -64(%7,%0,8)                 \n\t" // SA2
        "vpcmpeqd         %%zmm14, %%zmm18, %%k2             \n\t" // C3
        "vfmadd231pd      %%zmm20, %%zmm7 , %%zmm19          \n\t" // F4
        "vfmadd231pd      %%zmm20, %%zmm7 , %%zmm15          \n\t" // F4 - dup - 19 - 15
        "vmovups        (%5,%0,8), %%zmm20                   \n\t" // LY
        "vmovups          %%zmm16, %%zmm12                   \n\t" // BA1
        "vmovups          %%zmm16, %%zmm26                   \n\t" // BB1
        "vmovups        (%7,%0,8), %%zmm17                   \n\t" // LA2

        "vmovups          %%zmm28, %%zmm24                   \n\t" // CHPT-3
        "vmovups       %%zmm18, -64(%8,%0,8)                 \n\t" // SA3
        "kandw            %%k0   , %%k2   , %%k0             \n\t" // red - 0 - 2 -> 0
        "vpcmpeqd         %%zmm15, %%zmm19, %%k3             \n\t" // C4
        "vfmadd231pd      %%zmm20, %%zmm4 , %%zmm16          \n\t" // F1
        "vfmadd231pd      %%zmm20, %%zmm4 , %%zmm12          \n\t" // F1 - dup - 16 - 12
        "vmovups          %%zmm17, %%zmm13                   \n\t" // BA2
        "kandw            %%k0   , %%k3   , %%k3             \n\t" // red - 0 - 3 -> 3
        "vmovups          %%zmm17, %%zmm27                   \n\t" // BB2
        "vmovups        (%8,%0,8), %%zmm18                   \n\t" // LA3
        "kxorw            %%k3   , %%k4   , %%k3             \n\t"
        "vmovups          %%zmm29, %%zmm25                   \n\t" // CHPT-4
        "vmovups       %%zmm19, -64(%9,%0,8)                 \n\t" // SA4
        "vpcmpeqd         %%zmm12, %%zmm16, %%k0             \n\t" // C1
        "vfmadd231pd      %%zmm20, %%zmm5 , %%zmm17          \n\t" // F2
        "vfmadd231pd      %%zmm20, %%zmm5 , %%zmm13          \n\t" // F2 - dup - 17 - 13
        "vmovups          %%zmm18, %%zmm14                   \n\t" // BA3
        "vmovups          %%zmm18, %%zmm28                   \n\t" // BB3
        "vmovups        (%9,%0,8), %%zmm19                   \n\t" // LA4

        "addq       $8 , %0                                  \n\t"

        "ktestw    %%k3, %%k4                                \n\t"
        "jnz        5f                                       \n\t"

        "subq       $8 , %1                                  \n\t"
        "jnz        1b                                       \n\t"
        "jmp        2b                                       \n\t"

        // error handler to epilogue
        "4:                                                  \n\t"
        "addq       $1 , %2                                  \n\t"
        "vmovups          %%zmm30, %%zmm16                   \n\t" // LA1

        "vmovups     -64(%5,%0,8), %%zmm20                   \n\t" // LY
        "vmovups          %%zmm16, %%zmm12                   \n\t" // BA1
        "vmovups          %%zmm16, %%zmm26                   \n\t" // BB1
        "vmovups        %%zmm31, %%zmm17                   \n\t" // LA2

        "vfmadd231pd      %%zmm20, %%zmm4 , %%zmm16          \n\t" // F1
        "vfmadd231pd      %%zmm20, %%zmm4 , %%zmm12          \n\t" // F1 - dup - 16 - 12
        "vmovups          %%zmm17, %%zmm13                   \n\t" // BA2
        "vmovups          %%zmm17, %%zmm27                   \n\t" // BB2
        "vmovups          %%zmm24, %%zmm18                   \n\t" // LA3

        "vpcmpeqd         %%zmm12, %%zmm16, %%k0             \n\t" // C1
        "vfmadd231pd      %%zmm20, %%zmm5 , %%zmm17          \n\t" // F2
        "vfmadd231pd      %%zmm20, %%zmm5 , %%zmm13          \n\t" // F2 - dup - 17 - 13
        "vmovups          %%zmm18, %%zmm14                   \n\t" // BA3
        "vmovups          %%zmm18, %%zmm28                   \n\t" // BB3
        "vmovups          %%zmm25, %%zmm19                   \n\t" // LA4

        "vmovups          %%zmm26, %%zmm30                   \n\t" // CHPT-1
        "vmovups       %%zmm16, -64(%6,%0,8)                 \n\t" // SA1
        "vpcmpeqd         %%zmm13, %%zmm17, %%k1             \n\t" // C2
        "vfmadd231pd      %%zmm20, %%zmm6 , %%zmm18          \n\t" // F3
        "vfmadd231pd      %%zmm20, %%zmm6 , %%zmm14          \n\t" // F3 - dup - 18 - 14
        "vmovups          %%zmm19, %%zmm15                   \n\t" // BA4
        "vmovups          %%zmm19, %%zmm29                   \n\t" // BB4

        "kandw            %%k0   , %%k1   , %%k0             \n\t" // red - 0 - 1 -> 0
        "vmovups          %%zmm27, %%zmm31                   \n\t" // CHPT-2
        "vmovups       %%zmm17, -64(%7,%0,8)                 \n\t" // SA2
        "vpcmpeqd         %%zmm14, %%zmm18, %%k2             \n\t" // C3
        "vfmadd231pd      %%zmm20, %%zmm7 , %%zmm19          \n\t" // F4
        "vfmadd231pd      %%zmm20, %%zmm7 , %%zmm15          \n\t" // F4 - dup - 19 - 15


        "vmovups          %%zmm28, %%zmm24                   \n\t" // CHPT-3
        "vmovups       %%zmm18, -64(%8,%0,8)                 \n\t" // SA3
        "kandw            %%k0   , %%k2   , %%k0             \n\t" // red - 0 - 2 -> 0
        "vpcmpeqd         %%zmm15, %%zmm19, %%k3             \n\t" // C4
        "kandw            %%k0   , %%k3   , %%k0             \n\t" // red - 0 - 3 -> 0

        "kxorw            %%k3   , %%k4   , %%k3             \n\t"
        "vmovups       %%zmm19, -64(%9,%0,8)                 \n\t" // SA4

        "ktestw    %%k3, %%k4                                \n\t"
        "jnz        5f                                       \n\t"
        "jmp        6f                                       \n\t"

        // still incorrect
        "5:                                                  \n\t"
        "movq              $-1, (%10)                        \n\t"
        // reduction - nothing to reduce
        "6:                                                  \n\t"
        "vzeroupper                                          \n\t"
        :
            "+r" (i),    // 0   
            "+r" (n),    // 1
            "+r" (err)   // 2
        :
            "r" (alpha), // 3
            "r" (x),     // 4
            "r" (y),     // 5
            "r" (a1),    // 6
            "r" (a2),    // 7
            "r" (a3),    // 8
            "r" (a4),    // 9
            "r" (status) // 10
        : "cc", 
        "%xmm4", "%xmm5", "%xmm6", "%xmm7",
        "%xmm12", "%xmm13", "%xmm14", "%xmm15",
        "memory"
    );
    return err;
}

void ftblas_dger_ft(long lda, long m, long n, double alpha, double *A, double *x, double *y)
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
                long int status = 0;
                long int err_num = dger_ft_kernel_sp(n8, &alpha, ptr_a1, ptr_a2, ptr_a3, ptr_a4, ptr_xi, y, &status);
                if (err_num) 
                {
                    printf("detected %ld error, error status is %ld\n", err_num, status);
                }
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
                long int status = 0;
                long int err_num = dger_ft_kernel_sp(n8, &alpha, ptr_a1, ptr_a2, ptr_a3, ptr_a4, ptr_xi, y, &status);
                if (err_num) 
                {
                    printf("detected %ld error, error status is %ld\n", err_num, status);
                }
            }

            j = n8;

            register double x1 = x[i];
            register double x2 = x[i + 1];
            register double x3 = x[i + 2];
            register double x4 = x[i + 3];
            

            while (j < n)
            {
                register int in = i * lda + j;
                register double yj = y[j];
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
            A[i * lda + j] += x1 * y[j];
        }
    }
}