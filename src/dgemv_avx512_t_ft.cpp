#include "../include/ftblas.h"

static long dgemv_t_ft_kernel(long int n, double *a1, double *a2, double *a3, double *a4, double *x, double *y)
{
    register long int i = 0;
    register long int err_num = 0;
    __asm__ __volatile__(
        "vxorps       %%zmm4 , %%zmm4 , %%zmm4              \n\t"
        "vxorps       %%zmm5 , %%zmm5 , %%zmm5              \n\t"
        "vxorps       %%zmm6 , %%zmm6 , %%zmm6              \n\t"
        "vxorps       %%zmm7 , %%zmm7 , %%zmm7              \n\t"

        "vxorps       %%zmm8 , %%zmm8 , %%zmm8              \n\t"
        "vxorps       %%zmm9 , %%zmm9 , %%zmm9              \n\t"
        "vxorps           %%zmm10, %%zmm10, %%zmm10             \n\t"
        "vxorps       %%zmm11, %%zmm11, %%zmm11             \n\t"

        "kxnorw       %%k4   , %%k4   , %%k4                \n\t"

        // Prologue
        "vmovups    (%3,%0,8), %%zmm12                      \n\t" // LX

        "vmovups      %%zmm4 , %%zmm28                      \n\t" // B1
        "vmovups    (%5,%0,8), %%zmm16                      \n\t" // LA1
        
        "vfmadd231pd  %%zmm16, %%zmm12, %%zmm4              \n\t" // F1
        "vmovups      %%zmm5 , %%zmm29                      \n\t" // B2
        "vmovups    (%6,%0,8), %%zmm17                      \n\t" // LA2
        "vmovups      %%zmm12, %%zmm27                      \n\t" // BX
        
        "vfmadd231pd  %%zmm16, %%zmm12, %%zmm8              \n\t" // F1 - dup - 4 - 8
        "vfmadd231pd  %%zmm17, %%zmm12, %%zmm5              \n\t" // F2
        "vmovups      %%zmm6 , %%zmm30                      \n\t" // B3
        "vmovups    (%7,%0,8), %%zmm18                      \n\t" // LA3

        "addq       $8 , %0                                 \n\t"
        "subq       $8 , %1                                 \n\t"
        "jz         2f                                      \n\t"

        ".p2align 4                                         \n\t"
        "1:                                                 \n\t"
        "vmovups      %%zmm28, %%zmm0                       \n\t" // B11 - 28 - 0
        "vpcmpeqd     %%zmm4 , %%zmm8 , %%k0                \n\t" // C1
        "vfmadd231pd  %%zmm17, %%zmm12, %%zmm9              \n\t" // F2 - dup - 5 - 9
        "vfmadd231pd  %%zmm18, %%zmm12, %%zmm6              \n\t" // F3
        "vmovups      %%zmm7 , %%zmm31                      \n\t" // B4
        "vmovups -64(%8,%0,8), %%zmm19                      \n\t" // LA4   
        "vmovups    (%3,%0,8), %%zmm12                      \n\t" // LX5

        "vmovups      %%zmm29, %%zmm1                       \n\t" // B22 - 29 - 1
        "vpcmpeqd     %%zmm5 , %%zmm9 , %%k1                \n\t" // C2
        "vfmadd231pd  %%zmm18, %%zmm27, %%zmm10             \n\t" // F3 - dup - 6 - 10
        "vfmadd231pd  %%zmm19, %%zmm27, %%zmm7              \n\t" // F4
        "vmovups      %%zmm4 , %%zmm28                      \n\t" // B5
        "vmovups    (%5,%0,8), %%zmm16                      \n\t" // LA5  

        "kandw        %%k0   , %%k1   , %%k0                \n\t" // check - red - 0 - 1
        "vmovups      %%zmm30, %%zmm2                       \n\t" // B22 - 29 - 1
        "vpcmpeqd     %%zmm6 , %%zmm10, %%k2                \n\t" // C3
        "vfmadd231pd  %%zmm19, %%zmm27, %%zmm11             \n\t" // F4 - dup - 7 - 11
        "vfmadd231pd  %%zmm16, %%zmm12, %%zmm4              \n\t" // F5
        "vmovups      %%zmm5 , %%zmm29                      \n\t" // B6
        "vmovups    (%6,%0,8), %%zmm17                      \n\t" // LA6  
        "kandw        %%k0   , %%k2   , %%k0                \n\t" // check - red - 0 - 2
        "vmovups      %%zmm12, %%zmm27                      \n\t" // BX

        "vmovups      %%zmm31, %%zmm3                       \n\t" // B22 - 29 - 3
        "vpcmpeqd     %%zmm7 , %%zmm11, %%k3                \n\t" // C4
        "vfmadd231pd  %%zmm16, %%zmm12, %%zmm8              \n\t" // F5 - dup - 4 - 8
        "vfmadd231pd  %%zmm17, %%zmm12, %%zmm5              \n\t" // F6
        "vmovups      %%zmm6 , %%zmm30                      \n\t" // B7
        "kandw        %%k0   , %%k3   , %%k0                \n\t" // check - red - 0 - 3
        "vmovups    (%7,%0,8), %%zmm18                      \n\t" // LA7

        "kxorw        %%k0   , %%k4   , %%k0                \n\t"

        "addq             $8     , %0                           \n\t"

        "ktestw       %%k0   , %%k4                         \n\t"
        "jnz           4f                                    \n\t"

        "subq           $8 , %1                          \n\t"
        "jnz            1b                               \n\t"

        // Epilogue
        "2:                                                 \n\t"
        "vmovups      %%zmm28, %%zmm0                       \n\t" // B11 - 28 - 0
        "vpcmpeqd     %%zmm4 , %%zmm8 , %%k0                \n\t" // C1
        "vfmadd231pd  %%zmm17, %%zmm12, %%zmm9              \n\t" // F2 - dup - 5 - 9
        "vfmadd231pd  %%zmm18, %%zmm12, %%zmm6              \n\t" // F3
        "vmovups      %%zmm7 , %%zmm31                      \n\t" // B4
        "vmovups -64(%8,%0,8), %%zmm19                      \n\t" // LA4

        "vmovups      %%zmm29, %%zmm1                       \n\t" // B22 - 29 - 1
        "vpcmpeqd     %%zmm5 , %%zmm9 , %%k1                \n\t" // C2
        "vfmadd231pd  %%zmm18, %%zmm12, %%zmm10             \n\t" // F3 - dup - 6 - 10
        "vfmadd231pd  %%zmm19, %%zmm12, %%zmm7              \n\t" // F4

        "kandw        %%k0   , %%k1   , %%k0                \n\t" // check - red - 0 - 1

        "vmovups      %%zmm30, %%zmm2                       \n\t" // B22 - 30 - 2
        "vpcmpeqd     %%zmm6 , %%zmm10, %%k2                \n\t" // C3
        "vfmadd231pd  %%zmm19, %%zmm12, %%zmm11             \n\t" // F4 - dup - 7 - 11
        "kandw        %%k0   , %%k2   , %%k0                \n\t" // check - red - 0 - 2

        "vmovups      %%zmm31, %%zmm3                       \n\t" // B22 - 31 - 3
        "vpcmpeqd     %%zmm7 , %%zmm11, %%k3                \n\t" // C4

        "kandw        %%k0   , %%k3   , %%k0                \n\t" // check - red - 0 - 3
        "kxorw        %%k0   , %%k4   , %%k0                \n\t"
        "ktestw       %%k0   , %%k4                         \n\t"
        "jnz           7f                                    \n\t"

        // reduction
        "3:                                                 \n\t"

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

        "vextractf128     $1 , %%ymm4, %%xmm12              \n\t"
        "vextractf128     $1 , %%ymm5, %%xmm13              \n\t"
        "vextractf128     $1 , %%ymm6, %%xmm14              \n\t"
        "vextractf128     $1 , %%ymm7, %%xmm15              \n\t"

        "vaddpd       %%xmm4 , %%xmm12, %%xmm4              \n\t"
        "vaddpd       %%xmm5 , %%xmm13, %%xmm5              \n\t"
        "vaddpd       %%xmm6 , %%xmm14, %%xmm6              \n\t"
        "vaddpd       %%xmm7 , %%xmm15, %%xmm7              \n\t"

        "vhaddpd      %%xmm4 , %%xmm4 , %%xmm4              \n\t"
        "vhaddpd      %%xmm5 , %%xmm5 , %%xmm5              \n\t"
        "vhaddpd      %%xmm6 , %%xmm6 , %%xmm6              \n\t"
        "vhaddpd      %%xmm7 , %%xmm7 , %%xmm7              \n\t"

        "vmovsd       %%xmm4 ,    (%4)                      \n\t"
        "vmovsd       %%xmm5 ,   8(%4)                      \n\t"
        "vmovsd       %%xmm6 ,  16(%4)                      \n\t"
        "vmovsd       %%xmm7 ,  24(%4)                      \n\t"
        "jmp          5f                                    \n\t"

        // error handler to loop body
        "4:                                                 \n\t"
        "addq          $1     , %2                          \n\t"
        "subq       $8 , %0                                 \n\t"
        // recover from prelogue
        "vmovups -64(%3,%0,8), %%zmm12                      \n\t" // LX

        "vmovups      %%zmm0 , %%zmm4                       \n\t" // restore from chpt
        "vmovups      %%zmm0 , %%zmm8                       \n\t" // restore from chpt
        "vmovups -64(%5,%0,8), %%zmm16                      \n\t" // LA1
        
        "vfmadd231pd  %%zmm16, %%zmm12, %%zmm4              \n\t" // F1
        "vmovups      %%zmm1 , %%zmm5                       \n\t" // restore from chpt
        "vmovups      %%zmm1 , %%zmm9                       \n\t" // restore from chpt
        "vmovups -64(%6,%0,8), %%zmm17                      \n\t" // LA2
        "vmovups      %%zmm12, %%zmm27                      \n\t" // BX
        
        "vfmadd231pd  %%zmm16, %%zmm12, %%zmm8              \n\t" // F1 - dup - 4 - 8
        "vfmadd231pd  %%zmm17, %%zmm12, %%zmm5              \n\t" // F2
        "vmovups      %%zmm2 , %%zmm6                       \n\t" // restore from chpt
        "vmovups      %%zmm2 , %%zmm10                      \n\t" // restore from chpt
        "vmovups -64(%7,%0,8), %%zmm18                      \n\t" // LA3

        "vpcmpeqd     %%zmm4 , %%zmm8 , %%k0                \n\t" // C1
        "vfmadd231pd  %%zmm17, %%zmm12, %%zmm9              \n\t" // F2 - dup - 5 - 9
        "vfmadd231pd  %%zmm18, %%zmm12, %%zmm6              \n\t" // F3
        "vmovups      %%zmm3 , %%zmm7                       \n\t" // restore from chpt
        "vmovups      %%zmm3 , %%zmm11                      \n\t" // restore from chpt
        "vmovups -64(%8,%0,8), %%zmm19                      \n\t" // LA4   
        "vmovups    (%3,%0,8), %%zmm12                      \n\t" // LX5

        "vpcmpeqd     %%zmm5 , %%zmm9 , %%k1                \n\t" // C2
        "vfmadd231pd  %%zmm18, %%zmm27, %%zmm10             \n\t" // F3 - dup - 6 - 10
        "vfmadd231pd  %%zmm19, %%zmm27, %%zmm7              \n\t" // F4
        "vmovups      %%zmm4 , %%zmm28                      \n\t" // B5
        "vmovups    (%5,%0,8), %%zmm16                      \n\t" // LA5  

        "kandw        %%k0   , %%k1   , %%k0                \n\t" // check - red - 0 - 1
        "vpcmpeqd     %%zmm6 , %%zmm10, %%k2                \n\t" // C3
        "vfmadd231pd  %%zmm19, %%zmm27, %%zmm11             \n\t" // F4 - dup - 7 - 11
        "vfmadd231pd  %%zmm16, %%zmm12, %%zmm4              \n\t" // F5
        "vmovups      %%zmm5 , %%zmm29                      \n\t" // B6
        "vmovups    (%6,%0,8), %%zmm17                      \n\t" // LA6  
        "kandw        %%k0   , %%k2   , %%k0                \n\t" // check - red - 0 - 2
        "vmovups      %%zmm12, %%zmm27                      \n\t" // BX

        "vpcmpeqd     %%zmm7 , %%zmm11, %%k3                \n\t" // C4
        "vfmadd231pd  %%zmm16, %%zmm12, %%zmm8              \n\t" // F5 - dup - 4 - 8
        "vfmadd231pd  %%zmm17, %%zmm12, %%zmm5              \n\t" // F6
        "vmovups      %%zmm6 , %%zmm30                      \n\t" // B7
        "kandw        %%k0   , %%k3   , %%k0                \n\t" // check - red - 0 - 3
        "vmovups    (%7,%0,8), %%zmm18                      \n\t" // LA7

        "kxorw        %%k0   , %%k4   , %%k0                \n\t"

        "addq         $8     , %0                           \n\t"

        "ktestw       %%k0   , %%k4                         \n\t"
        "jnz          6f                                    \n\t"

        "subq       $8 , %1                                 \n\t"
        "jnz        1b                                      \n\t"
        "jmp        2b                                      \n\t"

        "7:                                                 \n\t"
        "addq          $1     , %2                          \n\t"
        "vmovups -64(%3,%0,8), %%zmm12                      \n\t" // LX

        "vmovups      %%zmm0 , %%zmm4                       \n\t" // restore from chpt
        "vmovups      %%zmm0 , %%zmm8                       \n\t" // restore from chpt
        "vmovups -64(%5,%0,8), %%zmm16                      \n\t" // LA1
        
        "vfmadd231pd  %%zmm16, %%zmm12, %%zmm4              \n\t" // F1
        "vmovups      %%zmm1 , %%zmm5                       \n\t" // restore from chpt
        "vmovups      %%zmm1 , %%zmm9                       \n\t" // restore from chpt
        "vmovups -64(%6,%0,8), %%zmm17                      \n\t" // LA2
        "vmovups      %%zmm12, %%zmm27                      \n\t" // BX
        
        "vfmadd231pd  %%zmm16, %%zmm12, %%zmm8              \n\t" // F1 - dup - 4 - 8
        "vfmadd231pd  %%zmm17, %%zmm12, %%zmm5              \n\t" // F2
        "vmovups      %%zmm2 , %%zmm6                       \n\t" // restore from chpt
        "vmovups      %%zmm2 , %%zmm10                      \n\t" // restore from chpt
        "vmovups -64(%7,%0,8), %%zmm18                      \n\t" // LA3

        "vpcmpeqd     %%zmm4 , %%zmm8 , %%k0                \n\t" // C1
        "vfmadd231pd  %%zmm17, %%zmm12, %%zmm9              \n\t" // F2 - dup - 5 - 9
        "vfmadd231pd  %%zmm18, %%zmm12, %%zmm6              \n\t" // F3
        "vmovups      %%zmm3 , %%zmm7                       \n\t" // restore from chpt
        "vmovups      %%zmm3 , %%zmm11                      \n\t" // restore from chpt
        "vmovups -64(%8,%0,8), %%zmm19                      \n\t" // LA4

        "vpcmpeqd     %%zmm5 , %%zmm9 , %%k1                \n\t" // C2
        "vfmadd231pd  %%zmm18, %%zmm12, %%zmm10             \n\t" // F3 - dup - 6 - 10
        "vfmadd231pd  %%zmm19, %%zmm12, %%zmm7              \n\t" // F4

        "kandw        %%k0   , %%k1   , %%k0                \n\t" // check - red - 0 - 1

        "vpcmpeqd     %%zmm6 , %%zmm10, %%k2                \n\t" // C3
        "vfmadd231pd  %%zmm19, %%zmm12, %%zmm11             \n\t" // F4 - dup - 7 - 11
        "kandw        %%k0   , %%k2   , %%k0                \n\t" // check - red - 0 - 2

        "vpcmpeqd     %%zmm7 , %%zmm11, %%k3                \n\t" // C4

        "kandw        %%k0   , %%k3   , %%k0                \n\t" // check - red - 0 - 3
        "ktestw       %%k0   , %%k4                         \n\t"
        "jz           6f                                    \n\t"
        "jmp          3b                                    \n\t"

        // still incorrect
        "6:                                                 \n\t"


        "5:                                                 \n\t"
        "vzeroupper                                         \n\t"

        :
            "+r" (i),   // 0    
            "+r" (n),   // 1
            "+r" (err_num)    // 2
        :
            "r" (x),    // 3
            "r" (y),    // 4
            "r" (a1),   // 5
            "r" (a2),   // 6
            "r" (a3),   // 7
            "r" (a4)    // 8
        : "cc", 
        "%xmm4", "%xmm5", "%xmm6", "%xmm7",
        "%xmm12", "%xmm13", "%xmm14", "%xmm15",
        "memory"
    );
    return err_num;
}

void ftblas_dgemv_t_ft(double *A, double *x, double *y, int m, int n, int lda)
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
            long err_num = dgemv_t_ft_kernel(n8, a1, a2, a3, a4, x, y_buffer);
            if (err_num) printf("detected err: %ld\n", err_num);
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