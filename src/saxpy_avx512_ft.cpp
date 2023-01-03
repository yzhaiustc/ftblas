#include <stdio.h>
#include <stdlib.h>
#include "../include/ftblas.h"


static long ft_saxpy_kernel(long n, float *x, float *y, float *alpha)
{

        long register i = 0;
        long err_num = 0;
        __asm__ __volatile__(
                "vbroadcastss           (%5), %%zmm0                \n\t" // alpha
                "kxnorw         %%k4 ,  %%k4 , %%k4                   \n\t" // dup
                /* Prologue */
                "vmovups                  (%3,%0,4), %%zmm20        \n\t" // X1

                "vmovups                  (%4,%0,4), %%zmm16        \n\t" // Y1
                "vmovups                64(%3,%0,4), %%zmm21        \n\t" // X2

                "vmovups                    %%zmm20, %%zmm28        \n\t" // B1
                "vmovups                64(%4,%0,4), %%zmm17        \n\t" // Y2
                "vmovups               128(%3,%0,4), %%zmm22        \n\t" // X3

                "vfmadd213ps       %%zmm16, %%zmm0 , %%zmm20        \n\t" // F1
                "vfmadd213ps       %%zmm16, %%zmm0 , %%zmm28        \n\t" // F1 - dup - 20 - 28
                "vmovups                    %%zmm21, %%zmm29        \n\t" // B2
                "vmovups               128(%4,%0,4), %%zmm18        \n\t" // Y3
                "vmovups               192(%3,%0,4), %%zmm23        \n\t" // X4


                "addq           $64, %0                              \n\t"
                "subq           $64, %1                              \n\t"
                "jz             2f                           \n\t"

                /* Loop Body */
                ".p2align 4                                         \n\t"
                "1:                                         \n\t"
                "vpcmpeqd       %%zmm20, %%zmm28, %%k0                 \n\t" // C1
                "vmovups                    %%zmm16, %%zmm24         \n\t" // BY
                "vmovups        %%zmm20,   -256(%4,%0,4)             \n\t" // S1
                "vfmadd213ps       %%zmm17, %%zmm0 , %%zmm21         \n\t" // F2
                "vfmadd213ps       %%zmm17, %%zmm0 , %%zmm29         \n\t" // F2 - dup - 21 - 29
                "vmovups                    %%zmm22, %%zmm30         \n\t" // B3
                "vmovups               -64(%4,%0,4), %%zmm19         \n\t" // Y4
                "vmovups                  (%3,%0,4), %%zmm20         \n\t" // X5

                "vpcmpeqd       %%zmm21, %%zmm29, %%k1                 \n\t" // C2
                "vmovups                    %%zmm17, %%zmm25         \n\t" // BY
                "vmovups        %%zmm21,   -192(%4,%0,4)             \n\t" // S2
                "vfmadd213ps       %%zmm18, %%zmm0 , %%zmm22         \n\t" // F3
                "vfmadd213ps       %%zmm18, %%zmm0 , %%zmm30         \n\t" // F3 - dup - 22 - 30
                "vmovups                    %%zmm23, %%zmm31         \n\t" // B4
                "vmovups                  (%4,%0,4), %%zmm16         \n\t" // Y5
                "vmovups                64(%3,%0,4), %%zmm21         \n\t" // X6

                "kandw          %%k0   , %%k1   , %%k0                 \n\t" // check - red - 0 - 1

                "vpcmpeqd       %%zmm22, %%zmm30, %%k2                 \n\t" // C3
                "vmovups                    %%zmm18, %%zmm26         \n\t" // BY
                "vmovups        %%zmm22,   -128(%4,%0,4)             \n\t" // S3
                "vfmadd213ps       %%zmm19, %%zmm0 , %%zmm23         \n\t" // F4
                "vfmadd213ps       %%zmm19, %%zmm0 , %%zmm31         \n\t" // F4 - dup - 23 - 31
                "vmovups                    %%zmm20, %%zmm28         \n\t" // B5
                "vmovups                64(%4,%0,4), %%zmm17         \n\t" // Y6
                "vmovups               128(%3,%0,4), %%zmm22         \n\t" // X7

                "vpcmpeqd       %%zmm23, %%zmm31, %%k3                 \n\t" // C4
                "vmovups                    %%zmm19, %%zmm27         \n\t" // BY
                "vmovups        %%zmm23,    -64(%4,%0,4)             \n\t" // S4
                "vfmadd213ps       %%zmm16, %%zmm0 , %%zmm20         \n\t" // F5
                "vfmadd213ps       %%zmm16, %%zmm0 , %%zmm28         \n\t" // F5 - dup - 20 - 28

                "kandw          %%k2   , %%k3   , %%k2                 \n\t" // check - red - 2 - 3

                "vmovups                    %%zmm21, %%zmm29         \n\t" // B6
                "vmovups               128(%4,%0,4), %%zmm18         \n\t" // Y7
                "vmovups               192(%3,%0,4), %%zmm23         \n\t" // X8

                "kandw          %%k0   , %%k2   , %%k0                 \n\t"
                "addq               $64 , %0                                   \n\t"
                "kxorw          %%k0   , %%k4   , %%k0                 \n\t"
                "ktestw         %%k0   , %%k4                          \n\t"
                "jnz            3f                                               \n\t"

                "subq           $64, %1                              \n\t"
                "jnz            1b                           \n\t"

                /* Epilogue */
                "2:                                          \n\t"
                "vpcmpeqd       %%zmm20, %%zmm28, %%k0                 \n\t" // C1
                "vmovups                    %%zmm16, %%zmm24         \n\t" // BY
                "vmovups        %%zmm20,   -256(%4,%0,4)             \n\t" // S1
                "vfmadd213ps       %%zmm17, %%zmm0 , %%zmm21         \n\t" // F2
                "vfmadd213ps       %%zmm17, %%zmm0 , %%zmm29         \n\t" // F2 - dup
                "vmovups                    %%zmm22, %%zmm30         \n\t" // B3
                "vmovups               -64(%4,%0,4), %%zmm19         \n\t" // Y4

                "vpcmpeqd       %%zmm21, %%zmm29, %%k1                 \n\t" // C2
                "vmovups                    %%zmm17, %%zmm25         \n\t" // BY
                "vmovups        %%zmm21,   -192(%4,%0,4)             \n\t" // S2
                "vfmadd213ps       %%zmm18, %%zmm0 , %%zmm22         \n\t" // F3
                "vfmadd213ps       %%zmm18, %%zmm0 , %%zmm30         \n\t" // F3 - dup 
                "vmovups                    %%zmm23, %%zmm31         \n\t" // B4

                "vpcmpeqd       %%zmm22, %%zmm30, %%k2                 \n\t" // C3
                "vmovups                    %%zmm18, %%zmm26         \n\t" // BY
                "vmovups        %%zmm22,   -128(%4,%0,4)             \n\t" // S3
                "vfmadd213ps       %%zmm19, %%zmm0 , %%zmm23         \n\t" // F4
                "vfmadd213ps       %%zmm19, %%zmm0 , %%zmm31         \n\t" // F4 - dup

                "kandw          %%k0   , %%k1   , %%k0                 \n\t" // check - red - 0 - 1

                "vpcmpeqd       %%zmm23, %%zmm31, %%k3                 \n\t" // C4
                "vmovups                    %%zmm19, %%zmm27         \n\t" // BY
                "kandw          %%k2   , %%k3   , %%k2                 \n\t" // check - red - 2 - 3
                "vmovups        %%zmm23,    -64(%4,%0,4)             \n\t" // S4

                "kandw          %%k0   , %%k2   , %%k0                 \n\t"
                "kxorw          %%k0   , %%k4   , %%k0                 \n\t"
                "ktestw         %%k0   , %%k4                          \n\t"
                "jnz            5f                                               \n\t"

                "jmp      4f                                 \n\t"

                /* Error Handler to Loop Body */
                "3:                                          \n\t"
                "addq           $1, %2                              \n\t"
                "subq           $64, %0                             \n\t"
                "vmovups              -256(%3,%0,4), %%zmm20        \n\t" // X1

                "vmovups        %%zmm24, %%zmm16                    \n\t" // restore from in-register chhkp - 16 - 24
                "vmovups              -192(%3,%0,4), %%zmm21        \n\t" // X2

                "vmovups                    %%zmm20, %%zmm28        \n\t" // B1
                "vmovups        %%zmm25, %%zmm17                    \n\t" // restore from in-register chhkp - 17 - 25
                "vmovups              -128(%3,%0,4), %%zmm22        \n\t" // X3

                "vfmadd213ps       %%zmm16, %%zmm0 , %%zmm20        \n\t" // F1
                "vfmadd213ps       %%zmm16, %%zmm0 , %%zmm28        \n\t" // F1 - dup - 20 - 28
                "vmovups                    %%zmm21, %%zmm29        \n\t" // B2
                "vmovups        %%zmm26, %%zmm18                    \n\t" // restore from in-register chhkp - 18 - 26
                "vmovups               -64(%3,%0,4), %%zmm23        \n\t" // X4

                "vpcmpeqd       %%zmm20, %%zmm28, %%k0                 \n\t" // C1
                "vmovups                    %%zmm16, %%zmm24         \n\t" // BY
                "vmovups        %%zmm20,   -256(%4,%0,4)             \n\t" // S1
                "vfmadd213ps       %%zmm17, %%zmm0 , %%zmm21         \n\t" // F2
                "vfmadd213ps       %%zmm17, %%zmm0 , %%zmm29         \n\t" // F2 - dup - 21 - 29
                "vmovups                    %%zmm22, %%zmm30         \n\t" // B3
                "vmovups        %%zmm27, %%zmm19                    \n\t" // restore from in-register chhkp - 19 - 27
                "vmovups                  (%3,%0,4), %%zmm20         \n\t" // X5

                "vpcmpeqd       %%zmm21, %%zmm29, %%k1                 \n\t" // C2
                "vmovups                    %%zmm17, %%zmm25         \n\t" // BY
                "vmovups        %%zmm21,   -192(%4,%0,4)             \n\t" // S2
                "vfmadd213ps       %%zmm18, %%zmm0 , %%zmm22         \n\t" // F3
                "vfmadd213ps       %%zmm18, %%zmm0 , %%zmm30         \n\t" // F3 - dup - 22 - 30
                "vmovups                    %%zmm23, %%zmm31         \n\t" // B4
                "vmovups                  (%4,%0,4), %%zmm16         \n\t" // Y5
                "vmovups                64(%3,%0,4), %%zmm21         \n\t" // X6

                "kandw          %%k0   , %%k1   , %%k0                 \n\t" // check - red - 0 - 1

                "vpcmpeqd       %%zmm22, %%zmm30, %%k2                 \n\t" // C3
                "vmovups                    %%zmm18, %%zmm26         \n\t" // BY
                "vmovups        %%zmm22,   -128(%4,%0,4)             \n\t" // S3
                "vfmadd213ps       %%zmm19, %%zmm0 , %%zmm23         \n\t" // F4
                "vfmadd213ps       %%zmm19, %%zmm0 , %%zmm31         \n\t" // F4 - dup - 23 - 31
                "vmovups                    %%zmm20, %%zmm28         \n\t" // B5
                "vmovups                64(%4,%0,4), %%zmm17         \n\t" // Y6
                "vmovups               128(%3,%0,4), %%zmm22         \n\t" // X7

                "vpcmpeqd       %%zmm23, %%zmm31, %%k3                 \n\t" // C4
                "vmovups                    %%zmm19, %%zmm27         \n\t" // BY
                "vmovups        %%zmm23,    -64(%4,%0,4)             \n\t" // S4
                "vfmadd213ps       %%zmm16, %%zmm0 , %%zmm20         \n\t" // F5
                "vfmadd213ps       %%zmm17, %%zmm0 , %%zmm28         \n\t" // F5 - dup - 20 - 28

                "kandw          %%k2   , %%k3   , %%k2                 \n\t" // check - red - 2 - 3

                "vmovups                    %%zmm21, %%zmm29         \n\t" // B6
                "vmovups               128(%4,%0,4), %%zmm18         \n\t" // Y7
                "vmovups               192(%3,%0,4), %%zmm23         \n\t" // X8

                "kandw          %%k0   , %%k2   , %%k0                 \n\t"
                "addq               $64 , %0                                   \n\t"
                "kxorw          %%k0   , %%k4   , %%k0                 \n\t"
                "ktestw         %%k0   , %%k4                          \n\t"
                "jnz            6f                                               \n\t"

                "subq           $64, %1                              \n\t"
                "jnz            1b                           \n\t"
                "jmp            2b                           \n\t"

                /* Error Handler to Epilogue */
                "5:                                          \n\t"
                "addq           $1, %2                              \n\t"
                "vmovups              -256(%3,%0,4), %%zmm20        \n\t" // X1

                "vmovups        %%zmm24, %%zmm16                    \n\t" // restore from in-register chhkp - 16 - 24
                "vmovups              -192(%3,%0,4), %%zmm21        \n\t" // X2

                "vmovups                    %%zmm20, %%zmm28        \n\t" // B1
                "vmovups        %%zmm25, %%zmm17                    \n\t" // restore from in-register chhkp - 17 - 25
                "vmovups              -128(%3,%0,4), %%zmm22        \n\t" // X3

                "vfmadd213ps       %%zmm16, %%zmm0 , %%zmm20        \n\t" // F1
                "vfmadd213ps       %%zmm16, %%zmm0 , %%zmm28        \n\t" // F1 - dup - 20 - 28
                "vmovups                    %%zmm21, %%zmm29        \n\t" // B2
                "vmovups        %%zmm26, %%zmm18                    \n\t" // restore from in-register chhkp - 18 - 26
                "vmovups               -64(%3,%0,4), %%zmm23        \n\t" // X4

                "vpcmpeqd       %%zmm20, %%zmm28, %%k0                 \n\t" // C1
                "vmovups        %%zmm20,   -256(%4,%0,4)             \n\t" // S1
                "vfmadd213ps       %%zmm17, %%zmm0 , %%zmm21         \n\t" // F2
                "vfmadd213ps       %%zmm17, %%zmm0 , %%zmm29         \n\t" // F2 - dup
                "vmovups                    %%zmm22, %%zmm30         \n\t" // B3
                "vmovups        %%zmm27, %%zmm19                    \n\t" // restore from in-register chhkp - 18 - 26

                "vpcmpeqd       %%zmm21, %%zmm29, %%k1                 \n\t" // C2
                "vmovups        %%zmm21,   -192(%4,%0,4)             \n\t" // S2
                "vfmadd213ps       %%zmm18, %%zmm0 , %%zmm22         \n\t" // F3
                "vfmadd213ps       %%zmm18, %%zmm0 , %%zmm30         \n\t" // F3 - dup 
                "vmovups                    %%zmm23, %%zmm31         \n\t" // B4

                "vpcmpeqd       %%zmm22, %%zmm30, %%k2                 \n\t" // C3
                "vmovups        %%zmm22,   -128(%4,%0,4)             \n\t" // S3
                "vfmadd213ps       %%zmm19, %%zmm0 , %%zmm23         \n\t" // F4
                "vfmadd213ps       %%zmm19, %%zmm0 , %%zmm31         \n\t" // F4 - dup

                "kandw          %%k0   , %%k1   , %%k0                 \n\t" // check - red - 0 - 1

                "vpcmpeqd       %%zmm23, %%zmm31, %%k3                 \n\t" // C4
                "kandw          %%k2   , %%k3   , %%k2                 \n\t" // check - red - 2 - 3
                "vmovups        %%zmm23,    -64(%4,%0,4)             \n\t" // S4

                "kandw          %%k0   , %%k2   , %%k0                 \n\t"
                "kxorw          %%k0   , %%k4   , %%k0                 \n\t"
                "ktestw         %%k0   , %%k4                          \n\t"
                "jnz            7f                                               \n\t"
                "jmp            4f                                               \n\t"

                /* If Loop Body is still incorrect */
                "6:                                          \n\t"
                "addq           $1, %2                              \n\t"
                "jmp            4f                                       \n\t"

                /* If Epilogue is still incorrect */
                "7:                                          \n\t"
                "addq           $1, %2                              \n\t"
                "jmp            4f                                       \n\t"


                "4:                                          \n\t"
                "vzeroupper                                  \n\t"


                : "+r"(i),      // 0
                  "+r"(n),      // 1
                  "+r"(err_num) // 2
                : "r"(x),       // 3
                  "r"(y),       // 4
                  "r"(alpha)    // 5
                : "cc",
                  "%xmm0", "%xmm1", "%xmm2", "%xmm3",
                  "%xmm4", "%xmm5", "%xmm6", "%xmm7",
                  "%xmm8", "%xmm9", "%xmm10", "%xmm11",
                  "%xmm12", "%xmm13", "%xmm14", "%xmm15",
                  "memory");
        return err_num;
}

int ft_saxpy(long n, float da, float *x, long inc_x, float *y, long inc_y)
{
        long i=0;
        long ix=0,iy=0;

        if ( n <= 0 )  return(0);

        if ( (inc_x == 1) && (inc_y == 1) )
        {

                long n1 = n & -64;
                if ( n1 )
                        ft_saxpy_kernel(n1, x, y , &da );

                i = n1;
                while(i < n)
                {

                        y[i] += da * x[i] ;
                        i++ ;

                }
                return(0);


        }

        long n1 = n & -4;

        while(i < n1)
        {

                float m1      = da * x[ix] ;
                float m2      = da * x[ix+inc_x] ;
                float m3      = da * x[ix+2*inc_x] ;
                float m4      = da * x[ix+3*inc_x] ;

                y[iy]         += m1 ;
                y[iy+inc_y]   += m2 ;
                y[iy+2*inc_y] += m3 ;
                y[iy+3*inc_y] += m4 ;

                ix  += inc_x*4 ;
                iy  += inc_y*4 ;
                i+=4 ;

        }

        while(i < n)
        {

                y[iy] += da * x[ix] ;
                ix  += inc_x ;
                iy  += inc_y ;
                i++ ;

        }
        return(0);

}

void ftblas_saxpy_ft(const long int n, const float alpha, const float *x, const long int inc_x, const float *y, const long int inc_y)
{
    ft_saxpy(n, alpha, (float*)x, inc_x, (float*)y, inc_y);
}
