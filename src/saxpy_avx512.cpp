#include <stdio.h>
#include <stdlib.h>
#include "../include/ftblas.h"

static void ori_saxpy_kernel(long n, float *x, float *y, float *alpha)
{

        long register i = 0;

        __asm__ __volatile__(
                "vbroadcastss           (%4), %%zmm0                \n\t" // alpha

                ".p2align 4                                         \n\t"
                "1:                                         \n\t"
                "vmovups                 0(%2,%0,4), %%zmm20         \n\t" // 4 * x
                "vmovups                64(%2,%0,4), %%zmm21         \n\t" // 4 * x
                "vmovups               128(%2,%0,4), %%zmm22         \n\t" // 4 * x
                "vmovups               192(%2,%0,4), %%zmm23         \n\t" // 4 * x

                "vmovups                  (%3,%0,4), %%zmm16         \n\t" // 4 * y
                "vmovups                64(%3,%0,4), %%zmm17         \n\t" // 4 * y
                "vmovups               128(%3,%0,4), %%zmm18         \n\t" // 4 * y
                "vmovups               192(%3,%0,4), %%zmm19         \n\t" // 4 * y

                "vfmadd231ps       %%zmm20, %%zmm0 , %%zmm16         \n\t" // y += alpha * x
                "vfmadd231ps       %%zmm21, %%zmm0 , %%zmm17         \n\t" // y += alpha * x
                "vfmadd231ps       %%zmm22, %%zmm0 , %%zmm18         \n\t" // y += alpha * x
                "vfmadd231ps       %%zmm23, %%zmm0 , %%zmm19         \n\t" // y += alpha * x

                "vmovups        %%zmm16,    (%3,%0,4)                \n\t"
                "vmovups        %%zmm17,  64(%3,%0,4)                \n\t"
                "vmovups        %%zmm18, 128(%3,%0,4)                \n\t"
                "vmovups        %%zmm19, 192(%3,%0,4)                \n\t"

                "addq           $64, %0                              \n\t"
                "subq           $64, %1                              \n\t"
                "jnz            1b                           \n\t"
                "vzeroupper                                  \n\t"
                "ret \n\t"


                : "+r"(i),   // 0
                  "+r"(n)       // 1
                : "r"(x),       // 2
                  "r"(y),       // 3
                  "r"(alpha) // 4
                : "cc",
                  "%xmm0", "%xmm1", "%xmm2", "%xmm3",
                  "%xmm4", "%xmm5", "%xmm6", "%xmm7",
                  "%xmm8", "%xmm9", "%xmm10", "%xmm11",
                  "%xmm12", "%xmm13", "%xmm14", "%xmm15",
                  "memory");
}

int ori_saxpy(long n, float da, float *x, long inc_x, float *y, long inc_y)
{
        long i=0;
        long ix=0,iy=0;

        if ( n <= 0 )  return(0);

        if ( (inc_x == 1) && (inc_y == 1) )
        {

                long n1 = n & -64;
//                n1 = (n1 > 0) ? n1 - 16 : n1;
                if ( n1 )
                        ori_saxpy_kernel(n1, x, y , &da );

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

void ftblas_saxpy_ori(const long int n, const float alpha, const float *x, const long int inc_x, const float *y, const long int inc_y)
{
    ori_saxpy(n, alpha, (float*)x, inc_x, (float*)y, inc_y);
}
