#include <stdio.h>
#include <stdlib.h>
#include "../include/ftblas.h"

static void ori_daxpy_kernel(long n, double *x, double *y, double *alpha)
{

        long register i = 0;

        __asm__ __volatile__(
                "vbroadcastsd           (%4), %%zmm0                \n\t" // alpha

                "vmovups                  (%2,%0,8), %%zmm20         \n\t" // X1
                "vmovups                  (%3,%0,8), %%zmm16         \n\t" // Y1
                "vmovups                64(%2,%0,8), %%zmm21         \n\t" // X2
                "vfmadd231pd       %%zmm20, %%zmm0 , %%zmm16         \n\t" // F1
                "vmovups                64(%3,%0,8), %%zmm17         \n\t" // Y2
                "vmovups               128(%2,%0,8), %%zmm22         \n\t" // X3

                "addq           $24, %0                              \n\t"
                "subq           $24, %1                              \n\t"
                "jz             2f                           \n\t"


                ".p2align 4                                         \n\t"
                "1:                                         \n\t"

                "vmovups        %%zmm16,   -192(%3,%0,8)             \n\t" // S1
                "vfmadd231pd       %%zmm21, %%zmm0 , %%zmm17         \n\t" // F2
                "vmovups               -64(%3,%0,8), %%zmm18         \n\t" // Y3
                "vmovups                  (%2,%0,8), %%zmm23         \n\t" // X4

                "vmovups        %%zmm17, -128(%3,%0,8)               \n\t" // S2
                "vfmadd231pd       %%zmm22, %%zmm0 , %%zmm18         \n\t" // F3
                "vmovups                  (%3,%0,8), %%zmm19         \n\t" // Y4
                "vmovups                64(%2,%0,8), %%zmm20         \n\t" // X5

                "vmovups        %%zmm18,  -64(%3,%0,8)               \n\t" // S3
                "vfmadd231pd       %%zmm23, %%zmm0 , %%zmm19         \n\t" // F4
                "vmovups                64(%3,%0,8), %%zmm16         \n\t" // Y5
                "vmovups               128(%2,%0,8), %%zmm21         \n\t" // X6

                "vmovups        %%zmm19,     (%3,%0,8)               \n\t" // S4
                "vfmadd231pd       %%zmm20, %%zmm0 , %%zmm16         \n\t" // F5
                "vmovups               128(%3,%0,8), %%zmm17         \n\t" // Y6
                "vmovups               192(%2,%0,8), %%zmm22         \n\t" // X7

                "addq           $32, %0                              \n\t"
                "subq           $32, %1                              \n\t"
                "jnz            1b                           \n\t"

                "2:                                          \n\t"
                "vmovups        %%zmm16,   -192(%3,%0,8)             \n\t" // S1
                "vfmadd231pd       %%zmm21, %%zmm0 , %%zmm17         \n\t" // F2
                "vmovups               -64(%3,%0,8), %%zmm18         \n\t" // Y3

                "vmovups        %%zmm17, -128(%3,%0,8)               \n\t" // S2
                "vfmadd231pd       %%zmm22, %%zmm0 , %%zmm18         \n\t" // F3

                "vmovups        %%zmm18,  -64(%3,%0,8)               \n\t" // S3

                
                "vzeroupper                                  \n\t"


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

void oridaxpy_compute(long int n, double da, double *x, long int inc_x, double *y, long int inc_y)
{
        long int i=0;
        long int ix=0,iy=0; 
        if ( n <= 0 )  return;
        if ( (inc_x == 1) && (inc_y == 1) )
        {
                long int n1 = n & -32;
                n1 = (n1 > 0) ? n1 - 8 : n1;
                if ( n1 )
                        ori_daxpy_kernel(n1, x, y , &da );

                i = n1;
                while(i < n)
                {
                        y[i] += da * x[i] ;
                        i++ ;
                }
        }
        long int n1 = n & -4;
        while(i < n1)
        {
                double m1      = da * x[ix] ;
                double m2      = da * x[ix+inc_x] ;
                double m3      = da * x[ix+2*inc_x] ;
                double m4      = da * x[ix+3*inc_x] ;
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
        return;
}

void ftblas_daxpy_ori(const long int n, const double alpha, const double *x, const long int inc_x, const double *y, const long int inc_y)
{
  int TOTAL_THREADS=atoi(getenv("OMP_NUM_THREADS"));
  if (TOTAL_THREADS<=1)
  {
    oridaxpy_compute(n,alpha,(double *)x,inc_x,(double *)y,inc_y);
    return;
  }
  int tid;
  int max_cpu_num=(int)sysconf(_SC_NPROCESSORS_ONLN);
  if (TOTAL_THREADS>max_cpu_num) TOTAL_THREADS=max_cpu_num;
#pragma omp parallel for schedule(static)
  for (tid = 0; tid < TOTAL_THREADS; tid++)
  {
    long int NUM_DIV_NUM_THREADS = n / TOTAL_THREADS * TOTAL_THREADS;
    long int DIM_LEN = n / TOTAL_THREADS;
    long int EDGE_LEN = (NUM_DIV_NUM_THREADS == n) ? n / TOTAL_THREADS : n - NUM_DIV_NUM_THREADS + DIM_LEN;
    if (tid == 0)
      oridaxpy_compute(EDGE_LEN,alpha,(double *)x,inc_x,(double *)y,inc_y);
    else
      oridaxpy_compute(DIM_LEN,alpha,(double *)(x + EDGE_LEN + (tid - 1) * DIM_LEN), inc_x,(double *)(y + EDGE_LEN + (tid - 1) * DIM_LEN),inc_y);
  }
  return;
}