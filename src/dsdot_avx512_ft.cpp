#include <stdio.h>
#include <stdlib.h>
#include "../include/ftblas.h"

static long dsdot_kernel(long n, double *x, double *y, double *dot)
{

  long register i = 0;
  long register err_num = 0;
  __asm__ __volatile__(

      "vxorps           %%zmm0 , %%zmm0 , %%zmm0                   \n\t"
      "vxorps           %%zmm1 , %%zmm1 , %%zmm1                   \n\t"
      "vxorps           %%zmm2 , %%zmm2 , %%zmm2                   \n\t"
      "vxorps           %%zmm3 , %%zmm3 , %%zmm3                   \n\t"

      "vxorps           %%zmm4 , %%zmm4 , %%zmm4                   \n\t"
      "vxorps           %%zmm5 , %%zmm5 , %%zmm5                   \n\t"
      "vxorps           %%zmm6 , %%zmm6 , %%zmm6                   \n\t"
      "vxorps           %%zmm7 , %%zmm7 , %%zmm7                   \n\t"

      "kxnorw   %%k4   , %%k4   , %%k4                 \n\t"

      /* Prologue */
      "vmovups       (%3, %0, 4), %%zmm16               \n\t" // X1
      "vmovups       (%4, %0, 4), %%zmm24               \n\t" // Y1
      "vmovups     64(%3, %0, 4), %%zmm17               \n\t" // X2
      "vmovups           %%zmm4 , %%zmm28               \n\t" // B1
      "vmovups     64(%4, %0, 4), %%zmm25               \n\t" // Y2
      "vmovups    128(%3, %0, 4), %%zmm18               \n\t" // X3
      "vfmadd231ps  %%zmm16, %%zmm24, %%zmm0            \n\t" // F1
      "vfmadd231ps  %%zmm16, %%zmm24, %%zmm4            \n\t" // F1 - dup - 0 - 4
      "vmovups           %%zmm5 , %%zmm29               \n\t" // B2
      "vmovups    128(%4, %0, 4), %%zmm26               \n\t" // Y3
      "vmovups    192(%3, %0, 4), %%zmm19               \n\t" // X4

      "addq       $64 , %0               \n\t"
      "subq   $64 , %1                   \n\t"
      "jz         3f                       \n\t"

      ".p2align 4                                \n\t"
      "1:                                  \n\t"
      "vpcmpeqd     %%zmm0 , %%zmm4 , %%k0                \n\t" // C1
      "vmovups           %%zmm28, %%zmm12                 \n\t" // chkpt - 28 - 12
      "vfmadd231ps  %%zmm17, %%zmm25, %%zmm1              \n\t" // F2
      "vfmadd231ps  %%zmm17, %%zmm25, %%zmm5              \n\t" // F2 - dup - 1 - 5
      "vmovups           %%zmm6 , %%zmm30                 \n\t" // B3
      "vmovups    -64(%4, %0, 4), %%zmm27                 \n\t" // Y4
      "vmovups       (%3, %0, 4), %%zmm16                 \n\t" // X5

      "vpcmpeqd     %%zmm1 , %%zmm5 , %%k1                \n\t" // C2
      "vmovups           %%zmm29, %%zmm13                 \n\t" // chkpt - 29 - 13
      "vfmadd231ps  %%zmm18, %%zmm26, %%zmm2              \n\t" // F3
      "vfmadd231ps  %%zmm18, %%zmm26, %%zmm6              \n\t" // F3 - dup - 2 - 6

      "kandw          %%k0   , %%k1   , %%k0              \n\t" // check - red - 0 - 1

      "vmovups           %%zmm7 , %%zmm31                 \n\t" // B4
      "vmovups       (%4, %0, 4), %%zmm24                 \n\t" // Y5
      "vmovups     64(%3, %0, 4), %%zmm17                 \n\t" // X6

      "vpcmpeqd     %%zmm2 , %%zmm6 , %%k2                \n\t" // C3
      "vmovups           %%zmm30, %%zmm14                 \n\t" // chkpt - 30 - 14
      "vfmadd231ps  %%zmm19, %%zmm27, %%zmm3              \n\t" // F4
      "vfmadd231ps  %%zmm19, %%zmm27, %%zmm7              \n\t" // F4 - dup - 3 - 7
      "vmovups           %%zmm4 , %%zmm28                 \n\t" // B5
      "vmovups     64(%4, %0, 4), %%zmm25                 \n\t" // Y6
      "vmovups    128(%3, %0, 4), %%zmm18                 \n\t" // X7

      "vpcmpeqd     %%zmm3 , %%zmm7 , %%k3                \n\t" // C4
      "vmovups           %%zmm31, %%zmm15                 \n\t" // chkpt - 31 - 15
      "vfmadd231ps  %%zmm16, %%zmm24, %%zmm0              \n\t" // F5
      "vfmadd231ps  %%zmm16, %%zmm24, %%zmm4              \n\t" // F5 - dup - 0 - 4
      
      "kandw          %%k2   , %%k3   , %%k2              \n\t" // check - red - 2 - 3

      "vmovups           %%zmm5 , %%zmm29                 \n\t" // B6

      "kandw          %%k0   , %%k2   , %%k0              \n\t"

      "vmovups    128(%4, %0, 4), %%zmm26                 \n\t" // Y7
      "vmovups    192(%3, %0, 4), %%zmm19                 \n\t" // X8
      
      "kxorw          %%k0   , %%k4   , %%k0              \n\t"

      "addq       $64 , %0               \n\t"

      "ktestw         %%k0   , %%k4                       \n\t"
      "jnz        2f                       \n\t"
      "subq     $64 , %1                 \n\t"
      "jnz        1b                       \n\t"

      "3:   \n\t"
      /* Epilogue */
      "vpcmpeqd     %%zmm0 , %%zmm4 , %%k0                \n\t" // C1
      "vmovups           %%zmm28, %%zmm12                 \n\t" // chkpt - 28 - 12
      "vfmadd231ps  %%zmm17, %%zmm25, %%zmm1              \n\t" // F2
      "vfmadd231ps  %%zmm17, %%zmm25, %%zmm5              \n\t" // F2 - dup - 1 - 5
      "vmovups           %%zmm6 , %%zmm30                 \n\t" // B3
      "vmovups    -64(%4, %0, 4), %%zmm27                 \n\t" // Y4

      "vpcmpeqd     %%zmm1 , %%zmm5 , %%k1                \n\t" // C2
      "vmovups           %%zmm29, %%zmm13                 \n\t" // chkpt - 29 - 13
      "vfmadd231ps  %%zmm18, %%zmm26, %%zmm2              \n\t" // F3
      "vfmadd231ps  %%zmm18, %%zmm26, %%zmm6              \n\t" // F3 - dup - 2 - 6

      "kandw          %%k0   , %%k1   , %%k0              \n\t" // check - red - 0 - 1

      "vmovups           %%zmm7 , %%zmm31                 \n\t" // B4

      "vpcmpeqd     %%zmm2 , %%zmm6 , %%k2                \n\t" // C3
      "vmovups           %%zmm30, %%zmm14                 \n\t" // chkpt - 30 - 14
      "vfmadd231ps  %%zmm19, %%zmm27, %%zmm3              \n\t" // F4
      "vfmadd231ps  %%zmm19, %%zmm27, %%zmm7              \n\t" // F4 - dup - 3 - 7
      

      "vpcmpeqd     %%zmm3 , %%zmm7 , %%k3                \n\t" // C4
      "vmovups           %%zmm31, %%zmm15                 \n\t" // chkpt - 31 - 15

      "kandw          %%k2   , %%k3   , %%k2              \n\t" // check - red - 2 - 3
      "kandw          %%k0   , %%k2   , %%k0              \n\t"
      "kxorw          %%k0   , %%k4   , %%k0              \n\t"
      "ktestw         %%k0   , %%k4                       \n\t"
      "jnz            6f                                  \n\t"

      "5:                                                 \n\t"
      "vaddps       %%zmm0 , %%zmm1 , %%zmm0              \n\t"
      "vaddps       %%zmm2 , %%zmm3 , %%zmm2              \n\t"

      "vaddps       %%zmm0 , %%zmm2 , %%zmm0              \n\t"

      "vextractf64x4     $0, %%zmm0 , %%ymm1              \n\t"
      "vextractf64x4     $1, %%zmm0 , %%ymm2              \n\t"
      "vaddps       %%ymm1 , %%ymm2 , %%ymm1              \n\t"

      "vextractf128         $1 , %%ymm1 , %%xmm2                  \n\t"
      "vaddps       %%xmm1 , %%xmm2 , %%xmm1                \n\t"
      "vhaddps      %%xmm1 , %%xmm1 , %%xmm1                \n\t"
      "vhaddps      %%xmm1 , %%xmm1 , %%xmm1                \n\t"

      "vmovss           %%xmm1 ,    (%5)        \n\t"
      "jmp 4f     \n\t"

      /* Error Handler to Loop Body */
      "2:                                               \n\t"
      "addq       $1 , %2                \n\t"
      "subq     $64 , %0                 \n\t"
      "vmovups   -256(%3, %0, 4), %%zmm16               \n\t" // X1
      "vmovups   -256(%4, %0, 4), %%zmm24               \n\t" // Y1
      "vmovups   -192(%3, %0, 4), %%zmm17               \n\t" // X2
      "vmovups           %%zmm12, %%zmm0                \n\t" // restore from chkpt - 12 - 0
      "vmovups           %%zmm12, %%zmm4                \n\t" // restore from chkpt - 12 - 4
      "vmovups   -192(%4, %0, 4), %%zmm25               \n\t" // Y2
      "vmovups   -128(%3, %0, 4), %%zmm18               \n\t" // X3
      "vfmadd231ps  %%zmm16, %%zmm24, %%zmm0            \n\t" // F1
      "vfmadd231ps  %%zmm16, %%zmm24, %%zmm4            \n\t" // F1 - dup - 0 - 4
      "vmovups           %%zmm13, %%zmm1                \n\t" // restore from chkpt - 13 - 1
      "vmovups           %%zmm13, %%zmm5                \n\t" // restore from chkpt - 13 - 5
      "vmovups   -128(%4, %0, 4), %%zmm26               \n\t" // Y3
      "vmovups    -64(%3, %0, 4), %%zmm19               \n\t" // X4

      "vpcmpeqd     %%zmm0 , %%zmm4 , %%k0                \n\t" // C1
      "vfmadd231ps  %%zmm17, %%zmm25, %%zmm1              \n\t" // F2
      "vfmadd231ps  %%zmm17, %%zmm25, %%zmm5              \n\t" // F2 - dup - 1 - 5
      "vmovups           %%zmm14, %%zmm2                  \n\t" // restore from chkpt - 14 - 2
      "vmovups           %%zmm14, %%zmm6                  \n\t" // restore from chkpt - 14 - 6
      "vmovups    -64(%4, %0, 4), %%zmm27                 \n\t" // Y4
      "vmovups       (%3, %0, 4), %%zmm16                 \n\t" // X5

      "vpcmpeqd     %%zmm1 , %%zmm5 , %%k1                \n\t" // C2
      "vfmadd231ps  %%zmm18, %%zmm26, %%zmm2              \n\t" // F3
      "vfmadd231ps  %%zmm18, %%zmm26, %%zmm6              \n\t" // F3 - dup - 2 - 6

      "kandw          %%k0   , %%k1   , %%k0              \n\t" // check - red - 0 - 1

      "vmovups           %%zmm15, %%zmm3                  \n\t" // restore from chkpt - 15 - 3
      "vmovups           %%zmm15, %%zmm7                  \n\t" // restore from chkpt - 15 - 7
      "vmovups       (%4, %0, 4), %%zmm24                 \n\t" // Y5
      "vmovups     64(%3, %0, 4), %%zmm17                 \n\t" // X6

      "vpcmpeqd     %%zmm2 , %%zmm6 , %%k2                \n\t" // C3
      "vfmadd231ps  %%zmm19, %%zmm27, %%zmm3              \n\t" // F4
      "vfmadd231ps  %%zmm19, %%zmm27, %%zmm7              \n\t" // F4 - dup - 3 - 7
      "vmovups           %%zmm4 , %%zmm28                 \n\t" // B5
      "vmovups     64(%4, %0, 4), %%zmm25                 \n\t" // Y6
      "vmovups    128(%3, %0, 4), %%zmm18                 \n\t" // X7

      "vpcmpeqd     %%zmm3 , %%zmm7 , %%k3                \n\t" // C4
      "vfmadd231ps  %%zmm16, %%zmm24, %%zmm0              \n\t" // F5
      "vfmadd231ps  %%zmm16, %%zmm24, %%zmm4              \n\t" // F5 - dup - 0 - 4

      "kandw          %%k2   , %%k3   , %%k2              \n\t" // check - red - 2 - 3

      "vmovups           %%zmm5 , %%zmm29                 \n\t" // B6
      "vmovups    128(%4, %0, 4), %%zmm26                 \n\t" // Y7
      "vmovups    192(%3, %0, 4), %%zmm19                 \n\t" // X8

      
      "kandw          %%k0   , %%k2   , %%k0              \n\t"
      "kxorw          %%k0   , %%k4   , %%k0              \n\t"
      "ktestw         %%k0   , %%k4                       \n\t"
      "jnz            4f                                  \n\t"

      "addq       $64 , %0               \n\t"
      "subq     $64 , %1                 \n\t"
      "jnz        1b                       \n\t"

      "jmp 3b                                             \n\t"

      /* Error Handler to Epilogue */
      "6:                                                 \n\t"
      /* Restore From Prologue */
      "addq       $1 , %2                \n\t"
      "vmovups   -256(%3, %0, 4), %%zmm16               \n\t" // X1
      "vmovups   -256(%4, %0, 4), %%zmm24               \n\t" // Y1
      "vmovups   -192(%3, %0, 4), %%zmm17               \n\t" // X2
      "vmovups           %%zmm12, %%zmm0                \n\t" // restore from chkpt - 12 - 0
      "vmovups           %%zmm12, %%zmm4                \n\t" // restore from chkpt - 12 - 4
      "vmovups   -192(%4, %0, 4), %%zmm25               \n\t" // Y2
      "vmovups   -128(%3, %0, 4), %%zmm18               \n\t" // X3
      "vfmadd231ps  %%zmm16, %%zmm24, %%zmm0            \n\t" // F1
      "vfmadd231ps  %%zmm16, %%zmm24, %%zmm4            \n\t" // F1 - dup - 0 - 4
      "vmovups           %%zmm13, %%zmm1                \n\t" // restore from chkpt - 13 - 1
      "vmovups           %%zmm13, %%zmm5                \n\t" // restore from chkpt - 13 - 5
      "vmovups   -128(%4, %0, 4), %%zmm26               \n\t" // Y3
      "vmovups    -64(%3, %0, 4), %%zmm19               \n\t" // X4

      "vpcmpeqd     %%zmm0 , %%zmm4 , %%k0                \n\t" // C1
      "vfmadd231ps  %%zmm17, %%zmm25, %%zmm1              \n\t" // F2
      "vfmadd231ps  %%zmm17, %%zmm25, %%zmm5              \n\t" // F2 - dup - 1 - 5
      "vmovups           %%zmm14, %%zmm2                  \n\t" // restore from chkpt - 14 - 2
      "vmovups           %%zmm14, %%zmm6                  \n\t" // restore from chkpt - 14 - 6
      "vmovups    -64(%4, %0, 4), %%zmm27                 \n\t" // Y4

      "vpcmpeqd     %%zmm1 , %%zmm5 , %%k1                \n\t" // C2
      "vfmadd231ps  %%zmm18, %%zmm26, %%zmm2              \n\t" // F3
      "vfmadd231ps  %%zmm18, %%zmm26, %%zmm6              \n\t" // F3 - dup - 2 - 6

      "kandw          %%k0   , %%k1   , %%k0              \n\t" // check - red - 0 - 1

      "vmovups           %%zmm15, %%zmm3                  \n\t" // restore from chkpt - 15 - 3
      "vmovups           %%zmm15, %%zmm7                  \n\t" // restore from chkpt - 15 - 7

      "vpcmpeqd     %%zmm2 , %%zmm6 , %%k2                \n\t" // C3
      "vfmadd231ps  %%zmm19, %%zmm27, %%zmm3              \n\t" // F4
      "vfmadd231ps  %%zmm19, %%zmm27, %%zmm7              \n\t" // F4 - dup - 3 - 7
      
      "vpcmpeqd     %%zmm3 , %%zmm7 , %%k3                \n\t" // C4

      "kandw          %%k2   , %%k3   , %%k2              \n\t" // check - red - 2 - 3
      "kandw          %%k0   , %%k2   , %%k0              \n\t"
      "kxorw          %%k0   , %%k4   , %%k0              \n\t"
      "ktestw         %%k0   , %%k4                       \n\t"
      "jnz            4f                                  \n\t"

      "jmp 5b                                             \n\t"

      "7: \n\t" // if still incorrect
      "addq       $1 , %2                \n\t"

      "4: \n\t" // exit
      "vzeroupper                   \n\t"
      : "+r"(i), // 0
        "+r"(n), // 1
        "+r"(err_num)  // 2
      : "r"(x),  // 3
        "r"(y),  // 4
        "r"(dot) // 5
      : "cc",
        "%xmm0", "%xmm1", "%xmm2", "%xmm3",
        "%xmm4", "%xmm5", "%xmm6", "%xmm7",
        "%xmm8", "%xmm9", "%xmm10", "%xmm11",
        "%xmm12", "%xmm13", "%xmm14", "%xmm15",
        "memory");

  return err_num;

}

double ft_dsdot(long n, double *x, long inc_x, double *y, long inc_y)
{
	long i=0;
	long ix=0,iy=0;
	double dot = 0.0 ;

	double mydot = 0.0;
    double asmdot = 0.0;

	long n1;

	if ( n <= 0 )  return(dot);

	if ( (inc_x == 1) && (inc_y == 1) )
	{

	    n1 = n & (long)(-64);
		n1 = n1 < 256 ? 0 : n1;
		if ( n1)
		{
			double *x1=x;
			double *y1=y;
			long n2 = 256;
			long err_num = 0;
			while (i<n1) 
			{
				err_num += dsdot_kernel(n2, x1, y1 , &asmdot );
				mydot += (double)asmdot;
				asmdot=0.;
				x1 += n2;
				y1 += n2;
				i  += n2;
			}
			if (err_num) printf("detected and corrected %ld error\n", err_num);
		}

		i = n1;
		while(i < n)
		{
			dot += (double)y[i] * (double)x[i] ;
			i++ ;
		}

		dot+=mydot;
		return(dot);


	}

	n1 = n & (long)(-2);

	while(i < n1)
	{
		dot += (double)y[iy] * (double)x[ix] + (double)y[iy+inc_y] * (double)x[ix+inc_x];

		ix  += inc_x*2 ;
		iy  += inc_y*2 ;
		i+=2 ;

	}

	while(i < n)
	{
		dot += (double)y[iy] * (double)x[ix] ;

		ix  += inc_x ;
		iy  += inc_y ;
		i++ ;

	}
	return(dot);

}

double ftblas_dsdot_ft(const long int n, const double *x, const long int inc_x, const double *y, const long int inc_y)
{
    ft_dsdot(n, (double*)x, inc_x, (double*)y, inc_y);
}