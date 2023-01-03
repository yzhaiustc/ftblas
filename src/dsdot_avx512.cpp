#include <stdio.h>
#include <stdlib.h>
#include "../include/ftblas.h"

static void ori_dsdot_kernel(long n, double *x, double *y, double *dot)
{

    long register i = 0;

    __asm__ __volatile__(
        "vxorps		%%zmm4, %%zmm4, %%zmm4	             \n\t"
        "vxorps		%%zmm5, %%zmm5, %%zmm5	             \n\t"
        "vxorps		%%zmm6, %%zmm6, %%zmm6	             \n\t"
        "vxorps		%%zmm7, %%zmm7, %%zmm7	             \n\t"

        ".p2align 4				             \n\t"
        "1:				             \n\t"
        "vmovups                  (%2,%0,4) , %%zmm16         \n\t" // move x[4]
        "vmovups                64(%2,%0,4) , %%zmm17         \n\t" // move x[4]
        "vmovups               128(%2,%0,4) , %%zmm18         \n\t" // move x[4]
        "vmovups               192(%2,%0,4) , %%zmm19         \n\t" // move x[4]

        "vmovups                  (%3,%0,4) , %%zmm20         \n\t" // move y[4]
        "vmovups                64(%3,%0,4) , %%zmm21         \n\t" // move y[4]
        "vmovups               128(%3,%0,4) , %%zmm22         \n\t" // move y[4]
        "vmovups               192(%3,%0,4) , %%zmm23         \n\t" // move y[4]

        "vfmadd231ps        %%zmm16, %%zmm20, %%zmm4          \n\t"
        "vfmadd231ps        %%zmm17, %%zmm21, %%zmm5          \n\t"
        "vfmadd231ps        %%zmm18, %%zmm22, %%zmm6          \n\t"
        "vfmadd231ps        %%zmm19, %%zmm23, %%zmm7          \n\t"

        "addq		$64 , %0	  	     \n\t"
        "subq	    $64 , %1		     \n\t"
        "jnz		1b		             \n\t"

        "vaddps       %%zmm4 , %%zmm5 , %%zmm4              \n\t"
        "vaddps       %%zmm6 , %%zmm7 , %%zmm6              \n\t"
        "vaddps       %%zmm4 , %%zmm6 , %%zmm4              \n\t"

        "vextractf64x4     $0, %%zmm4 , %%ymm1              \n\t"
        "vextractf64x4     $1, %%zmm4 , %%ymm2              \n\t"
        "vaddps       %%ymm1 , %%ymm2 , %%ymm1              \n\t"

        "vextractf128	    $1 , %%ymm1 , %%xmm2	        \n\t"
        "vaddps       %%xmm1 , %%xmm2 , %%xmm1	            \n\t"
        "vhaddps      %%xmm1 , %%xmm1 , %%xmm1	            \n\t"
        "vhaddps      %%xmm1 , %%xmm1 , %%xmm1	            \n\t"

        "vmovss		%%xmm1,    (%4)		\n\t"
        "vzeroupper				\n\t"
        "ret				\n\t"


        : "+r"(i), // 0
          "+r"(n)  // 1
        : "r"(x),  // 2
          "r"(y),  // 3
          "r"(dot) // 4
        : "cc",
          "%xmm0", "%xmm1",
          "%xmm2", "%xmm3",
          "%xmm4", "%xmm5",
          "%xmm6", "%xmm7",
//          "%xmm8" , "%xmm9" , "%xmm10", "%xmm11",
//          "%xmm12", "%xmm13", "%xmm14", "%xmm15",
          "memory");
}

double ori_dsdot(long n, double *x, long inc_x, double *y, long inc_y)
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

		if ( n1 )
		{
			double *x1=x;
			double *y1=y;
			long n2 = 256;
			while (i<n1) 
			{
				ori_dsdot_kernel(n2, x1, y1 , &asmdot );
				mydot += (double)asmdot;
				asmdot=0.;
				x1 += n2;
				y1 += n2;
				i  += n2;
			}
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

double ftblas_dsdot_ori(const long int n, const double *x, const long int inc_x, const double *y, const long int inc_y)
{
    ori_dsdot(n, (double*)x, inc_x, (double*)y, inc_y);
}