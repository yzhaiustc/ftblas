#include <stdio.h>
#include <stdlib.h>
#include "../include/ftblas.h"

static void dswap_kernel(long n, double *x, double *y)
{

	long i = 0;

	__asm__ __volatile__(
			".p2align 4				            			 \n\t"
			"1:				            	    			 \n\t"
			"vmovupd	 0(%2, %0, 8) ,	%%zmm16     		 \n\t"
			"vmovupd	64(%2, %0, 8) ,	%%zmm17	     		 \n\t"
			"vmovupd   128(%2, %0, 8) ,	%%zmm18	     		 \n\t"
			"vmovupd   192(%2, %0, 8) ,	%%zmm19	     		 \n\t"

			"vmovupd	 0(%3, %0, 8) ,	%%zmm20	     		 \n\t"
			"vmovupd	64(%3, %0, 8) ,	%%zmm21	     		 \n\t"
			"vmovupd   128(%3, %0, 8) ,	%%zmm22	     		 \n\t"
			"vmovupd   192(%3, %0, 8) ,	%%zmm23	     		 \n\t"

			"vmovupd   		   %%zmm20,	  0(%2, %0, 8)	     \n\t"
			"vmovupd   		   %%zmm21,  64(%2, %0, 8)	     \n\t"
			"vmovupd   		   %%zmm22, 128(%2, %0, 8)	     \n\t"
			"vmovupd   		   %%zmm23, 192(%2, %0, 8)	     \n\t"

			"vmovupd   		   %%zmm16,	  0(%3, %0, 8)	     \n\t"
			"vmovupd   		   %%zmm17,  64(%3, %0, 8)	     \n\t"
			"vmovupd   		   %%zmm18, 128(%3, %0, 8)	     \n\t"
			"vmovupd   		   %%zmm19, 192(%3, %0, 8)	     \n\t"

			"addq		$32, %0		  	 	    	 \n\t"
			"subq	    $32, %1			             \n\t"
			"jnz		1b		             	     \n\t"

			"vzeroupper					    \n\t"

			: "+r"(i),		// 0
			  "+r"(n)		// 1
			: "r"(x), 		// 2
			  "r"(y)		// 3
			: "cc",
//				"%xmm0", "%xmm1", "%xmm2", "%xmm3",
//				"%xmm4", "%xmm5", "%xmm6", "%xmm7",
//				"%xmm8", "%xmm9", "%xmm10", "%xmm11",
//				"%xmm12", "%xmm13", "%xmm14", "%xmm15",
				"memory");
}

int ft_dswap(long n, double *x, long inc_x, double *y, long inc_y) {
  long i = 0;
  long ix = 0, iy = 0;
  double temp;

  if (n <= 0)
    return (0);

  if ((inc_x == 1) && (inc_y == 1)) {

      long n1 = n & -32;
      if (n1 > 0) {
        dswap_kernel(n1, x, y);
        i = n1;
      }

      while (i < n) {
        temp = y[i];
        y[i] = x[i];
        x[i] = temp;
        i++;

      }
  } else {

    while (i < n) {
      temp = y[iy];
      y[iy] = x[ix];
      x[ix] = temp;
      ix += inc_x;
      iy += inc_y;
      i++;

    }

  }
  return (0);

}

void ftblas_dswap(const int n, const double *x, const int inc_x, const double *y, const int inc_y){
    ft_dswap(n, (double *)x, inc_x, (double *)y, inc_y);
}