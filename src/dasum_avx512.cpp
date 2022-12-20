#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/ftblas.h"

static void ori_dasum_kernel(long n, double *x, double *res)
{

	long register i = 0;

	__asm__ __volatile__(

		"vxorpd		%%zmm4 , %%zmm4 , %%zmm4	             \n\t"
		"vxorpd		%%zmm5 , %%zmm5 , %%zmm5	             \n\t"
		"vxorpd		%%zmm6 , %%zmm6 , %%zmm6	             \n\t"
		"vxorpd		%%zmm7 , %%zmm7 , %%zmm7	             \n\t"

		"vpcmpeqd 	%%xmm0, %%xmm0, %%xmm0	             \n\t"
		"vbroadcastsd	%%xmm0 , %%zmm0					 \n\t"
		"vpsrlq		  $1, %%zmm0, %%zmm0           \n\t"

		".p2align 4				             \n\t"
		"1:				             \n\t"
		"prefetcht0        1024(%2, %0, 8)                 \n\t" // PX
		"vandpd        (%2, %0, 8)	, %%zmm0 , %%zmm16 				\n\t"
		"vandpd      64(%2, %0, 8)	, %%zmm0 , %%zmm17 				\n\t"
		"prefetcht0        1152(%2, %0, 8)                 \n\t" // PX
		"vandpd     128(%2, %0, 8)	, %%zmm0 , %%zmm18 				\n\t"
		"vandpd     192(%2, %0, 8)	, %%zmm0 , %%zmm19 				\n\t"


		"vaddpd      %%zmm16, %%zmm4 , %%zmm4 				\n\t"
		"vaddpd      %%zmm17, %%zmm5 , %%zmm5 				\n\t"
		"vaddpd      %%zmm18, %%zmm6 , %%zmm6 				\n\t"
		"vaddpd      %%zmm19, %%zmm7 , %%zmm7 				\n\t"

		"addq		$32 , %0	  	     \n\t"
		"subq	    $32 , %1		     \n\t"
		"jnz		1b		             \n\t"


		"2: 							 \n\t"
		"vaddpd       %%zmm4 , %%zmm5 , %%zmm4              \n\t"
		"vaddpd       %%zmm6 , %%zmm7 , %%zmm6              \n\t"

		"vaddpd       %%zmm4 , %%zmm6 , %%zmm4              \n\t"

		"vextractf64x4     $0, %%zmm4 , %%ymm1              \n\t"
		"vextractf64x4     $1, %%zmm4 , %%ymm2              \n\t"
		"vaddpd       %%ymm1 , %%ymm2 , %%ymm1              \n\t"

		"vextractf128	    $1 , %%ymm1 , %%xmm2	            \n\t"
		"vaddpd       %%xmm1 , %%xmm2 , %%xmm1	            \n\t"
		"vhaddpd      %%xmm1 , %%xmm1 , %%xmm1	            \n\t"

		"vmovsd		%%xmm1 ,    (%3)		\n\t"
		"vzeroupper				\n\t"
		"ret				\n\t"

		: "+r"(i), // 0
		  "+r"(n)  // 1
		: "r"(x),  //2
		  "r"(res) // 3
		: "cc",
		  "%xmm0", "%xmm1", "%xmm2", "%xmm3",
		  "%xmm4", "%xmm5", "%xmm6", "%xmm7",
		  "memory");
}

double ftblas_dasum_ori(const long int n, const double *x, const long int inc_x) {
  long int i = 0;
  long int j = 0;
  double sumf = 0.0;
  double res = 0.0;
  long int n1;

  if (n <= 0 || inc_x <= 0)
    return sumf;

  if (inc_x == 1) {
    n1 = n & -64;

    if (n1 > 0) {
      ori_dasum_kernel(n1, (double *)x, &res);
      i = n1;
    }
    //printf("res = %f\n", res);
    while (i < n) {
      res += fabs(x[i]);
      i++;
    }

    sumf = res;

  } else {
    long int inc = -4 * inc_x;
    long int n1 = n & inc;
    //printf("inc = %d, n1 = %d, n = %d\n", inc, n1, n);
    register double sum1, sum2;
    sum1 = 0.0;
    sum2 = 0.0;
    while (j < n1) {
      //printf("j = %d, i = %d, x[%d] = %f, x[%d] = %f, x[%d] = %f, x[%d] = %f\n", j, i, i, x[i], i + inc_x, x[i + inc_x], i + 2 * inc_x, x[i + 2 * inc_x], i + 3 * inc_x, x[i + 3 * inc_x]);
      sum1 += fabs(x[i]);
      sum2 += fabs(x[i + inc_x]);
      sum1 += fabs(x[i + 2 * inc_x]);
      sum2 += fabs(x[i + 3 * inc_x]);

      i += inc_x * 4;
      j -= inc;

    }
    sumf = sum1 + sum2;
    while (i < n) {
      //printf("j = %d, n = %d, i = %d, inc_x = %d, sumf = %f, x[%d] = %f\n", j, n, i, inc_x, sumf, i, x[i]);
      sumf += fabs(x[i]);
      i += inc_x;
      j++;
    }

  }
  
  return sumf;
}