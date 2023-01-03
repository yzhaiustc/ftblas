#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/ftblas.h"

static void idamax_kernel(long n, double *x, double *maxf, long *init_idx, long *incr_idx, long *maxi)
{

	long register i = 0;

	__asm__ __volatile__(
			"vxorpd		%%zmm4 , %%zmm4 , %%zmm4	             \n\t"
			"vxorpd		%%zmm5 , %%zmm5 , %%zmm5	             \n\t"
			"vxorpd		%%zmm6 , %%zmm6 , %%zmm6	             \n\t"
			"vxorpd		%%zmm7 , %%zmm7 , %%zmm7	             \n\t"

			"vxorpd		%%zmm24, %%zmm24, %%zmm24	             \n\t"
			"vxorpd		%%zmm25, %%zmm25, %%zmm25	             \n\t"
			"vxorpd		%%zmm26, %%zmm26, %%zmm26	             \n\t"
			"vxorpd		%%zmm27, %%zmm27, %%zmm27	             \n\t"

			"vpcmpeqd 	%%xmm0, %%xmm0, %%xmm0	             \n\t"
			"vbroadcastsd	%%xmm0 , %%zmm0					 \n\t"
			"vpsrlq		  $1, %%zmm0, %%zmm0           \n\t"

			"kxnorw   %%k4   , %%k4   , %%k4                 \n\t"

			".p2align 4				             \n\t"
			"1:				             \n\t"

			"vmovupd                  (%3,%0,8), %%zmm16       \n\t" // 4 * x
			"vmovupd                64(%3,%0,8), %%zmm17       \n\t" // 4 * x
			"vmovupd               128(%3,%0,8), %%zmm18       \n\t" // 4 * x
			"vmovupd               192(%3,%0,8), %%zmm19       \n\t" // 4 * x

			"vmovupd              %%zmm4 , %%zmm28       		\n\t"
			"vmovupd              %%zmm5 , %%zmm29       		\n\t"
			"vmovupd              %%zmm6 , %%zmm30       		\n\t"
			"vmovupd              %%zmm7 , %%zmm31       		\n\t"

			"vandpd      %%zmm16, %%zmm0 , %%zmm20 				\n\t"
			"vandpd      %%zmm17, %%zmm0 , %%zmm21 				\n\t"
			"vandpd      %%zmm18, %%zmm0 , %%zmm22 				\n\t"
			"vandpd      %%zmm19, %%zmm0 , %%zmm23 				\n\t"

			"vandpd      %%zmm16, %%zmm0 , %%zmm16 				\n\t"
			"vandpd      %%zmm17, %%zmm0 , %%zmm17 				\n\t"
			"vandpd      %%zmm18, %%zmm0 , %%zmm18 				\n\t"
			"vandpd      %%zmm19, %%zmm0 , %%zmm19 				\n\t"

			"vmaxpd      %%zmm24, %%zmm20, %%zmm24 				\n\t"
			"vmaxpd      %%zmm25, %%zmm21, %%zmm25 				\n\t"
			"vmaxpd      %%zmm26, %%zmm22, %%zmm26 				\n\t"
			"vmaxpd      %%zmm27, %%zmm23, %%zmm27 				\n\t"
			
			"vmaxpd      %%zmm4 , %%zmm16, %%zmm4 				\n\t"
			"vmaxpd      %%zmm5 , %%zmm17, %%zmm5 				\n\t"
			"vmaxpd      %%zmm6 , %%zmm18, %%zmm6 				\n\t"
			"vmaxpd      %%zmm7 , %%zmm19, %%zmm7 				\n\t"

			"vpcmpeqd     %%zmm24, %%zmm4 , %%k0                \n\t"
			"vpcmpeqd     %%zmm25, %%zmm5 , %%k1                \n\t"
			"vpcmpeqd     %%zmm26, %%zmm6 , %%k2                \n\t"
			"vpcmpeqd     %%zmm27, %%zmm7 , %%k3                \n\t"

			"kandq        %%k0   , %%k1   , %%k0                \n\t"
			"kandq        %%k2   , %%k3   , %%k2                \n\t"
			"kandq        %%k0   , %%k2   , %%k0                \n\t"
			"kxorw        %%k0   , %%k4   , %%k0                \n\t"
			"ktestw       %%k0   , %%k4                         \n\t"
			"jnz          6f           \n\t"


			"addq		$32 , %0	  	     \n\t"
			"subq	    $32 , %1		     \n\t"
			"jnz		1b		             \n\t"

			"2:									\n\t"
			"vmaxpd      %%zmm4, %%zmm5, %%zmm4 \n\t"
			"vmaxpd      %%zmm6, %%zmm7, %%zmm6 \n\t"
			"vmaxpd      %%zmm4, %%zmm6, %%zmm4 \n\t"

      		"vextractf64x4     $0, %%zmm4 , %%ymm1              \n\t"
      		"vextractf64x4     $1, %%zmm4 , %%ymm2              \n\t"

			"vmaxpd      %%ymm2 , %%ymm1 , %%ymm2 				\n\t"
			"vextractf128	 $1 , %%ymm2 , %%xmm1	     		\n\t"
			"vmaxpd      %%xmm2 , %%xmm1 , %%xmm2 		 		\n\t"		
			"movhlps     %%xmm2 , %%xmm1 				  		\n\t"
			"vmaxpd      %%xmm2 , %%xmm1 , %%xmm2 		  		\n\t"

			"vmovsd		%%xmm2,    (%4)	\n\t"
			"vzeroupper				\n\t"

			"3:						\n\t"
			"vbroadcastsd              %%xmm2 , %%zmm24          		\n\t"
			"vmovdqu64               (%5,%1,8), %%zmm0          		\n\t"
			"vmovdqu64               (%6,%1,8), %%zmm1          		\n\t"
			
			"vpcmpeqd 		%%xmm2 , %%xmm2, %%xmm2	             		\n\t"
			"vbroadcastsd	%%xmm2 , %%zmm2					 	 		\n\t"
			"vpsrlq		  		$1 , %%zmm2, %%zmm2           	 		\n\t"

			".p2align 4				             						\n\t"
			"4:				             								\n\t"
			"vmovupd				 (%3,%1,8), %%zmm16					\n\t"
			"vandpd                    %%zmm16, %%zmm2 , %%zmm12        \n\t"
			"vandpd                    %%zmm16, %%zmm2 , %%zmm17        \n\t" // dup
			"vcmpeqpd                  %%zmm24, %%zmm17, %%k1           \n\t"
			"vcmpeqpd                  %%zmm24, %%zmm12, %%k0           \n\t"
			"kxorw					   %%k0   , %%k1   , %%k2			\n\t"
			"ktestw              	   %%k2   , %%k4	          	 	\n\t"
			"jnz 8f														\n\t"


			"ktestw              	   %%k0   , %%k4	          	 	\n\t"
			"jnz 5f														\n\t"

			"vaddpd                    %%zmm0 , %%zmm1 , %%zmm0         \n\t"
			"addq		$8 , %1	  	     								\n\t"
			"subq	    $8 , %0		     								\n\t"
			"jnz		4b		             							\n\t"


			"5: \n\t"

      		"vextractf64x4     $0, %%zmm12, %%ymm14               \n\t"
      		"vextractf64x4     $1, %%zmm12, %%ymm15               \n\t"

      		"vextractf64x4     $0, %%zmm0 , %%ymm1                \n\t"
      		"vextractf64x4     $1, %%zmm0 , %%ymm2                \n\t"

			"vcmpgtpd		     %%ymm14, %%ymm15, %%ymm4	      \n\t"
			"vblendvpd           %%ymm4 , %%ymm2 , %%ymm1, %%ymm0 \n\t"
			"vmaxpd      		 %%ymm14, %%ymm15, %%ymm12 		  \n\t"

			"vmovupd 			%%ymm12 , %%ymm2				  \n\t"
			"vextractf128	$1 , %%ymm2 , %%xmm12	     		  \n\t"	   // extract max elements
			"vextractf128	$1 , %%ymm0 , %%xmm11	     		  \n\t"    // extract max indices

			"vcmpgtpd		     %%xmm2 , %%xmm12, %%xmm4	      \n\t"    // calculate mask
			"vblendvpd           %%xmm4 , %%xmm11, %%xmm0, %%xmm0 \n\t"    // update max index with mask
			"vmaxpd      %%xmm2, %%xmm12, %%xmm2 		 \n\t"

			"vcmpeqpd		     %%xmm2 , %%xmm12, %%xmm6	      \n\t"    // calculate mask
			"vpcmpgtq		     %%xmm11 , %%xmm0, %%xmm5	      \n\t"    // calculate mask
			"vandpd             %%xmm5,  %%xmm6, %%xmm5         \n\t"    // take abs
			"vblendvpd           %%xmm5 , %%xmm11, %%xmm0, %%xmm0 \n\t"    // update max index with mask		

			"movhlps     %%xmm2 , %%xmm5 				  \n\t"
			"movhlps     %%xmm0 , %%xmm6 				  \n\t"

			"vcmpgtsd	 %%xmm2 , %%xmm5 , %%xmm4	      \n\t"    // calculate mask
			"vblendvpd   %%xmm4 , %%xmm6 , %%xmm0, %%xmm0 \n\t"    // update max index with mask
			"vmaxpd      %%xmm2 , %%xmm5 , %%xmm2 		  \n\t"

			"vcmpeqpd		     %%xmm2 , %%xmm5 , %%xmm7	      \n\t"    // calculate mask
			"vpcmpgtq		     %%xmm6 , %%xmm0 , %%xmm8	      \n\t"    // calculate mask
			"vandpd              %%xmm7 , %%xmm8 , %%xmm7         \n\t"    // take abs
			"vblendvpd           %%xmm7 , %%xmm6, %%xmm0, %%xmm0 \n\t"    // update max index with mask

			"vmovsd		 %%xmm2 ,    (%4)	\n\t"
			"vmovsd		 %%xmm0 ,    (%2)	\n\t"
			"vzeroupper				\n\t"
			"ret \n\t"


			"6: 					\n\t"
			"vmovupd                  (%3,%0,8), %%zmm16       \n\t" // 4 * x
			"vmovupd                64(%3,%0,8), %%zmm17       \n\t" // 4 * x
			"vmovupd               128(%3,%0,8), %%zmm18       \n\t" // 4 * x
			"vmovupd               192(%3,%0,8), %%zmm19       \n\t" // 4 * x

			"vmovupd              %%zmm28, %%zmm4        		\n\t"
			"vmovupd              %%zmm29, %%zmm5        		\n\t"
			"vmovupd              %%zmm30, %%zmm6        		\n\t"
			"vmovupd              %%zmm31, %%zmm7        		\n\t"

			"vmovupd              %%zmm28, %%zmm24        		\n\t"
			"vmovupd              %%zmm29, %%zmm25        		\n\t"
			"vmovupd              %%zmm30, %%zmm26        		\n\t"
			"vmovupd              %%zmm31, %%zmm27        		\n\t"

			"vandpd      %%zmm16, %%zmm0 , %%zmm20 				\n\t"
			"vandpd      %%zmm17, %%zmm0 , %%zmm21 				\n\t"
			"vandpd      %%zmm18, %%zmm0 , %%zmm22 				\n\t"
			"vandpd      %%zmm19, %%zmm0 , %%zmm23 				\n\t"

			"vandpd      %%zmm16, %%zmm0 , %%zmm16 				\n\t"
			"vandpd      %%zmm17, %%zmm0 , %%zmm17 				\n\t"
			"vandpd      %%zmm18, %%zmm0 , %%zmm18 				\n\t"
			"vandpd      %%zmm19, %%zmm0 , %%zmm19 				\n\t"

			"vmaxpd      %%zmm24, %%zmm20, %%zmm24 				\n\t"
			"vmaxpd      %%zmm25, %%zmm21, %%zmm25 				\n\t"
			"vmaxpd      %%zmm26, %%zmm22, %%zmm26 				\n\t"
			"vmaxpd      %%zmm27, %%zmm23, %%zmm27 				\n\t"
			
			"vmaxpd      %%zmm4 , %%zmm16, %%zmm4 				\n\t"
			"vmaxpd      %%zmm5 , %%zmm17, %%zmm5 				\n\t"
			"vmaxpd      %%zmm6 , %%zmm18, %%zmm6 				\n\t"
			"vmaxpd      %%zmm7 , %%zmm19, %%zmm7 				\n\t"

			"vpcmpeqd     %%zmm24, %%zmm4 , %%k0                \n\t"
			"vpcmpeqd     %%zmm25, %%zmm5 , %%k1                \n\t"
			"vpcmpeqd     %%zmm26, %%zmm6 , %%k2                \n\t"
			"vpcmpeqd     %%zmm27, %%zmm7 , %%k3                \n\t"

			"kandq        %%k0   , %%k1   , %%k0                \n\t"
			"kandq        %%k2   , %%k3   , %%k2                \n\t"
			"kandq        %%k0   , %%k2   , %%k0                \n\t"
			"kxorw        %%k0   , %%k4   , %%k0                \n\t"
			"ktestw       %%k0   , %%k4                         \n\t"
			"jnz          7f           \n\t"


			"addq		$32 , %0	  	     \n\t"
			"subq	    $32 , %1		     \n\t"
			"jnz		1b		             \n\t"
			"jmp		2b					 \n\t"

			"7:								 \n\t"
			"ret							 \n\t"

			"8:								 \n\t"
			"vmovupd				 (%3,%1,8), %%zmm16					\n\t"
			"vandpd                    %%zmm16, %%zmm2 , %%zmm12        \n\t"
			"vandpd                    %%zmm16, %%zmm2 , %%zmm17        \n\t" // dup
			"vcmpeqpd                  %%zmm24, %%zmm17, %%k1           \n\t"
			"vcmpeqpd                  %%zmm24, %%zmm12, %%k0           \n\t"
			"kxorw					   %%k0   , %%k1   , %%k2			\n\t"
			"ktestw              	   %%k2   , %%k4	          	 	\n\t"
			"jnz 7b														\n\t"


			"ktestw              	   %%k0   , %%k4	          	 	\n\t"
			"jnz 5b														\n\t"

			"vaddpd                    %%zmm0 , %%zmm1 , %%zmm0         \n\t"
			"addq		$8 , %1	  	     								\n\t"
			"subq	    $8 , %0		     								\n\t"
			"jnz		4b		             							\n\t"
			
			: "+r"(i), // 0
				"+r"(n),	// 1
				"+r"(maxi)	// 2
			: "r"(x),	//3
				"r"(maxf), // 4
				"r"(init_idx), // 5
				"r"(incr_idx) // 6
			: "cc",
				"%xmm0", "%xmm1","%xmm2", "%xmm3",
				"%xmm4", "%xmm5","%xmm6", "%xmm7",
				"%xmm8", "%xmm9","%xmm10", "%xmm11",
				"%xmm12", "%xmm13", "%xmm14", "%xmm15",
				"memory");
}

double ft_idamax(long n, double *x, long inc_x)
{
  long i = 0;
  long j = 0;
  double maxf = 0.0;
  long max = 0;

  if (n <= 0 || inc_x <= 0)
    return (max);

  if (inc_x == 1) {

    long n1 = n & -32;
    if (n1 > 0) {
      long init_idx[8]  = {0 , 1 , 2 , 3 , 4 , 5 , 6 , 7};
      long incr_idx[8]  = {8 , 8 , 8 , 8 , 8 , 8 , 8 , 8};
      idamax_kernel(n1, x, &maxf, init_idx, incr_idx, &max);
      
      i = n1;
      //printf("max = %ld, maxf = %f\n", max, maxf);
    } else {
      maxf = fabs(x[0]);
      i++;
    }

    while (i < n) {
      if (fabs(x[i]) > maxf) {
        max = i;
        maxf = fabs(x[i]);
      }
      i++;
    }
    return (max);

  } else {

    max = 0;
    maxf = fabs(x[0]);

    long n1 = n & -4;
    while (j < n1) {

      if (fabs(x[i]) > maxf) {
        max = j;
        maxf = fabs(x[i]);
      }
      if (fabs(x[i + inc_x]) > maxf) {
        max = j + 1;
        maxf = fabs(x[i + inc_x]);
      }
      if (fabs(x[i + 2 * inc_x]) > maxf) {
        max = j + 2;
        maxf = fabs(x[i + 2 * inc_x]);
      }
      if (fabs(x[i + 3 * inc_x]) > maxf) {
        max = j + 3;
        maxf = fabs(x[i + 3 * inc_x]);
      }

      i += inc_x * 4;

      j += 4;

    }

    while (j < n) {
      if (fabs(x[i]) > maxf) {
        max = j;
        maxf = fabs(x[i]);
      }
      i += inc_x;
      j++;
    }
    return (max);
  }
}

int ftblas_idamax_ft(const long int n, const double *x, const long int inc_x){
    return (int)ft_idamax(n, (double*)x, inc_x);
}