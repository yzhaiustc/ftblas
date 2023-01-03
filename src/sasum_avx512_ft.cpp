#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/ftblas.h"

static long sasum_kernel(long n, float *x, float *res)
{

    long register i = 0;
    long register err_num = 0;
    __asm__ __volatile__(

        "vxorps     %%zmm4 , %%zmm4 , %%zmm4                 \n\t"
        "vxorps     %%zmm5 , %%zmm5 , %%zmm5                 \n\t"
        "vxorps     %%zmm6 , %%zmm6 , %%zmm6                 \n\t"
        "vxorps     %%zmm7 , %%zmm7 , %%zmm7                 \n\t"

        "vxorps     %%zmm24, %%zmm24, %%zmm24                \n\t"
        "vxorps     %%zmm25, %%zmm25, %%zmm25                \n\t"
        "vxorps     %%zmm26, %%zmm26, %%zmm26                \n\t"
        "vxorps     %%zmm27, %%zmm27, %%zmm27                \n\t"

        /* For Abs */
        "vpcmpeqw   %%xmm0, %%xmm0, %%xmm0                   \n\t"
        "vbroadcastss   %%xmm0 , %%zmm0                      \n\t"
        "vpsrld       $1, %%zmm0, %%zmm0                     \n\t"

        "kxnord   %%k4   , %%k4   , %%k4                     \n\t"

        /* Prologue */
        "vmovups       (%3, %0, 4)  , %%zmm16               \n\t" // L1
        "vmovups     %%zmm4 , %%zmm8                        \n\t" // B1
        "vmovups     64(%3, %0, 4)  , %%zmm17               \n\t" // L2
        "vandps      %%zmm16, %%zmm0 , %%zmm20              \n\t" // N1
        "vandps      %%zmm16, %%zmm0 , %%zmm16              \n\t" // N1 - dup - 16 - 20
        "vmovups     %%zmm5 , %%zmm9                        \n\t" // B2
        "vmovups    128(%3, %0, 4)  , %%zmm18               \n\t" // L3
        "vaddps      %%zmm16, %%zmm4 , %%zmm24              \n\t" // D1
        "vaddps      %%zmm16, %%zmm4 , %%zmm4               \n\t" // D1 - dup - 4 - 24
        "vandps      %%zmm17, %%zmm0 , %%zmm21              \n\t" // N2
        "vandps      %%zmm17, %%zmm0 , %%zmm17              \n\t" // N2 - dup - 17 - 21
        "vmovups     %%zmm6 , %%zmm10                       \n\t" // B3
        "vmovups    192(%3, %0, 4)  , %%zmm19               \n\t" // L4

        "addq       $64 , %0             \n\t"
        "subq       $64 , %1             \n\t"
        "jz         2f                   \n\t"

        ".p2align 4                          \n\t"
        "1:                          \n\t"
        /* Loop Body */
        "vpcmpeqd       %%zmm24, %%zmm4 , %%k0              \n\t" // C1
        "vmovups     %%zmm8 , %%zmm28                       \n\t" // chpt1 - 8 - 28
        "vaddps      %%zmm21, %%zmm5 , %%zmm25              \n\t" // D2
        "vaddps      %%zmm21, %%zmm5 , %%zmm5               \n\t" // D2 - dup - 25 - 5
        "vandps      %%zmm18, %%zmm0 , %%zmm22              \n\t" // N3
        "vandps      %%zmm18, %%zmm0 , %%zmm18              \n\t" // N3 - dup - 18 - 22
        "vmovups     %%zmm7 , %%zmm11                       \n\t" // B4
        "vmovups       (%3, %0, 4)  , %%zmm16               \n\t" // L5

        "vpcmpeqd       %%zmm25, %%zmm5 , %%k1              \n\t" // C2
        "vmovups     %%zmm9 , %%zmm29                       \n\t" // chpt2 - 9 - 29
        "vaddps      %%zmm22, %%zmm6 , %%zmm26              \n\t" // D3
        "vaddps      %%zmm22, %%zmm6 , %%zmm6               \n\t" // D3 - dup - 26 - 6
        "vandps      %%zmm19, %%zmm0 , %%zmm23              \n\t" // N4
        "vandps      %%zmm19, %%zmm0 , %%zmm19              \n\t" // N4 - dup - 23 - 19
        "vmovups     %%zmm4 , %%zmm8                        \n\t" // B5
        "vmovups     64(%3, %0, 4)  , %%zmm17               \n\t" // L6

        "kandw          %%k0   , %%k1   , %%k0              \n\t" // check - red - 0 - 1

        "vpcmpeqd       %%zmm26, %%zmm6 , %%k2              \n\t" // C3
        "vmovups     %%zmm10, %%zmm30                       \n\t" // chpt3 - 10 - 30
        "vaddps      %%zmm23, %%zmm7 , %%zmm27              \n\t" // D4
        "vaddps      %%zmm23, %%zmm7 , %%zmm7               \n\t" // D4 - dup - 27 - 7
        "vandps      %%zmm16, %%zmm0 , %%zmm20              \n\t" // N5
        "vandps      %%zmm16, %%zmm0 , %%zmm16              \n\t" // N5 - dup - 20 - 16
        "vmovups     %%zmm5 , %%zmm9                        \n\t" // B6
        "vmovups    128(%3, %0, 4)  , %%zmm18               \n\t" // L7

        "vpcmpeqd       %%zmm27, %%zmm7 , %%k3              \n\t" // C4
        "vmovups     %%zmm11, %%zmm31                       \n\t" // chpt4 - 11 - 31
        "vaddps      %%zmm16, %%zmm4 , %%zmm24              \n\t" // D5
        "vaddps      %%zmm16, %%zmm4 , %%zmm4               \n\t" // D5 - dup - 24 - 4

        "kandw          %%k2   , %%k3   , %%k2              \n\t" // check - red - 2 - 3

        "vandps      %%zmm17, %%zmm0 , %%zmm21              \n\t" // N6
        "vandps      %%zmm17, %%zmm0 , %%zmm17              \n\t" // N6 - dup - 20 - 16

        "kandw          %%k0   , %%k2   , %%k0              \n\t"

        "vmovups     %%zmm6 , %%zmm10                       \n\t" // B7

        "kxorw          %%k0   , %%k4   , %%k0              \n\t"

        "vmovups    192(%3, %0, 4)  , %%zmm19               \n\t" // L8
        "addq       $64 , %0             \n\t"

        "ktestw         %%k0   , %%k4                       \n\t"
        "jnz            3f                                  \n\t" // if error, go to error handler

        "subq       $64 , %1             \n\t"
        "jnz        1b                   \n\t"


        "2:                              \n\t"
        /* Epilogue */
        "vpcmpeqd       %%zmm24, %%zmm4 , %%k0              \n\t" // C1
        "vmovups     %%zmm8 , %%zmm28                       \n\t" // chpt1 - 8 - 28
        "vaddps      %%zmm21, %%zmm5 , %%zmm25              \n\t" // D2
        "vaddps      %%zmm21, %%zmm5 , %%zmm5               \n\t" // D2 - dup - 25 - 5
        "vandps      %%zmm18, %%zmm0 , %%zmm22              \n\t" // N3
        "vandps      %%zmm18, %%zmm0 , %%zmm18              \n\t" // N3 - dup - 18 - 22
        "vmovups     %%zmm7 , %%zmm11                       \n\t" // B4

        "vpcmpeqd       %%zmm25, %%zmm5 , %%k1              \n\t" // C2
        "vmovups     %%zmm9 , %%zmm29                       \n\t" // chpt2 - 9 - 29
        "vaddps      %%zmm22, %%zmm6 , %%zmm26              \n\t" // D3
        "vaddps      %%zmm22, %%zmm6 , %%zmm6               \n\t" // D3 - dup - 26 - 6

        "kandw          %%k0   , %%k1   , %%k0              \n\t" // check - red - 0 - 1

        "vandps      %%zmm19, %%zmm0 , %%zmm23              \n\t" // N4
        "vandps      %%zmm19, %%zmm0 , %%zmm19              \n\t" // N4 - dup - 23 - 19

        "vpcmpeqd       %%zmm26, %%zmm6 , %%k2              \n\t" // C3
        "vmovups     %%zmm10, %%zmm30                       \n\t" // chpt3 - 10 - 30
        "vaddps      %%zmm23, %%zmm7 , %%zmm27              \n\t" // D4
        "vaddps      %%zmm23, %%zmm7 , %%zmm7               \n\t" // D4 - dup - 27 - 7

        "vpcmpeqd       %%zmm27, %%zmm7 , %%k3              \n\t" // C4
        "vmovups     %%zmm11, %%zmm31                       \n\t" // chpt4 - 11 - 31

        "kandw          %%k2   , %%k3   , %%k2              \n\t" // check - red - 2 - 3
        "kandw          %%k0   , %%k2   , %%k0              \n\t"
        "kxorw          %%k0   , %%k4   , %%k0              \n\t"
        "ktestw         %%k0   , %%k4                       \n\t"
        "jnz            4f                                  \n\t"

        /* Reduction */
        "vaddps       %%zmm4 , %%zmm5 , %%zmm4              \n\t"
        "vaddps       %%zmm6 , %%zmm7 , %%zmm6              \n\t"

        "vaddps       %%zmm4 , %%zmm6 , %%zmm4              \n\t"

        "vextractf64x4     $0, %%zmm4 , %%ymm1              \n\t"
        "vextractf64x4     $1, %%zmm4 , %%ymm2              \n\t"
        "vaddps       %%ymm1 , %%ymm2 , %%ymm1              \n\t"

        "vextractf128       $1 , %%ymm1 , %%xmm2                \n\t"
        "vaddps       %%xmm1 , %%xmm2 , %%xmm1              \n\t"
        "vhaddps      %%xmm1 , %%xmm1 , %%xmm1              \n\t"
        "vhaddps      %%xmm1 , %%xmm1 , %%xmm1              \n\t"

        "vmovss     %%xmm1 ,    (%4)        \n\t"
        
        "jmp 4f             \n\t"

        /* Error Handler to Loop Body */
        "3:                              \n\t"
        "subq       $64 , %0             \n\t"
		"addq       $1  , %2             \n\t" // increment error number detected
        "vmovups -256(%3, %0, 4)    , %%zmm16               \n\t" // L1
        "vmovups     %%zmm28, %%zmm4                        \n\t" // restore from 28 - 4
        "vmovups -192(%3, %0, 4)    , %%zmm17               \n\t" // L2
        "vandps      %%zmm16, %%zmm0 , %%zmm20              \n\t" // N1
        "vandps      %%zmm16, %%zmm0 , %%zmm16              \n\t" // N1 - dup - 16 - 20
        "vmovups     %%zmm29, %%zmm5                        \n\t" // restore from 29 - 5
        "vmovups -128(%3, %0, 4)    , %%zmm18               \n\t" // L3
        "vaddps      %%zmm16, %%zmm4 , %%zmm24              \n\t" // D1
        "vaddps      %%zmm16, %%zmm4 , %%zmm4               \n\t" // D1 - dup - 4 - 24
        "vandps      %%zmm17, %%zmm0 , %%zmm21              \n\t" // N2
        "vandps      %%zmm17, %%zmm0 , %%zmm17              \n\t" // N2 - dup - 17 - 21
        "vmovups     %%zmm30, %%zmm6                        \n\t" // restore from 30 - 6
        "vmovups  -64(%3, %0, 4)    , %%zmm19               \n\t" // L4

        "vpcmpeqd       %%zmm24, %%zmm4 , %%k0              \n\t" // C1
        "vmovups     %%zmm8 , %%zmm28                       \n\t" // chpt1 - 8 - 28
        "vaddps      %%zmm21, %%zmm5 , %%zmm25              \n\t" // D2
        "vaddps      %%zmm21, %%zmm5 , %%zmm5               \n\t" // D2 - dup - 25 - 5
        "vandps      %%zmm18, %%zmm0 , %%zmm22              \n\t" // N3
        "vandps      %%zmm18, %%zmm0 , %%zmm18              \n\t" // N3 - dup - 18 - 22
        "vmovups     %%zmm31, %%zmm7                        \n\t" // restore from 31 - 7
        "vmovups       (%3, %0, 4)  , %%zmm16               \n\t" // L5

        "vpcmpeqd       %%zmm25, %%zmm5 , %%k1              \n\t" // C2
        "vmovups     %%zmm9 , %%zmm29                       \n\t" // chpt2 - 9 - 29
        "vaddps      %%zmm22, %%zmm6 , %%zmm26              \n\t" // D3
        "vaddps      %%zmm22, %%zmm6 , %%zmm6               \n\t" // D3 - dup - 26 - 6
        "vandps      %%zmm19, %%zmm0 , %%zmm23              \n\t" // N4
        "vandps      %%zmm19, %%zmm0 , %%zmm19              \n\t" // N4 - dup - 23 - 19
        "vmovups     %%zmm4 , %%zmm8                        \n\t" // B5
        "vmovups     64(%3, %0, 4)  , %%zmm17               \n\t" // L6

        "kandw          %%k0   , %%k1   , %%k0              \n\t" // check - red - 0 - 1

        "vpcmpeqd       %%zmm26, %%zmm6 , %%k2              \n\t" // C3
        "vmovups     %%zmm10, %%zmm30                       \n\t" // chpt3 - 10 - 30
        "vaddps      %%zmm23, %%zmm7 , %%zmm27              \n\t" // D4
        "vaddps      %%zmm23, %%zmm7 , %%zmm7               \n\t" // D4 - dup - 27 - 7
        "vandps      %%zmm16, %%zmm0 , %%zmm20              \n\t" // N5
        "vandps      %%zmm16, %%zmm0 , %%zmm16              \n\t" // N5 - dup - 20 - 16
        "vmovups     %%zmm5 , %%zmm9                        \n\t" // B6
        "vmovups    128(%3, %0, 4)  , %%zmm18               \n\t" // L7

        "vpcmpeqd       %%zmm27, %%zmm7 , %%k3              \n\t" // C4
        "vmovups     %%zmm11, %%zmm31                       \n\t" // chpt4 - 11 - 31
        "vaddps      %%zmm16, %%zmm4 , %%zmm24              \n\t" // D5
        "vaddps      %%zmm16, %%zmm4 , %%zmm4               \n\t" // D5 - dup - 24 - 4

        "kandw          %%k2   , %%k3   , %%k2              \n\t" // check - red - 2 - 3

        "vandps      %%zmm17, %%zmm0 , %%zmm21              \n\t" // N6
        "vandps      %%zmm17, %%zmm0 , %%zmm17              \n\t" // N6 - dup - 20 - 16

        "kandw          %%k0   , %%k2   , %%k0              \n\t"

        "vmovups     %%zmm6 , %%zmm10                       \n\t" // B7

        "kxorw          %%k0   , %%k4   , %%k0              \n\t"

        "vmovups    192(%3, %0, 4)  , %%zmm19               \n\t" // L8

        "ktestw         %%k0   , %%k4                       \n\t"
        "jnz            4f                                  \n\t"

        "addq       $64 , %0             \n\t"
        "subq       $64 , %1             \n\t"
        "jnz        1b                   \n\t"
        
        "jmp        2b                   \n\t"

        "4:                              \n\t"
        "vzeroupper             \n\t"

        : "+r"(i), // 0
          "+r"(n),  // 1
          "+r"(err_num)  // 2
        : "r"(x),  // 3
          "r"(res) // 4
        : "cc",
          "%xmm0", "%xmm1", "%xmm2", "%xmm3",
          "%xmm4", "%xmm5", "%xmm6", "%xmm7",
          "memory");
	return err_num;
}

float ft_sasum(long n, float *x, long inc_x) {
  long i = 0;
  long j = 0;
  float sumf = 0.0;
  float res = 0.0;
  long n1;

  if (n <= 0 || inc_x <= 0)
    return sumf;

  if (inc_x == 1) {
    n1 = n & -64;

    if (n1 > 0) {
      long err_num = sasum_kernel(n1, x, &res);
      if (err_num) {
        printf("detected error number: %ld\n", err_num);
      }
      i = n1;
    }
    //printf("res = %f\n", res);
    while (i < n) {
      res += fabs(x[i]);
      i++;
    }

    sumf = res;

  } else {
    long inc = -4 * inc_x;
    long n1 = n & inc;
    //printf("inc = %d, n1 = %d, n = %d\n", inc, n1, n);
    register float sum1, sum2;
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

float ftblas_sasum_ft(const long int n, const float *x, const long int inc_x) {
   return(ft_sasum(n, (float*)x, inc_x));
}