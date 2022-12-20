#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/ftblas.h"

static long dasum_kernel(long n, double *x, double *res)
{

      long register i = 0;
      long err_num = 0;
      __asm__ __volatile__(

            "vxorpd           %%zmm4 , %%zmm4 , %%zmm4                   \n\t"
            "vxorpd           %%zmm5 , %%zmm5 , %%zmm5                   \n\t"
            "vxorpd           %%zmm6 , %%zmm6 , %%zmm6                   \n\t"
            "vxorpd           %%zmm7 , %%zmm7 , %%zmm7                   \n\t"

            "vxorpd           %%zmm8 , %%zmm8 , %%zmm8                   \n\t"
            "vxorpd           %%zmm9 , %%zmm9 , %%zmm9                   \n\t"
            "vxorpd           %%zmm10, %%zmm10, %%zmm10                  \n\t"
            "vxorpd           %%zmm11, %%zmm11, %%zmm11                  \n\t"

            "vpcmpeqd   %%xmm0, %%xmm0, %%xmm0               \n\t"
            "vbroadcastsd     %%xmm0 , %%zmm0                            \n\t"
            "vpsrlq             $1, %%zmm0, %%zmm0           \n\t"

            "kxnorw   %%k4   , %%k4   , %%k4                 \n\t"
            "vandpd      %%zmm16, %%zmm0 , %%zmm20                      \n\t"
            /* Prologue */
            "vmovupd       (%3, %0, 8)    , %%zmm16                     \n\t" // L1

            "vandpd      %%zmm16, %%zmm0 , %%zmm20                      \n\t" // N1
            "vandpd      %%zmm16, %%zmm0 , %%zmm16                      \n\t" // N1 - dup - 20 - 16
            "vmovupd     64(%3, %0, 8)    , %%zmm17                     \n\t" // L2

            "vmovupd     %%zmm4 , %%zmm8                                \n\t" // B1 - 4 - 8
            "vandpd      %%zmm17, %%zmm0 , %%zmm21                      \n\t" // N2
            "vandpd      %%zmm17, %%zmm0 , %%zmm17                      \n\t" // N2 - dup - 21 - 17
            "vmovupd    128(%3, %0, 8)    , %%zmm18                     \n\t" // L3

            "vaddpd      %%zmm20, %%zmm4 , %%zmm24                      \n\t" // A1
            "vaddpd      %%zmm16, %%zmm4 , %%zmm4                       \n\t" // A1 - dup - 24 - 4
            "vmovupd     %%zmm5 , %%zmm9                                \n\t" // B2 - 5 - 9
            "vandpd      %%zmm18, %%zmm0 , %%zmm22                      \n\t" // N3
            "vandpd      %%zmm18, %%zmm0 , %%zmm18                      \n\t" // N3 - dup - 22 - 18
            "vmovupd    192(%3, %0, 8)    , %%zmm19                     \n\t" // L4

            "addq       $32 , %0               \n\t"
            "subq     $32 , %1                 \n\t"
            "jz             2f                         \n\t"

            ".p2align 4                                \n\t"

            /* Loop Body */
            "1:                                  \n\t"
            "prefetcht0        1024(%3, %0, 8)                 \n\t" // PX
            "vpcmpeqd     %%zmm24, %%zmm4 , %%k0                \n\t" // C1
            "vmovupd     %%zmm8 , %%zmm28                               \n\t" // chkpt - 8 - 28
            "vaddpd      %%zmm21, %%zmm5 , %%zmm25                      \n\t" // A2
            "vaddpd      %%zmm17, %%zmm5 , %%zmm5                       \n\t" // A2 - dup - 25 - 5
            "vmovupd     %%zmm6 , %%zmm10                               \n\t" // B3 - 6 - 10
            "vandpd      %%zmm19, %%zmm0 , %%zmm23                      \n\t" // N4
            "vandpd      %%zmm19, %%zmm0 , %%zmm19                      \n\t" // N4 - dup - 23 - 19
            "vmovupd       (%3, %0, 8)    , %%zmm16                     \n\t" // L5

            "vpcmpeqd     %%zmm25, %%zmm5 , %%k1                \n\t" // C2
            "vmovupd     %%zmm9 , %%zmm29                               \n\t" // chkpt - 9 - 29
            "vaddpd      %%zmm22, %%zmm6 , %%zmm26                      \n\t" // A3
            "vaddpd      %%zmm18, %%zmm6 , %%zmm6                       \n\t" // A3 - dup - 26 - 6
            "vmovupd     %%zmm7 , %%zmm11                               \n\t" // B4 - 7 - 11
            "vandpd      %%zmm16, %%zmm0 , %%zmm20                      \n\t" // N5
            "vandpd      %%zmm16, %%zmm0 , %%zmm16                      \n\t" // N5 - dup - 20 - 16
            "vmovupd     64(%3, %0, 8)    , %%zmm17                     \n\t" // L6

            "kandw        %%k0   , %%k1   , %%k0                \n\t" // check - red - 0 - 1

            "prefetcht0        1152(%3, %0, 8)                 \n\t" // PX
            "vpcmpeqd     %%zmm25, %%zmm5 , %%k2                \n\t" // C3
            "vmovupd     %%zmm10, %%zmm30                               \n\t" // chkpt - 10 - 30
            "vaddpd      %%zmm23, %%zmm7 , %%zmm27                      \n\t" // A4
            "vaddpd      %%zmm19, %%zmm7 , %%zmm7                       \n\t" // A4 - dup - 27 - 7
            "vmovupd     %%zmm4 , %%zmm8                                \n\t" // B5 - 4 - 8
            "vandpd      %%zmm17, %%zmm0 , %%zmm21                      \n\t" // N6
            "vandpd      %%zmm17, %%zmm0 , %%zmm17                      \n\t" // N6 - dup - 21 - 17
            "vmovupd    128(%3, %0, 8)    , %%zmm18                     \n\t" // L7

            "vpcmpeqd     %%zmm27, %%zmm7 , %%k3                \n\t" // C4
            "vmovupd     %%zmm11, %%zmm31                               \n\t" // chkpt - 11 - 31
            "kandw        %%k2   , %%k3   , %%k2                \n\t"
            "vaddpd      %%zmm20, %%zmm4 , %%zmm24                      \n\t" // A5
            "vaddpd      %%zmm16, %%zmm4 , %%zmm4                       \n\t" // A5 - dup - 24 - 4

            "kandw        %%k0   , %%k2   , %%k0                \n\t"

            "vmovupd     %%zmm5 , %%zmm9                                \n\t" // B6 - 5 - 9

            "kxorw        %%k0   , %%k4   , %%k0                \n\t"

            "vandpd      %%zmm18, %%zmm0 , %%zmm22                      \n\t" // N7
            "vandpd      %%zmm18, %%zmm0 , %%zmm18                      \n\t" // N7 - dup - 22 - 18
            "vmovupd    192(%3, %0, 8)    , %%zmm19                     \n\t" // L8

            "addq       $32 , %0               \n\t"

            "ktestw       %%k0   , %%k4                         \n\t"
            "jnz          3f           \n\t"

            
            "subq     $32 , %1                 \n\t"
            "jnz        1b                       \n\t"


            "2:                                        \n\t"
            /* Epilogue */
            "vpcmpeqd     %%zmm24, %%zmm4 , %%k0                \n\t" // C1
            "vmovupd     %%zmm8 , %%zmm28                               \n\t" // chkpt - 8 - 28
            "vaddpd      %%zmm21, %%zmm5 , %%zmm25                      \n\t" // A2
            "vaddpd      %%zmm17, %%zmm5 , %%zmm5                       \n\t" // A2 - dup - 25 - 5
            "vmovupd     %%zmm6 , %%zmm10                               \n\t" // B3 - 6 - 10
            "vandpd      %%zmm19, %%zmm0 , %%zmm23                      \n\t" // N4
            "vandpd      %%zmm19, %%zmm0 , %%zmm19                      \n\t" // N4 - dup - 23 - 19

            "vpcmpeqd     %%zmm25, %%zmm5 , %%k1                \n\t" // C2
            "vmovupd     %%zmm9 , %%zmm29                               \n\t" // chkpt - 9 - 29
            "vaddpd      %%zmm22, %%zmm6 , %%zmm26                      \n\t" // A3
            "vaddpd      %%zmm18, %%zmm6 , %%zmm6                       \n\t" // A3 - dup - 26 - 6
            "vmovupd     %%zmm7 , %%zmm11                               \n\t" // B4 - 7 - 11

            "kandw        %%k0   , %%k1   , %%k0                \n\t" // check - red - 0 - 1

            "vpcmpeqd     %%zmm25, %%zmm5 , %%k2                \n\t" // C3
            "vmovupd     %%zmm10, %%zmm30                               \n\t" // chkpt - 10 - 30
            "vaddpd      %%zmm23, %%zmm7 , %%zmm27                      \n\t" // A4
            "vaddpd      %%zmm19, %%zmm7 , %%zmm7                       \n\t" // A4 - dup - 27 - 7

            "vpcmpeqd     %%zmm27, %%zmm7 , %%k3                \n\t" // C4
            "vmovupd     %%zmm11, %%zmm31                               \n\t" // chkpt - 11 - 31

            

            "kandw        %%k2   , %%k3   , %%k2                \n\t"
            "kandw        %%k0   , %%k2   , %%k0                \n\t"
            "kxorw        %%k0   , %%k4   , %%k0                \n\t"
            "ktestw       %%k0   , %%k4                         \n\t"
            "jnz          6f           \n\t"

            "5:                                                                           \n\t"
            "vaddpd       %%zmm4 , %%zmm5 , %%zmm4              \n\t"
            "vaddpd       %%zmm6 , %%zmm7 , %%zmm6              \n\t"
            "vaddpd       %%zmm4 , %%zmm6 , %%zmm4              \n\t"

            "vextractf64x4     $0, %%zmm4 , %%ymm1              \n\t"
            "vextractf64x4     $1, %%zmm4 , %%ymm2              \n\t"
            "vaddpd       %%ymm1 , %%ymm2 , %%ymm1              \n\t"

            "vextractf128         $1 , %%ymm1 , %%xmm2              \n\t"
            "vaddpd       %%xmm1 , %%xmm2 , %%xmm1                \n\t"
            "vhaddpd      %%xmm1 , %%xmm1 , %%xmm1                \n\t"

            "vmovsd           %%xmm1 ,    (%4)                                \n\t"
            "jmp        4f                                                          \n\t"

            "3:                                        \n\t"
            "addq     $1  , %2                 \n\t"
            "subq     $32 , %0                 \n\t"
            "vmovupd -256(%3, %0, 8)      , %%zmm16                     \n\t" // L1

            "vandpd      %%zmm16, %%zmm0 , %%zmm20                      \n\t" // N1
            "vandpd      %%zmm16, %%zmm0 , %%zmm16                      \n\t" // N1 - dup - 20 - 16
            "vmovupd -192(%3, %0, 8)      , %%zmm17                     \n\t" // L2

            "vmovupd     %%zmm28, %%zmm4                                \n\t" // restore - 28 - 4
            "vandpd      %%zmm17, %%zmm0 , %%zmm21                      \n\t" // N2
            "vandpd      %%zmm17, %%zmm0 , %%zmm17                      \n\t" // N2 - dup - 21 - 17
            "vmovupd -128(%3, %0, 8)      , %%zmm18                     \n\t" // L3

            "vaddpd      %%zmm20, %%zmm4 , %%zmm24                      \n\t" // A1
            "vaddpd      %%zmm16, %%zmm4 , %%zmm4                       \n\t" // A1 - dup - 24 - 4
            "vmovupd     %%zmm29, %%zmm5                                \n\t" // restore - 29 - 5
            "vandpd      %%zmm18, %%zmm0 , %%zmm22                      \n\t" // N3
            "vandpd      %%zmm18, %%zmm0 , %%zmm18                      \n\t" // N3 - dup - 22 - 18
            "vmovupd  -64(%3, %0, 8)      , %%zmm19                     \n\t" // L4

            "vpcmpeqd     %%zmm24, %%zmm4 , %%k0                \n\t" // C1
            "vmovupd     %%zmm8 , %%zmm28                               \n\t" // chkpt - 8 - 28
            "vaddpd      %%zmm21, %%zmm5 , %%zmm25                      \n\t" // A2
            "vaddpd      %%zmm17, %%zmm5 , %%zmm5                       \n\t" // A2 - dup - 25 - 5
            "vmovupd     %%zmm30, %%zmm6                                \n\t" // restore - 30 - 6
            "vandpd      %%zmm19, %%zmm0 , %%zmm23                      \n\t" // N4
            "vandpd      %%zmm19, %%zmm0 , %%zmm19                      \n\t" // N4 - dup - 23 - 19
            "vmovupd       (%3, %0, 8)    , %%zmm16                     \n\t" // L5

            "vpcmpeqd     %%zmm25, %%zmm5 , %%k1                \n\t" // C2
            "vmovupd     %%zmm9 , %%zmm29                               \n\t" // chkpt - 9 - 29
            "vaddpd      %%zmm22, %%zmm6 , %%zmm26                      \n\t" // A3
            "vaddpd      %%zmm18, %%zmm6 , %%zmm6                       \n\t" // A3 - dup - 26 - 6
            "vmovupd     %%zmm31, %%zmm7                                \n\t" // restore - 31 - 7
            "vandpd      %%zmm16, %%zmm0 , %%zmm20                      \n\t" // N5
            "vandpd      %%zmm16, %%zmm0 , %%zmm16                      \n\t" // N5 - dup - 20 - 16
            "vmovupd     64(%3, %0, 8)    , %%zmm17                     \n\t" // L6

            "kandw        %%k0   , %%k1   , %%k0                \n\t" // check - red - 0 - 1

            "vpcmpeqd     %%zmm25, %%zmm5 , %%k2                \n\t" // C3
            "vmovupd     %%zmm10, %%zmm30                               \n\t" // chkpt - 10 - 30
            "vaddpd      %%zmm23, %%zmm7 , %%zmm27                      \n\t" // A4
            "vaddpd      %%zmm19, %%zmm7 , %%zmm7                       \n\t" // A4 - dup - 27 - 7
            "vmovupd     %%zmm4 , %%zmm8                                \n\t" // B5 - 4 - 8
            "vandpd      %%zmm17, %%zmm0 , %%zmm21                      \n\t" // N6
            "vandpd      %%zmm17, %%zmm0 , %%zmm17                      \n\t" // N6 - dup - 21 - 17
            "vmovupd    128(%3, %0, 8)    , %%zmm18                     \n\t" // L7

            "vpcmpeqd     %%zmm27, %%zmm7 , %%k3                \n\t" // C4
            "vmovupd     %%zmm11, %%zmm31                               \n\t" // chkpt - 11 - 31
            "kandw        %%k2   , %%k3   , %%k2                \n\t"
            "vaddpd      %%zmm20, %%zmm4 , %%zmm24                      \n\t" // A5
            "vaddpd      %%zmm16, %%zmm4 , %%zmm4                       \n\t" // A5 - dup - 24 - 4

            "kandw        %%k0   , %%k2   , %%k0                \n\t"

            "vmovupd     %%zmm5 , %%zmm9                                \n\t" // B6 - 5 - 9

            "kxorw        %%k0   , %%k4   , %%k0                \n\t"

            "vandpd      %%zmm18, %%zmm0 , %%zmm22                      \n\t" // N7
            "vandpd      %%zmm18, %%zmm0 , %%zmm18                      \n\t" // N7 - dup - 22 - 18
            "vmovupd    192(%3, %0, 8)    , %%zmm19                     \n\t" // L8

            "addq       $32 , %0               \n\t"

            "ktestw       %%k0   , %%k4                         \n\t"
            "jnz          7f           \n\t"

            
            "subq     $32 , %1                 \n\t"
            "jnz        1b                       \n\t"
            
            "jmp        2b                       \n\t"

            "6:                                                                           \n\t"
            /* Restart from epilogue */
            "addq     $1  , %2                 \n\t"
            "vmovupd -256(%3, %0, 8)      , %%zmm16                     \n\t" // L1

            "vandpd      %%zmm16, %%zmm0 , %%zmm20                      \n\t" // N1
            "vandpd      %%zmm16, %%zmm0 , %%zmm16                      \n\t" // N1 - dup - 20 - 16
            "vmovupd -192(%3, %0, 8)      , %%zmm17                     \n\t" // L2

            "vmovupd     %%zmm28, %%zmm4                                \n\t" // restore - 28 - 4
            "vandpd      %%zmm17, %%zmm0 , %%zmm21                      \n\t" // N2
            "vandpd      %%zmm17, %%zmm0 , %%zmm17                      \n\t" // N2 - dup - 21 - 17
            "vmovupd -128(%3, %0, 8)      , %%zmm18                     \n\t" // L3

            "vaddpd      %%zmm20, %%zmm4 , %%zmm24                      \n\t" // A1
            "vaddpd      %%zmm16, %%zmm4 , %%zmm4                       \n\t" // A1 - dup - 24 - 4
            "vmovupd     %%zmm29, %%zmm5                                \n\t" // restore - 29 - 5
            "vandpd      %%zmm18, %%zmm0 , %%zmm22                      \n\t" // N3
            "vandpd      %%zmm18, %%zmm0 , %%zmm18                      \n\t" // N3 - dup - 22 - 18
            "vmovupd  -64(%3, %0, 8)      , %%zmm19                     \n\t" // L4

            "vpcmpeqd     %%zmm24, %%zmm4 , %%k0                \n\t" // C1
            "vmovupd     %%zmm8 , %%zmm28                               \n\t" // chkpt - 8 - 28
            "vaddpd      %%zmm21, %%zmm5 , %%zmm25                      \n\t" // A2
            "vaddpd      %%zmm17, %%zmm5 , %%zmm5                       \n\t" // A2 - dup - 25 - 5
            "vmovupd     %%zmm30, %%zmm6                                \n\t" // restore - 30 - 6
            "vandpd      %%zmm19, %%zmm0 , %%zmm23                      \n\t" // N4
            "vandpd      %%zmm19, %%zmm0 , %%zmm19                      \n\t" // N4 - dup - 23 - 19

            "vpcmpeqd     %%zmm25, %%zmm5 , %%k1                \n\t" // C2
            "vmovupd     %%zmm9 , %%zmm29                               \n\t" // chkpt - 9 - 29
            "vaddpd      %%zmm22, %%zmm6 , %%zmm26                      \n\t" // A3
            "vaddpd      %%zmm18, %%zmm6 , %%zmm6                       \n\t" // A3 - dup - 26 - 6
            "vmovupd     %%zmm31, %%zmm7                                \n\t" // restore - 31 - 7

            "kandw        %%k0   , %%k1   , %%k0                \n\t" // check - red - 0 - 1

            "vpcmpeqd     %%zmm25, %%zmm5 , %%k2                \n\t" // C3
            "vmovupd     %%zmm10, %%zmm30                               \n\t" // chkpt - 10 - 30
            "vaddpd      %%zmm23, %%zmm7 , %%zmm27                      \n\t" // A4
            "vaddpd      %%zmm19, %%zmm7 , %%zmm7                       \n\t" // A4 - dup - 27 - 7

            "vpcmpeqd     %%zmm27, %%zmm7 , %%k3                \n\t" // C4
            "vmovupd     %%zmm11, %%zmm31                               \n\t" // chkpt - 11 - 31

            "kandw        %%k2   , %%k3   , %%k2                \n\t"
            "kandw        %%k0   , %%k2   , %%k0                \n\t"
            "kxorw        %%k0   , %%k4   , %%k0                \n\t"
            "ktestw       %%k0   , %%k4                         \n\t"
            
            "jnz          7f           \n\t"
            "jmp          5b           \n\t"

            "7:										   \n\t"
            "addq     $1  , %2                 \n\t"

            "4:                                        \n\t"

            : "+r"(i), // 0
              "+r"(n), // 1
              "+r"(err_num) // 2
            : "r"(x),  // 3
              "r"(res) // 4
            : "cc",
              "%xmm0", "%xmm1", "%xmm2", "%xmm3",
              "%xmm4", "%xmm5", "%xmm6", "%xmm7",
              "memory");
	return err_num;
}

double ftblas_dasum_ft(const long int n, const double *x, const long int inc_x) {
  long int i = 0;
  long int j = 0;
  double sumf = 0.0;
  double res = 0.0;
  long int n1;

  if (n <= 0 || inc_x <= 0)
    return sumf;

  if (inc_x == 1) {
    n1 = n & -32;

    if (n1 > 0) {
      long int err_num = dasum_kernel(n1, (double *)x, &res);
      if (err_num) printf("detected error number %ld\n", err_num);
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