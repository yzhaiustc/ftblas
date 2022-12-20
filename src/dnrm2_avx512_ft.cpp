#include <stdio.h>
#include <stdlib.h>
#include "../include/ftblas.h"

static long int dnrm2_kernel(long int n, double *x, double *res)
{

	long int register i = 0;
	long int register err_num = 0;
	
	__asm__ __volatile__(
		"vxorpd		%%zmm4 , %%zmm4 , %%zmm4	             \n\t"
		"vxorpd		%%zmm5 , %%zmm5 , %%zmm5	             \n\t"
		"vxorpd		%%zmm6 , %%zmm6 , %%zmm6	             \n\t"
		"vxorpd		%%zmm7 , %%zmm7 , %%zmm7	             \n\t"

		"vxorpd		%%zmm24, %%zmm24, %%zmm24	             \n\t"
		"vxorpd		%%zmm25, %%zmm25, %%zmm25	             \n\t"
		"vxorpd		%%zmm26, %%zmm26, %%zmm26	             \n\t"
		"vxorpd		%%zmm27, %%zmm27, %%zmm27	             \n\t"

		"kxnorw   %%k4   , %%k4   , %%k4                 \n\t"

		/* Prologue */
		"vmovupd                   (%3,%0,8), %%zmm16         \n\t" // L1
		"vmovupd     %%zmm4 , %%zmm8						  \n\t" // B1
		"vmovupd                 64(%3,%0,8), %%zmm17         \n\t" // L2
		"vfmadd231pd           %%zmm16, %%zmm16, %%zmm24      \n\t" // F1-A
		"vmovupd     %%zmm5 , %%zmm9						  \n\t" // B2
		"vmovupd                128(%3,%0,8), %%zmm18         \n\t" // L3
		"vfmadd231pd           %%zmm16, %%zmm16, %%zmm4       \n\t" // F1-B
		"vfmadd231pd           %%zmm17, %%zmm17, %%zmm25      \n\t" // F2-A
		"vmovupd     %%zmm6 , %%zmm10						  \n\t" // B3
		"vmovupd                192(%3,%0,8), %%zmm19         \n\t" // L4

		"addq		$32 , %0	  	     \n\t"
		"subq	    $32 , %1		     \n\t"
		"jz 		2f		             \n\t"

		".p2align 4				             \n\t"
		"1:				             \n\t"
		"vpcmpeqd     		   %%zmm24, %%zmm4 , %%k0         \n\t" // C1
		"vmovupd     %%zmm8 , %%zmm28						  \n\t" // chkpt - 28 - 8
		"vfmadd231pd           %%zmm17, %%zmm17, %%zmm5       \n\t" // F2-B
		"vfmadd231pd           %%zmm18, %%zmm18, %%zmm26      \n\t" // F3-A
		"vmovupd     %%zmm7 , %%zmm11						  \n\t" // B4
		"vmovupd                   (%3,%0,8), %%zmm16         \n\t" // L5

		"vpcmpeqd     		   %%zmm25, %%zmm5 , %%k1         \n\t" // C2
		"vmovupd     %%zmm9 , %%zmm29						  \n\t" // chkpt - 29 - 9
		"vfmadd231pd           %%zmm18, %%zmm18, %%zmm6       \n\t" // F3-B
		"vfmadd231pd           %%zmm19, %%zmm19, %%zmm27      \n\t" // F4-A
		"vmovupd     %%zmm4 , %%zmm8						  \n\t" // B5
		"vmovupd                 64(%3,%0,8), %%zmm17         \n\t" // L6

		"kandw        		   %%k0   , %%k1   , %%k0         \n\t" // check - 0 - 1

		"vpcmpeqd     		   %%zmm26, %%zmm6 , %%k2         \n\t" // C3
		"vmovupd     %%zmm10, %%zmm30						  \n\t" // chkpt - 30 - 10
		"vfmadd231pd           %%zmm19, %%zmm19, %%zmm7       \n\t" // F4-B
		"vfmadd231pd           %%zmm16, %%zmm16, %%zmm24      \n\t" // F5-A
		"vmovupd     %%zmm5 , %%zmm9						  \n\t" // B6
		"vmovupd                128(%3,%0,8), %%zmm18         \n\t" // L7


		"vpcmpeqd     		   %%zmm27, %%zmm7 , %%k3         \n\t" // C4
		"vmovupd     %%zmm11, %%zmm31						  \n\t" // chkpt - 31 - 11

        "kandw          %%k2   , %%k3   , %%k2              \n\t" // check - red - 2 - 3

		"vfmadd231pd           %%zmm16, %%zmm16, %%zmm4       \n\t" // F5-B

        "kandw          %%k0   , %%k2   , %%k0              \n\t"

		"vfmadd231pd           %%zmm17, %%zmm17, %%zmm25      \n\t" // F6-A

        "kxorw          %%k0   , %%k4   , %%k0              \n\t"

		"vmovupd     %%zmm6 , %%zmm10						  \n\t" // B7

       

		"vmovupd                192(%3,%0,8), %%zmm19         \n\t" // L8

		"addq		$32 , %0	  	     \n\t"
		"ktestw         %%k0   , %%k4                       \n\t"
        "jnz            3f                                  \n\t"

		"subq	    $32 , %1		     \n\t"
		"jnz		1b		             \n\t"

		/* Prologue */
		"2: \n\t"
		"vpcmpeqd     		   %%zmm24, %%zmm4 , %%k0         \n\t" // C1
		"vmovupd     %%zmm8 , %%zmm28						  \n\t" // chkpt - 28 - 8
		"vfmadd231pd           %%zmm17, %%zmm17, %%zmm5       \n\t" // F2-B
		"vfmadd231pd           %%zmm18, %%zmm18, %%zmm26      \n\t" // F3-A
		"vmovupd     %%zmm7 , %%zmm11						  \n\t" // B4

		"vpcmpeqd     		   %%zmm25, %%zmm5 , %%k1         \n\t" // C2
		"vmovupd     %%zmm9 , %%zmm29						  \n\t" // chkpt - 29 - 9
		"vfmadd231pd           %%zmm18, %%zmm18, %%zmm6       \n\t" // F3-B
		"vfmadd231pd           %%zmm19, %%zmm19, %%zmm27      \n\t" // F4-A

		"kandw        		   %%k0   , %%k1   , %%k0         \n\t" // check - 0 - 1

		"vpcmpeqd     		   %%zmm26, %%zmm6 , %%k2         \n\t" // C3
		"vmovupd     %%zmm10, %%zmm30						  \n\t" // chkpt - 30 - 10
		"vfmadd231pd           %%zmm19, %%zmm19, %%zmm7       \n\t" // F4-B

		"vpcmpeqd     		   %%zmm27, %%zmm7 , %%k3         \n\t" // C4
		"vmovupd     %%zmm11, %%zmm31						  \n\t" // chkpt - 31 - 11

        "kandw          %%k2   , %%k3   , %%k2              \n\t" // check - red - 2 - 3
        "kandw          %%k0   , %%k2   , %%k0              \n\t"
        "kxorw          %%k0   , %%k4   , %%k0              \n\t"
        "ktestw         %%k0   , %%k4                       \n\t"
        "jnz            6f                                  \n\t"

		/* Reduction */
		"5: \n\t"
		"vaddpd       %%zmm4 , %%zmm5 , %%zmm4              \n\t"
		"vaddpd       %%zmm6 , %%zmm7 , %%zmm6              \n\t"
		"vaddpd       %%zmm4 , %%zmm6 , %%zmm4              \n\t"

		"vextractf64x4     $0, %%zmm4 , %%ymm1              \n\t"
		"vextractf64x4     $1, %%zmm4 , %%ymm2              \n\t"
		"vaddpd       %%ymm1 , %%ymm2 , %%ymm1              \n\t"

		"vextractf128	    $1 , %%ymm1 , %%xmm2	            \n\t"
		"vaddpd       %%xmm1 , %%xmm2 , %%xmm1	            \n\t"
		"vhaddpd      %%xmm1 , %%xmm1 , %%xmm1	            \n\t"

		"vmovsd		%%xmm1,    (%4)	\n\t"
		"jmp		4f 					 \n\t"

		"3:				             \n\t"
		"subq	    $32 , %0		     \n\t"
		"vmovupd               -256(%3,%0,8), %%zmm16         \n\t" // L1
		"vmovupd     %%zmm28, %%zmm4						  \n\t" // restore - 28 - 4
		"vmovupd     %%zmm28, %%zmm24   					  \n\t" // restore - 28 - 24
		"vmovupd               -192(%3,%0,8), %%zmm17         \n\t" // L2
		"vfmadd231pd           %%zmm16, %%zmm16, %%zmm24      \n\t" // F1-A
		"vmovupd     %%zmm29, %%zmm5						  \n\t" // restore - 29 - 5
		"vmovupd     %%zmm29, %%zmm25						  \n\t" // restore - 29 - 25
		"vmovupd               -128(%3,%0,8), %%zmm18         \n\t" // L3
		"vfmadd231pd           %%zmm16, %%zmm16, %%zmm4       \n\t" // F1-B
		"vfmadd231pd           %%zmm17, %%zmm17, %%zmm25      \n\t" // F2-A
		"vmovupd     %%zmm30, %%zmm6						  \n\t" // restore - 30 - 6
		"vmovupd     %%zmm30, %%zmm26						  \n\t" // restore - 30 - 26
		"vmovupd                -64(%3,%0,8), %%zmm19         \n\t" // L4

		"vpcmpeqd     		   %%zmm24, %%zmm4 , %%k0         \n\t" // C1
		"vmovupd     %%zmm8 , %%zmm28						  \n\t" // chkpt - 28 - 8
		"vfmadd231pd           %%zmm17, %%zmm17, %%zmm5       \n\t" // F2-B
		"vfmadd231pd           %%zmm18, %%zmm18, %%zmm26      \n\t" // F3-A
		"vmovupd     %%zmm31, %%zmm7						  \n\t" // restore - 31 - 7
		"vmovupd     %%zmm31, %%zmm27						  \n\t" // restore - 31 - 27
		"vmovupd                   (%3,%0,8), %%zmm16         \n\t" // L5

		"vpcmpeqd     		   %%zmm25, %%zmm5 , %%k1         \n\t" // C2
		"vmovupd     %%zmm9 , %%zmm29						  \n\t" // chkpt - 29 - 9
		"vfmadd231pd           %%zmm18, %%zmm18, %%zmm6       \n\t" // F3-B
		"vfmadd231pd           %%zmm19, %%zmm19, %%zmm27      \n\t" // F4-A
		"vmovupd     %%zmm4 , %%zmm8						  \n\t" // B5
		"vmovupd                 64(%3,%0,8), %%zmm17         \n\t" // L6

		"kandw        		   %%k0   , %%k1   , %%k0         \n\t" // check - 0 - 1

		"vpcmpeqd     		   %%zmm26, %%zmm6 , %%k2         \n\t" // C3
		"vmovupd     %%zmm10, %%zmm30						  \n\t" // chkpt - 30 - 10
		"vfmadd231pd           %%zmm19, %%zmm19, %%zmm7       \n\t" // F4-B
		"vfmadd231pd           %%zmm16, %%zmm16, %%zmm24      \n\t" // F5-A
		"vmovupd     %%zmm5 , %%zmm9						  \n\t" // B6
		"vmovupd                128(%3,%0,8), %%zmm18         \n\t" // L7


		"vpcmpeqd     		   %%zmm27, %%zmm7 , %%k3         \n\t" // C4
		"vmovupd     %%zmm11, %%zmm31						  \n\t" // chkpt - 31 - 11
		"vfmadd231pd           %%zmm16, %%zmm16, %%zmm4       \n\t" // F5-B
		"vfmadd231pd           %%zmm17, %%zmm17, %%zmm25      \n\t" // F6-A
		"vmovupd     %%zmm6 , %%zmm10						  \n\t" // B7
		"vmovupd                192(%3,%0,8), %%zmm19         \n\t" // L8

        "kandw          %%k2   , %%k3   , %%k2              \n\t" // check - red - 2 - 3
        "kandw          %%k0   , %%k2   , %%k0              \n\t"
        "kxorw          %%k0   , %%k4   , %%k0              \n\t"
        "ktestw         %%k0   , %%k4                       \n\t"
        "jnz            7f                                  \n\t"

		"addq		$32 , %0	  	     \n\t"
		"subq	    $32 , %1		     \n\t"
		"jnz		1b		             \n\t"
		"jmp		2b 					 \n\t"

		"6:				             \n\t"
		"vmovupd               -256(%3,%0,8), %%zmm16         \n\t" // L1
		"vmovupd     %%zmm28, %%zmm4						  \n\t" // restore - 28 - 4
		"vmovupd     %%zmm28, %%zmm24   					  \n\t" // restore - 28 - 24
		"vmovupd               -192(%3,%0,8), %%zmm17         \n\t" // L2
		"vfmadd231pd           %%zmm16, %%zmm16, %%zmm24      \n\t" // F1-A
		"vmovupd     %%zmm29, %%zmm5						  \n\t" // restore - 29 - 5
		"vmovupd     %%zmm29, %%zmm25						  \n\t" // restore - 29 - 25
		"vmovupd               -128(%3,%0,8), %%zmm18         \n\t" // L3
		"vfmadd231pd           %%zmm16, %%zmm16, %%zmm4       \n\t" // F1-B
		"vfmadd231pd           %%zmm17, %%zmm17, %%zmm25      \n\t" // F2-A
		"vmovupd     %%zmm30, %%zmm6						  \n\t" // restore - 30 - 6
		"vmovupd     %%zmm30, %%zmm26						  \n\t" // restore - 30 - 26
		"vmovupd                -64(%3,%0,8), %%zmm19         \n\t" // L4

		"vpcmpeqd     		   %%zmm24, %%zmm4 , %%k0         \n\t" // C1
		"vmovupd     %%zmm8 , %%zmm28						  \n\t" // chkpt - 28 - 8
		"vfmadd231pd           %%zmm17, %%zmm17, %%zmm5       \n\t" // F2-B
		"vfmadd231pd           %%zmm18, %%zmm18, %%zmm26      \n\t" // F3-A
		"vmovupd     %%zmm31, %%zmm7						  \n\t" // restore - 31 - 7
		"vmovupd     %%zmm31, %%zmm27						  \n\t" // restore - 31 - 27

		"vpcmpeqd     		   %%zmm25, %%zmm5 , %%k1         \n\t" // C2
		"vmovupd     %%zmm9 , %%zmm29						  \n\t" // chkpt - 29 - 9
		"vfmadd231pd           %%zmm18, %%zmm18, %%zmm6       \n\t" // F3-B
		"vfmadd231pd           %%zmm19, %%zmm19, %%zmm27      \n\t" // F4-A

		"kandw        		   %%k0   , %%k1   , %%k0         \n\t" // check - 0 - 1

		"vpcmpeqd     		   %%zmm26, %%zmm6 , %%k2         \n\t" // C3
		"vmovupd     %%zmm10, %%zmm30						  \n\t" // chkpt - 30 - 10
		"vfmadd231pd           %%zmm19, %%zmm19, %%zmm7       \n\t" // F4-B

		"vpcmpeqd     		   %%zmm27, %%zmm7 , %%k3         \n\t" // C4
		"vmovupd     %%zmm11, %%zmm31						  \n\t" // chkpt - 31 - 11

        "kandw          %%k2   , %%k3   , %%k2              \n\t" // check - red - 2 - 3
        "kandw          %%k0   , %%k2   , %%k0              \n\t"
        "kxorw          %%k0   , %%k4   , %%k0              \n\t"
        "ktestw         %%k0   , %%k4                       \n\t"
        "jnz            7f                                  \n\t"
		"jmp			5b 					 \n\t"

		"7:				             \n\t"
		"addq		$32 , %2	  	     \n\t"

		"4:				             \n\t"
		"vzeroupper             \n\t"

		: "+r"(i), // 0
		  "+r"(n), // 1
		  "+r"(err_num)  // 2
		: "r"(x),  // 3
		  "r"(res) // 4
		: "cc",
		  "%xmm0", "%xmm1",
		  "%xmm2", "%xmm3",
		  "%xmm4", "%xmm5",
		  "%xmm6", "%xmm7",
		  "%xmm8", "%xmm9", "%xmm10", "%xmm11",
		  "%xmm12", "%xmm13", "%xmm14", "%xmm15",
		  "memory");
	return err_num;
}

double ftdnrm2_compute(long int n, double *x, long int inc_x)
{
  long int i = 0;
  long int j = 0;
  double sumf = 0.0;
  double res = 0.0;
  long int n1;

  if (n <= 0 || inc_x <= 0)
    return sumf;

  if (inc_x == 1)
  {
    n1 = n & -32;
    if (n1 > 0)
    {
      long int err_num = dnrm2_kernel(n1, x, &res);
      if (err_num)
        printf("detected error number : %ld\n", err_num);
      i = n1;
    }
    while (i < n)
    {
      res += x[i] * x[i];
      i++;
    }

    sumf = res;
  }
  else
  {
    long int inc = -4 * inc_x;
    long int n1 = n & inc;
    register double sum1, sum2, sum3, sum4;
    sum1 = 0.0;
    sum2 = 0.0;
    sum3 = 0.0;
    sum4 = 0.0;
    while (j < n1)
    {
      sum1 += x[i] * x[i];
      sum2 += x[i + inc_x] * x[i + inc_x];
      sum3 += x[i + 2 * inc_x] * x[i + 2 * inc_x];
      sum4 += x[i + 3 * inc_x] * x[i + 3 * inc_x];

      i += inc_x * 4;
      j -= inc;
    }
    sumf = sum1 + sum2 + sum3 + sum4;
    while (i < n)
    {
      sumf += x[i] * x[i];
      i += inc_x;
      j++;
    }
  }
  return sumf;
}

double ftblas_dnrm2_ft(const long int n, const double *x, const long int inc_x)
{
  double res = 0.;
  int tid, TOTAL_THREADS = atoi(getenv("OMP_NUM_THREADS"));
  if (TOTAL_THREADS <= 1 || n < 4000000)
  {
    res += ftdnrm2_compute(n, (double *)x, inc_x);
    printf("res=%f\n",res);
    return sqrt(res);
  }
  int max_cpu_num = (int)sysconf(_SC_NPROCESSORS_ONLN);
  if (TOTAL_THREADS > max_cpu_num)
    TOTAL_THREADS = max_cpu_num;
#pragma omp parallel for schedule(static) reduction(+ \
                                                    : res)
  for (tid = 0; tid < TOTAL_THREADS; tid++)
  {
    long int NUM_DIV_NUM_THREADS = n / TOTAL_THREADS * TOTAL_THREADS;
    long int DIM_LEN = n / TOTAL_THREADS;
    long int EDGE_LEN = (NUM_DIV_NUM_THREADS == n) ? n / TOTAL_THREADS : n - NUM_DIV_NUM_THREADS + DIM_LEN;
    if (tid == 0)
      res += ftdnrm2_compute(EDGE_LEN, (double *)x, inc_x);
    else
      res += ftdnrm2_compute(DIM_LEN, (double *)(x + EDGE_LEN + (tid - 1) * DIM_LEN), inc_x);
  }
  return sqrt(res);
}