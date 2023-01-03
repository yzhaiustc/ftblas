#include <stdio.h>
#include <stdlib.h>
#include "../include/ftblas.h"

static void ori_dnrm2_kernel(long int n, double *x, double *res)
{

	long int register i = 0;

	__asm__ __volatile__(
		"vxorpd		%%zmm4 , %%zmm4 , %%zmm4	             \n\t"
		"vxorpd		%%zmm5 , %%zmm5 , %%zmm5	             \n\t"
		"vxorpd		%%zmm6 , %%zmm6 , %%zmm6	             \n\t"
		"vxorpd		%%zmm7 , %%zmm7 , %%zmm7	             \n\t"

		".p2align 4				             \n\t"
		"1:				             \n\t"
		"vmovupd                   (%2,%0,8), %%zmm16         \n\t"
		"vmovupd                 64(%2,%0,8), %%zmm17         \n\t"

		"vfmadd231pd           %%zmm16, %%zmm16, %%zmm4       \n\t"
		"vfmadd231pd           %%zmm17, %%zmm17, %%zmm5       \n\t"

		"vmovupd                128(%2,%0,8), %%zmm18         \n\t"
		"vmovupd                192(%2,%0,8), %%zmm19         \n\t"

		"vfmadd231pd           %%zmm18, %%zmm18, %%zmm6       \n\t"
		"vfmadd231pd           %%zmm19, %%zmm19, %%zmm7       \n\t"

		"addq		$32 , %0	  	     \n\t"
		"subq	    $32 , %1		     \n\t"
		"jnz		1b		             \n\t"

		"vaddpd       %%zmm4 , %%zmm5 , %%zmm4              \n\t"
		"vaddpd       %%zmm6 , %%zmm7 , %%zmm6              \n\t"
		"vaddpd       %%zmm4 , %%zmm6 , %%zmm4              \n\t"

		"vextractf64x4     $0, %%zmm4 , %%ymm1              \n\t"
		"vextractf64x4     $1, %%zmm4 , %%ymm2              \n\t"
		"vaddpd       %%ymm1 , %%ymm2 , %%ymm1              \n\t"

		"vextractf128	    $1 , %%ymm1 , %%xmm2	            \n\t"
		"vaddpd       %%xmm1 , %%xmm2 , %%xmm1	            \n\t"
		"vhaddpd      %%xmm1 , %%xmm1 , %%xmm1	            \n\t"

		"vmovsd		%%xmm1,    (%3)	\n\t"
		"vzeroupper				\n\t"


		: "+r"(i), // 0
		  "+r"(n)  // 1
		: "r"(x),  //2
		  "r"(res) // 3
		: "cc",
		  "%xmm0", "%xmm1",
		  "%xmm2", "%xmm3",
		  "%xmm4", "%xmm5",
		  "%xmm6", "%xmm7",
		  "memory");
}

double oridnrm2_compute(long int n, double *x, long int inc_x)
{
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
      ori_dnrm2_kernel(n1, x, &res);
      i = n1;
    }
    while (i < n) {
      res += x[i] * x[i];
      i++;
    }

    sumf = res;

  } else {
    long int inc = -4 * inc_x;
    long int n1 = n & inc;
    register double sum1, sum2, sum3, sum4;
    sum1 = 0.0;
    sum2 = 0.0;
    sum3 = 0.0;
    sum4 = 0.0;
    while (j < n1) {
      sum1 += x[i] * x[i];
      sum2 += x[i + inc_x] * x[i + inc_x];
      sum3 += x[i + 2 * inc_x] * x[i + 2 * inc_x];
      sum4 += x[i + 3 * inc_x] * x[i + 3 * inc_x];

      i += inc_x * 4;
      j -= inc;

    }
    sumf = sum1 + sum2 + sum3 + sum4;
    while (i < n) {
      sumf += x[i] * x[i];
      i += inc_x;
      j++;
    }

  }
  
  return sumf;
}

double ftblas_dnrm2_ori(const long int n, const double *x, const long int inc_x)
{
  double res = 0.;
  int tid, TOTAL_THREADS = atoi(getenv("OMP_NUM_THREADS"));
  if (TOTAL_THREADS <= 1 || n < 4000000)
  {
    res += oridnrm2_compute(n, (double *)x, inc_x);
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
      res += oridnrm2_compute(EDGE_LEN, (double *)x, inc_x);
    else
      res += oridnrm2_compute(DIM_LEN,(double *)(x + EDGE_LEN + (tid - 1) * DIM_LEN), inc_x);
  }
  return sqrt(res);
}