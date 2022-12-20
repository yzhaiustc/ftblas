#include <stdio.h>
#include <stdlib.h>
#include "../include/ftblas.h"

static long int ft_dscal_kernel(long int n, double *alpha, double *x)
{

    register long int i = 0;
    register long int err_num = 0;
    __asm__ __volatile__(
            "vbroadcastsd       (%3), %%zmm0        \n\t"

            "kxnorw   %%k4   , %%k4   , %%k4                 \n\t"

            /* Prologue */
            "vmovupd    0(%4, %0, 8), %%zmm16       \n\t" // LX1

            "vmulpd       %%zmm16, %%zmm0 , %%zmm20     \n\t" // F11
            "vmovupd   64(%4, %0, 8), %%zmm17       \n\t" // LX2


            "vmulpd       %%zmm16, %%zmm0 , %%zmm24     \n\t" // F21
            "vmulpd       %%zmm17, %%zmm0 , %%zmm21     \n\t" // F12
            "vmovupd  128(%4, %0, 8), %%zmm18       \n\t" // LX3


            "vpcmpeqd     %%zmm20, %%zmm24, %%k0                \n\t" // C1
            "vmulpd       %%zmm17, %%zmm0 , %%zmm25     \n\t" // F22
            "vmulpd       %%zmm18, %%zmm0 , %%zmm22     \n\t" // F13
            "vmovupd  192(%4, %0, 8), %%zmm19       \n\t" // LX4

            "addq       $32, %0             \n\t"
            "subq       $32, %1                 \n\t"
            "jz         2f                      \n\t"

            /* Loop Body */
            ".p2align 4                             \n\t"
            "1:                                     \n\t"
            "vmovupd    %%zmm16, %%zmm28            \n\t" // B1
            "vmovupd    %%zmm20, -256(%4, %0, 8)        \n\t" // S1
            "vpcmpeqd     %%zmm21, %%zmm25, %%k1                \n\t" // C2
            "vmulpd       %%zmm18, %%zmm0 , %%zmm26     \n\t" // F23
            "vmulpd       %%zmm19, %%zmm0 , %%zmm23     \n\t" // F14
            "vmovupd    0(%4, %0, 8), %%zmm16       \n\t" // LX5

            "vmovupd    %%zmm17, %%zmm29            \n\t" // B2
            "vmovupd    %%zmm21, -192(%4, %0, 8)        \n\t" // S2
            "vpcmpeqd     %%zmm22, %%zmm26, %%k2                \n\t" // C3
            "vmulpd       %%zmm19, %%zmm0 , %%zmm27     \n\t" // F24
            "vmulpd       %%zmm16, %%zmm0 , %%zmm20     \n\t" // F15
            "vmovupd   64(%4, %0, 8), %%zmm17       \n\t" // LX6

            "vmovupd    %%zmm18, %%zmm30            \n\t" // B3
            "kandw        %%k2   , %%k1   , %%k1                \n\t" // red - 1 - 2 - 1
            "vmovupd    %%zmm22, -128(%4, %0, 8)        \n\t" // S3
            "vpcmpeqd     %%zmm23, %%zmm27, %%k3                \n\t" // C4 
            "vmulpd       %%zmm16, %%zmm0 , %%zmm24     \n\t" // F25
            "vmulpd       %%zmm17, %%zmm0 , %%zmm21     \n\t" // F16
            "vmovupd  128(%4, %0, 8), %%zmm18       \n\t" // LX7

            "vmovupd    %%zmm19, %%zmm31            \n\t" // B4
            "kandw        %%k3   , %%k1   , %%k1                \n\t" // red - 1 - 3 - 1
            "vmovupd    %%zmm23, -64(%4, %0, 8)     \n\t" // S4
            "vpcmpeqd     %%zmm20, %%zmm24, %%k0                \n\t" // C5
            "vmulpd       %%zmm17, %%zmm0 , %%zmm25     \n\t" // F26
            "vmulpd       %%zmm18, %%zmm0 , %%zmm22     \n\t" // F17
            "kandw        %%k0   , %%k1   , %%k1                \n\t" // red - 1 - 0 - 1
            "vmovupd  192(%4, %0, 8), %%zmm19       \n\t" // LX8
            "kxorw        %%k1   , %%k4   , %%k1                \n\t"
            "addq       $32, %0             \n\t"
            "ktestw       %%k1   , %%k4                         \n\t"
            "jnz          4f           \n\t"
            "subq       $32, %1                 \n\t"
            "jnz        1b                      \n\t"


            /* Epilogue */

            "2: \n\t"
            "vmovupd    %%zmm16, %%zmm28            \n\t" // B1
            "vmovupd    %%zmm20, -256(%4, %0, 8)        \n\t" // S1
            "vpcmpeqd     %%zmm21, %%zmm25, %%k1                \n\t" // C2
            "vmulpd       %%zmm18, %%zmm0 , %%zmm26     \n\t" // F23
            "vmulpd       %%zmm19, %%zmm0 , %%zmm23     \n\t" // F14

            "vmovupd    %%zmm17, %%zmm29            \n\t" // B2
            "vmovupd    %%zmm21, -192(%4, %0, 8)        \n\t" // S2
            "vpcmpeqd     %%zmm22, %%zmm26, %%k2                \n\t" // C3
            "vmulpd       %%zmm19, %%zmm0 , %%zmm27     \n\t" // F24

            "kandw        %%k2   , %%k1   , %%k1                \n\t" // red - 1 - 2 - 1
            "vmovupd    %%zmm18, %%zmm30            \n\t" // B3
            "vmovupd    %%zmm22, -128(%4, %0, 8)        \n\t" // S3
            "vpcmpeqd     %%zmm23, %%zmm27, %%k3                \n\t" // C4 

            "vmovupd    %%zmm19, %%zmm31            \n\t" // B4
            "kandw        %%k3   , %%k1   , %%k1                \n\t" // red - 1 - 3 - 1
            "vmovupd    %%zmm23, -64(%4, %0, 8)     \n\t" // S4

            "kxorw        %%k1   , %%k4   , %%k1                \n\t"
            "ktestw       %%k1   , %%k4                         \n\t"
            "jnz          5f           \n\t"
            "jmp      3f                            \n\t"

            /* error handler to loop body */
            "4:                 \n\t"
			"addq       $1, %2             \n\t"
            "subq       $32, %0             \n\t"
            "vmovupd    %%zmm28, %%zmm16            \n\t" // restore from chkpt - 1

            "vmulpd       %%zmm16, %%zmm0 , %%zmm20     \n\t" // F11
            "vmovupd    %%zmm29, %%zmm17            \n\t" // restore from chkpt - 2


            "vmulpd       %%zmm16, %%zmm0 , %%zmm24     \n\t" // F21
            "vmulpd       %%zmm17, %%zmm0 , %%zmm21     \n\t" // F12
            "vmovupd    %%zmm30, %%zmm18            \n\t" // restore from chkpt - 3


            "vpcmpeqd     %%zmm20, %%zmm24, %%k0                \n\t" // C1
            "vmulpd       %%zmm17, %%zmm0 , %%zmm25     \n\t" // F22
            "vmulpd       %%zmm18, %%zmm0 , %%zmm22     \n\t" // F13
            "vmovupd    %%zmm31, %%zmm19            \n\t" // restore from chkpt - 4

            "vmovupd    %%zmm16, %%zmm28            \n\t" // B1
            "vmovupd    %%zmm20, -256(%4, %0, 8)        \n\t" // S1
            "vpcmpeqd     %%zmm21, %%zmm25, %%k1                \n\t" // C2
            "vmulpd       %%zmm18, %%zmm0 , %%zmm26     \n\t" // F23
            "vmulpd       %%zmm19, %%zmm0 , %%zmm23     \n\t" // F14
            "vmovupd    0(%4, %0, 8), %%zmm16       \n\t" // LX5

            "vmovupd    %%zmm17, %%zmm29            \n\t" // B2
            "vmovupd    %%zmm21, -192(%4, %0, 8)        \n\t" // S2
            "vpcmpeqd     %%zmm22, %%zmm26, %%k2                \n\t" // C3
            "vmulpd       %%zmm19, %%zmm0 , %%zmm27     \n\t" // F24
            "vmulpd       %%zmm16, %%zmm0 , %%zmm20     \n\t" // F15
            "vmovupd   64(%4, %0, 8), %%zmm17       \n\t" // LX6

            "vmovupd    %%zmm18, %%zmm30            \n\t" // B3
            "kandw        %%k2   , %%k1   , %%k1                \n\t" // red - 1 - 2 - 1
            "vmovupd    %%zmm22, -128(%4, %0, 8)        \n\t" // S3
            "vpcmpeqd     %%zmm23, %%zmm27, %%k3                \n\t" // C4 
            "vmulpd       %%zmm16, %%zmm0 , %%zmm24     \n\t" // F25
            "vmulpd       %%zmm17, %%zmm0 , %%zmm21     \n\t" // F16
            "vmovupd  128(%4, %0, 8), %%zmm18       \n\t" // LX7

            "vmovupd    %%zmm19, %%zmm31            \n\t" // B4
            "kandw        %%k3   , %%k1   , %%k1                \n\t" // red - 1 - 3 - 1
            "vmovupd    %%zmm23, -64(%4, %0, 8)     \n\t" // S4
            "vpcmpeqd     %%zmm20, %%zmm24, %%k0                \n\t" // C5
            "vmulpd       %%zmm17, %%zmm0 , %%zmm25     \n\t" // F26
            "vmulpd       %%zmm18, %%zmm0 , %%zmm22     \n\t" // F17
            "kandw        %%k0   , %%k1   , %%k1                \n\t" // red - 1 - 0 - 1
            "vmovupd  192(%4, %0, 8), %%zmm19       \n\t" // LX8
            "kxorw        %%k1   , %%k4   , %%k1                \n\t"
            "addq       $32, %0             \n\t"
            "ktestw       %%k1   , %%k4                         \n\t"
            "jnz          6f           \n\t"
            "subq       $32, %1                 \n\t"
            "jnz        1b                      \n\t"
            "jmp      2b                            \n\t"

			/* error handler to epilogue */
            "5: \n\t"
			"addq       $1, %2             \n\t"
            "vmovupd    %%zmm28, %%zmm16            \n\t" // restore from chkpt - 1

            "vmulpd       %%zmm16, %%zmm0 , %%zmm20     \n\t" // F11
            "vmovupd    %%zmm29, %%zmm17            \n\t" // restore from chkpt - 2


            "vmulpd       %%zmm16, %%zmm0 , %%zmm24     \n\t" // F21
            "vmulpd       %%zmm17, %%zmm0 , %%zmm21     \n\t" // F12
            "vmovupd    %%zmm30, %%zmm18            \n\t" // restore from chkpt - 3


            "vpcmpeqd     %%zmm20, %%zmm24, %%k0                \n\t" // C1
            "vmulpd       %%zmm17, %%zmm0 , %%zmm25     \n\t" // F22
            "vmulpd       %%zmm18, %%zmm0 , %%zmm22     \n\t" // F13
            "vmovupd    %%zmm31, %%zmm19            \n\t" // restore from chkpt - 4

            "vmovupd    %%zmm16, %%zmm28            \n\t" // B1
            "vmovupd    %%zmm20, -256(%4, %0, 8)        \n\t" // S1
            "vpcmpeqd     %%zmm21, %%zmm25, %%k1                \n\t" // C2
            "vmulpd       %%zmm18, %%zmm0 , %%zmm26     \n\t" // F23
            "vmulpd       %%zmm19, %%zmm0 , %%zmm23     \n\t" // F14

            "vmovupd    %%zmm17, %%zmm29            \n\t" // B2
            "vmovupd    %%zmm21, -192(%4, %0, 8)        \n\t" // S2
            "vpcmpeqd     %%zmm22, %%zmm26, %%k2                \n\t" // C3
            "vmulpd       %%zmm19, %%zmm0 , %%zmm27     \n\t" // F24

            "kandw        %%k2   , %%k1   , %%k1                \n\t" // red - 1 - 2 - 1
            "vmovupd    %%zmm18, %%zmm30            \n\t" // B3
            "vmovupd    %%zmm22, -128(%4, %0, 8)        \n\t" // S3
            "vpcmpeqd     %%zmm23, %%zmm27, %%k3                \n\t" // C4 

            "vmovupd    %%zmm19, %%zmm31            \n\t" // B4
            "kandw        %%k3   , %%k1   , %%k1                \n\t" // red - 1 - 3 - 1
            "vmovupd    %%zmm23, -64(%4, %0, 8)     \n\t" // S4

            "kxorw        %%k1   , %%k4   , %%k1                \n\t"
            "ktestw       %%k1   , %%k4                         \n\t"
            "jnz          6f           \n\t"
            "jmp      3f                            \n\t"

            "6: \n\t"
			"addq       $1, %2             \n\t"

            "3:                 \n\t"
            "vzeroupper                 \n\t"

            : "+r"(i),      // 0
              "+r"(n),      // 1
              "+r"(err_num) // 2
            : "r"(alpha),   // 3
              "r"(x)        // 4
            : "cc",
                "%xmm0", "%xmm1", "%xmm2", "%xmm3",
//              "%xmm4", "%xmm5", "%xmm6", "%xmm7",
//              "%xmm8", "%xmm9", "%xmm10", "%xmm11",
//              "%xmm12", "%xmm13", "%xmm14", "%xmm15",
                "memory");
	return err_num;
}

void ft_dscal_compute(long int n, double da, double *x, long int inc_x)
{
    long int i = 0;
    if (inc_x == 1)
    {
        if (da == 0.0)
        {
            for (i = 0; i < n; i++)
            {
                x[i] = 0.;
            }
        }
        else
        {
            long int n1 = n & -32;
            if (n1 > 0)
            {
                long int err_num = ft_dscal_kernel(n1, &da, x);
                if (err_num) printf("detected and corrected error number %ld\n", err_num);

                i = n1;
            }

            while (i < n)
            {
                x[i] *= da;
                i++;
            }
        }

    }
    else
    {
        for (i = 0; i < n; i += inc_x)
        {
            x[i] *= da;
        }
    }
    return;
}

void ftblas_dscal_ft(const long int n, const double alpha, const double *x, const long int inc_x)
{
  int TOTAL_THREADS=atoi(getenv("OMP_NUM_THREADS"));
  if (TOTAL_THREADS<=1 || n < 4000000)
  {
    ft_dscal_compute(n,alpha,(double *)x,inc_x);
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
      ft_dscal_compute(EDGE_LEN,alpha,(double *)x,inc_x);
    else
      ft_dscal_compute(DIM_LEN,alpha,(double *)(x + EDGE_LEN + (tid - 1) * DIM_LEN), inc_x);
  }
  return;
}