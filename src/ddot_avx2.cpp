#include "../include/ftblas.h"
#include<unistd.h>

static long ft_ddot_kernel(long n, double *x, double *y, double *dot)
{

  long register i = 0;
  long register err_num = 0;
  __asm__ __volatile__(

      "vxorpd		%%zmm0 , %%zmm0 , %%zmm0	             \n\t"
      "vxorpd		%%zmm1 , %%zmm1 , %%zmm1	             \n\t"
      "vxorpd		%%zmm2 , %%zmm2 , %%zmm2	             \n\t"
      "vxorpd		%%zmm3 , %%zmm3 , %%zmm3	             \n\t"

      "vxorpd		%%zmm4 , %%zmm4 , %%zmm4	             \n\t"
      "vxorpd		%%zmm5 , %%zmm5 , %%zmm5	             \n\t"
      "vxorpd		%%zmm6 , %%zmm6 , %%zmm6	             \n\t"
      "vxorpd		%%zmm7 , %%zmm7 , %%zmm7	             \n\t"

      "kxnorw   %%k4   , %%k4   , %%k4                 \n\t"

      /* Prologue */
      "vmovupd       (%3, %0, 8), %%zmm16               \n\t" // X1
      "vmovupd       (%4, %0, 8), %%zmm24               \n\t" // Y1
      "vmovupd     64(%3, %0, 8), %%zmm17               \n\t" // X2
      "vmovupd           %%zmm4 , %%zmm28               \n\t" // B1
      "vmovupd     64(%4, %0, 8), %%zmm25               \n\t" // Y2
      "vmovupd    128(%3, %0, 8), %%zmm18               \n\t" // X3
      "vfmadd231pd  %%zmm16, %%zmm24, %%zmm0            \n\t" // F1
      "vfmadd231pd  %%zmm16, %%zmm24, %%zmm4            \n\t" // F1 - dup - 0 - 4
      "vmovupd           %%zmm5 , %%zmm29               \n\t" // B2
      "vmovupd    128(%4, %0, 8), %%zmm26               \n\t" // Y3
      "vmovupd    192(%3, %0, 8), %%zmm19               \n\t" // X4

      "addq		$32 , %0	  	     \n\t"
      "subq	  $32 , %1  		     \n\t"
      "jz 		3f		             \n\t"

      ".p2align 4				             \n\t"
      "1:				             \n\t"
      "prefetcht0          1024(%3, %0, 8)                \n\t" // prefetch
      "vpcmpeqd     %%zmm0 , %%zmm4 , %%k0                \n\t" // C1
      "vmovupd           %%zmm28, %%zmm12                 \n\t" // chkpt - 28 - 12
      "prefetcht0          1024(%4, %0, 8)                \n\t" // prefetch
      "vfmadd231pd  %%zmm17, %%zmm25, %%zmm1              \n\t" // F2
      "vfmadd231pd  %%zmm17, %%zmm25, %%zmm5              \n\t" // F2 - dup - 1 - 5
      "vmovupd           %%zmm6 , %%zmm30                 \n\t" // B3
      "vmovupd    -64(%4, %0, 8), %%zmm27                 \n\t" // Y4
      "vmovupd       (%3, %0, 8), %%zmm16                 \n\t" // X5

      "vpcmpeqd     %%zmm1 , %%zmm5 , %%k1                \n\t" // C2
      "vmovupd           %%zmm29, %%zmm13                 \n\t" // chkpt - 29 - 13
      "vfmadd231pd  %%zmm18, %%zmm26, %%zmm2              \n\t" // F3
      "vfmadd231pd  %%zmm18, %%zmm26, %%zmm6              \n\t" // F3 - dup - 2 - 6

      "kandw          %%k0   , %%k1   , %%k0              \n\t" // check - red - 0 - 1

      "vmovupd           %%zmm7 , %%zmm31                 \n\t" // B4
      "vmovupd       (%4, %0, 8), %%zmm24                 \n\t" // Y5
      "vmovupd     64(%3, %0, 8), %%zmm17                 \n\t" // X6

      "vpcmpeqd     %%zmm2 , %%zmm6 , %%k2                \n\t" // C3
      "vmovupd           %%zmm30, %%zmm14                 \n\t" // chkpt - 30 - 14
      "vfmadd231pd  %%zmm19, %%zmm27, %%zmm3              \n\t" // F4
      "vfmadd231pd  %%zmm19, %%zmm27, %%zmm7              \n\t" // F4 - dup - 3 - 7

      "prefetcht0          1152(%3, %0, 8)                \n\t" // prefetch

      "vmovupd           %%zmm4 , %%zmm28                 \n\t" // B5
      "vmovupd     64(%4, %0, 8), %%zmm25                 \n\t" // Y6
      "vmovupd    128(%3, %0, 8), %%zmm18                 \n\t" // X7

      "vpcmpeqd     %%zmm3 , %%zmm7 , %%k3                \n\t" // C4
      "vmovupd           %%zmm31, %%zmm15                 \n\t" // chkpt - 31 - 15
      "vfmadd231pd  %%zmm16, %%zmm24, %%zmm0              \n\t" // F5
      "vfmadd231pd  %%zmm16, %%zmm24, %%zmm4              \n\t" // F5 - dup - 0 - 4
      
      "prefetcht0          1152(%4, %0, 8)                \n\t" // prefetch

      "kandw          %%k2   , %%k3   , %%k2              \n\t" // check - red - 2 - 3

      "vmovupd           %%zmm5 , %%zmm29                 \n\t" // B6

      "kandw          %%k0   , %%k2   , %%k0              \n\t"

      "vmovupd    128(%4, %0, 8), %%zmm26                 \n\t" // Y7
      "vmovupd    192(%3, %0, 8), %%zmm19                 \n\t" // X8
      
      "kxorw          %%k0   , %%k4   , %%k0              \n\t"

      "addq		$32 , %0	  	     \n\t"

      "ktestw         %%k0   , %%k4                       \n\t"
      "jnz		2f		             \n\t"
      "subq	    $32 , %1		     \n\t"
      "jnz		1b		             \n\t"

      "3:   \n\t"
      /* Epilogue */
      "vpcmpeqd     %%zmm0 , %%zmm4 , %%k0                \n\t" // C1
      "vmovupd           %%zmm28, %%zmm12                 \n\t" // chkpt - 28 - 12
      "vfmadd231pd  %%zmm17, %%zmm25, %%zmm1              \n\t" // F2
      "vfmadd231pd  %%zmm17, %%zmm25, %%zmm5              \n\t" // F2 - dup - 1 - 5
      "vmovupd           %%zmm6 , %%zmm30                 \n\t" // B3
      "vmovupd    -64(%4, %0, 8), %%zmm27                 \n\t" // Y4

      "vpcmpeqd     %%zmm1 , %%zmm5 , %%k1                \n\t" // C2
      "vmovupd           %%zmm29, %%zmm13                 \n\t" // chkpt - 29 - 13
      "vfmadd231pd  %%zmm18, %%zmm26, %%zmm2              \n\t" // F3
      "vfmadd231pd  %%zmm18, %%zmm26, %%zmm6              \n\t" // F3 - dup - 2 - 6

      "kandw          %%k0   , %%k1   , %%k0              \n\t" // check - red - 0 - 1

      "vmovupd           %%zmm7 , %%zmm31                 \n\t" // B4

      "vpcmpeqd     %%zmm2 , %%zmm6 , %%k2                \n\t" // C3
      "vmovupd           %%zmm30, %%zmm14                 \n\t" // chkpt - 30 - 14
      "vfmadd231pd  %%zmm19, %%zmm27, %%zmm3              \n\t" // F4
      "vfmadd231pd  %%zmm19, %%zmm27, %%zmm7              \n\t" // F4 - dup - 3 - 7
      

      "vpcmpeqd     %%zmm3 , %%zmm7 , %%k3                \n\t" // C4
      "vmovupd           %%zmm31, %%zmm15                 \n\t" // chkpt - 31 - 15

      "kandw          %%k2   , %%k3   , %%k2              \n\t" // check - red - 2 - 3
      "kandw          %%k0   , %%k2   , %%k0              \n\t"
      "kxorw          %%k0   , %%k4   , %%k0              \n\t"
      "ktestw         %%k0   , %%k4                       \n\t"
      "jnz            6f                                  \n\t"

      "5:                                                 \n\t"
      "vaddpd       %%zmm0 , %%zmm1 , %%zmm0              \n\t"
      "vaddpd       %%zmm2 , %%zmm3 , %%zmm2              \n\t"

      "vaddpd       %%zmm0 , %%zmm2 , %%zmm0              \n\t"

      "vextractf64x4     $0, %%zmm0 , %%ymm1              \n\t"
      "vextractf64x4     $1, %%zmm0 , %%ymm2              \n\t"
      "vaddpd       %%ymm1 , %%ymm2 , %%ymm1              \n\t"

      "vextractf128	    $1 , %%ymm1 , %%xmm2	            \n\t"
      "vaddpd       %%xmm1 , %%xmm2 , %%xmm1	            \n\t"
      "vhaddpd      %%xmm1 , %%xmm1 , %%xmm1	            \n\t"

      "vmovsd		%%xmm1 ,    (%5)		\n\t"
      "jmp 4f     \n\t"

      /* Error Handler to Loop Body */
      "2:                                               \n\t"
      "addq		$1 , %2	  	     \n\t"
      "subq	    $32 , %0		     \n\t"
      "vmovupd   -256(%3, %0, 8), %%zmm16               \n\t" // X1
      "vmovupd   -256(%4, %0, 8), %%zmm24               \n\t" // Y1
      "vmovupd   -192(%3, %0, 8), %%zmm17               \n\t" // X2
      "vmovupd           %%zmm12, %%zmm0                \n\t" // restore from chkpt - 12 - 0
      "vmovupd           %%zmm12, %%zmm4                \n\t" // restore from chkpt - 12 - 4
      "vmovupd   -192(%4, %0, 8), %%zmm25               \n\t" // Y2
      "vmovupd   -128(%3, %0, 8), %%zmm18               \n\t" // X3
      "vfmadd231pd  %%zmm16, %%zmm24, %%zmm0            \n\t" // F1
      "vfmadd231pd  %%zmm16, %%zmm24, %%zmm4            \n\t" // F1 - dup - 0 - 4
      "vmovupd           %%zmm13, %%zmm1                \n\t" // restore from chkpt - 13 - 1
      "vmovupd           %%zmm13, %%zmm5                \n\t" // restore from chkpt - 13 - 5
      "vmovupd   -128(%4, %0, 8), %%zmm26               \n\t" // Y3
      "vmovupd    -64(%3, %0, 8), %%zmm19               \n\t" // X4

      "vpcmpeqd     %%zmm0 , %%zmm4 , %%k0                \n\t" // C1
      "vfmadd231pd  %%zmm17, %%zmm25, %%zmm1              \n\t" // F2
      "vfmadd231pd  %%zmm17, %%zmm25, %%zmm5              \n\t" // F2 - dup - 1 - 5
      "vmovupd           %%zmm14, %%zmm2                  \n\t" // restore from chkpt - 14 - 2
      "vmovupd           %%zmm14, %%zmm6                  \n\t" // restore from chkpt - 14 - 6
      "vmovupd    -64(%4, %0, 8), %%zmm27                 \n\t" // Y4
      "vmovupd       (%3, %0, 8), %%zmm16                 \n\t" // X5

      "vpcmpeqd     %%zmm1 , %%zmm5 , %%k1                \n\t" // C2
      "vfmadd231pd  %%zmm18, %%zmm26, %%zmm2              \n\t" // F3
      "vfmadd231pd  %%zmm18, %%zmm26, %%zmm6              \n\t" // F3 - dup - 2 - 6

      "kandw          %%k0   , %%k1   , %%k0              \n\t" // check - red - 0 - 1

      "vmovupd           %%zmm15, %%zmm3                  \n\t" // restore from chkpt - 15 - 3
      "vmovupd           %%zmm15, %%zmm7                  \n\t" // restore from chkpt - 15 - 7
      "vmovupd       (%4, %0, 8), %%zmm24                 \n\t" // Y5
      "vmovupd     64(%3, %0, 8), %%zmm17                 \n\t" // X6

      "vpcmpeqd     %%zmm2 , %%zmm6 , %%k2                \n\t" // C3
      "vfmadd231pd  %%zmm19, %%zmm27, %%zmm3              \n\t" // F4
      "vfmadd231pd  %%zmm19, %%zmm27, %%zmm7              \n\t" // F4 - dup - 3 - 7
      "vmovupd           %%zmm4 , %%zmm28                 \n\t" // B5
      "vmovupd     64(%4, %0, 8), %%zmm25                 \n\t" // Y6
      "vmovupd    128(%3, %0, 8), %%zmm18                 \n\t" // X7

      "vpcmpeqd     %%zmm3 , %%zmm7 , %%k3                \n\t" // C4
      "vfmadd231pd  %%zmm16, %%zmm24, %%zmm0              \n\t" // F5
      "vfmadd231pd  %%zmm16, %%zmm24, %%zmm4              \n\t" // F5 - dup - 0 - 4

      "kandw          %%k2   , %%k3   , %%k2              \n\t" // check - red - 2 - 3

      "vmovupd           %%zmm5 , %%zmm29                 \n\t" // B6
      "vmovupd    128(%4, %0, 8), %%zmm26                 \n\t" // Y7
      "vmovupd    192(%3, %0, 8), %%zmm19                 \n\t" // X8

      
      "kandw          %%k0   , %%k2   , %%k0              \n\t"
      "kxorw          %%k0   , %%k4   , %%k0              \n\t"
      "ktestw         %%k0   , %%k4                       \n\t"
      "jnz            4f                                  \n\t"

      "addq		$32 , %0	  	     \n\t"
      "subq	    $32 , %1		     \n\t"
      "jnz		1b		             \n\t"

      "jmp 3b                                             \n\t"

      /* Error Handler to Epilogue */
      "6:                                                 \n\t"
      /* Restore From Prologue */
      "addq		$1 , %2	  	     \n\t"
      "vmovupd   -256(%3, %0, 8), %%zmm16               \n\t" // X1
      "vmovupd   -256(%4, %0, 8), %%zmm24               \n\t" // Y1
      "vmovupd   -192(%3, %0, 8), %%zmm17               \n\t" // X2
      "vmovupd           %%zmm12, %%zmm0                \n\t" // restore from chkpt - 12 - 0
      "vmovupd           %%zmm12, %%zmm4                \n\t" // restore from chkpt - 12 - 4
      "vmovupd   -192(%4, %0, 8), %%zmm25               \n\t" // Y2
      "vmovupd   -128(%3, %0, 8), %%zmm18               \n\t" // X3
      "vfmadd231pd  %%zmm16, %%zmm24, %%zmm0            \n\t" // F1
      "vfmadd231pd  %%zmm16, %%zmm24, %%zmm4            \n\t" // F1 - dup - 0 - 4
      "vmovupd           %%zmm13, %%zmm1                \n\t" // restore from chkpt - 13 - 1
      "vmovupd           %%zmm13, %%zmm5                \n\t" // restore from chkpt - 13 - 5
      "vmovupd   -128(%4, %0, 8), %%zmm26               \n\t" // Y3
      "vmovupd    -64(%3, %0, 8), %%zmm19               \n\t" // X4

      "vpcmpeqd     %%zmm0 , %%zmm4 , %%k0                \n\t" // C1
      "vfmadd231pd  %%zmm17, %%zmm25, %%zmm1              \n\t" // F2
      "vfmadd231pd  %%zmm17, %%zmm25, %%zmm5              \n\t" // F2 - dup - 1 - 5
      "vmovupd           %%zmm14, %%zmm2                  \n\t" // restore from chkpt - 14 - 2
      "vmovupd           %%zmm14, %%zmm6                  \n\t" // restore from chkpt - 14 - 6
      "vmovupd    -64(%4, %0, 8), %%zmm27                 \n\t" // Y4

      "vpcmpeqd     %%zmm1 , %%zmm5 , %%k1                \n\t" // C2
      "vfmadd231pd  %%zmm18, %%zmm26, %%zmm2              \n\t" // F3
      "vfmadd231pd  %%zmm18, %%zmm26, %%zmm6              \n\t" // F3 - dup - 2 - 6

      "kandw          %%k0   , %%k1   , %%k0              \n\t" // check - red - 0 - 1

      "vmovupd           %%zmm15, %%zmm3                  \n\t" // restore from chkpt - 15 - 3
      "vmovupd           %%zmm15, %%zmm7                  \n\t" // restore from chkpt - 15 - 7

      "vpcmpeqd     %%zmm2 , %%zmm6 , %%k2                \n\t" // C3
      "vfmadd231pd  %%zmm19, %%zmm27, %%zmm3              \n\t" // F4
      "vfmadd231pd  %%zmm19, %%zmm27, %%zmm7              \n\t" // F4 - dup - 3 - 7
      
      "vpcmpeqd     %%zmm3 , %%zmm7 , %%k3                \n\t" // C4

      "kandw          %%k2   , %%k3   , %%k2              \n\t" // check - red - 2 - 3
      "kandw          %%k0   , %%k2   , %%k0              \n\t"
      "kxorw          %%k0   , %%k4   , %%k0              \n\t"
      "ktestw         %%k0   , %%k4                       \n\t"
      "jnz            4f                                  \n\t"

      "jmp 5b                                             \n\t"

      "7: \n\t" // if still incorrect
      "addq		$1 , %2	  	     \n\t"

      "4: \n\t" // exit
      "vzeroupper				\n\t"
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

static void ori_ddot_kernel(long n, double *x, double *y, double *dot)
{
  long register i = 0;

  __asm__ __volatile__(

      "vxorpd		%%zmm0 , %%zmm0 , %%zmm0	             \n\t"
      "vxorpd		%%zmm1 , %%zmm1 , %%zmm1	             \n\t"
      "vxorpd		%%zmm2 , %%zmm2 , %%zmm2	             \n\t"
      "vxorpd		%%zmm3 , %%zmm3 , %%zmm3	             \n\t"

      ".p2align 4				             \n\t"
      "1:				             \n\t"
	    "prefetcht0          1024(%3, %0, 8)              \n\t"
      "vmovupd       (%3, %0, 8), %%zmm24               \n\t"
      "vmovupd     64(%3, %0, 8), %%zmm25               \n\t"
	    "prefetcht0          1152(%3, %0, 8)              \n\t"
      "vmovupd    128(%3, %0, 8), %%zmm26               \n\t"
      "vmovupd    192(%3, %0, 8), %%zmm27               \n\t"
	    "prefetcht0          1024(%2, %0, 8)              \n\t"
      "vfmadd231pd     (%2, %0, 8), %%zmm24, %%zmm0     \n\t"
      "vfmadd231pd   64(%2, %0, 8), %%zmm25, %%zmm1     \n\t"
	    "prefetcht0          1152(%2, %0, 8)              \n\t"
      "vfmadd231pd  128(%2, %0, 8), %%zmm26, %%zmm2     \n\t"
      "vfmadd231pd  192(%2, %0, 8), %%zmm27, %%zmm3     \n\t"
      "addq		$32 , %0	  	     \n\t"
      "subq	    $32 , %1		     \n\t"
      "jnz		1b		             \n\t"

      "3:   \n\t"
      "vaddpd       %%zmm0 , %%zmm1 , %%zmm0              \n\t"
      "vaddpd       %%zmm2 , %%zmm3 , %%zmm2              \n\t"

      "vaddpd       %%zmm0 , %%zmm2 , %%zmm0              \n\t"

      "vextractf64x4     $0, %%zmm0 , %%ymm1              \n\t"
      "vextractf64x4     $1, %%zmm0 , %%ymm2              \n\t"
      "vaddpd       %%ymm1 , %%ymm2 , %%ymm1              \n\t"

      "vextractf128	    $1 , %%ymm1 , %%xmm2	            \n\t"
      "vaddpd       %%xmm1 , %%xmm2 , %%xmm1	            \n\t"
      "vhaddpd      %%xmm1 , %%xmm1 , %%xmm1	            \n\t"

      "vmovsd		%%xmm1 ,    (%4)		\n\t"
      "vzeroupper				\n\t"

      : "+r"(i), // 0
        "+r"(n)  // 1
      : "r"(x),  // 2
        "r"(y),  // 3
        "r"(dot) // 4
      : "cc",
        "%xmm0", "%xmm1", "%xmm2", "%xmm3",
        "%xmm4", "%xmm5", "%xmm6", "%xmm7",
        "%xmm8", "%xmm9", "%xmm10", "%xmm11",
        "%xmm12", "%xmm13", "%xmm14", "%xmm15",
        "memory");
}

double my_ft_dot_compute(long int n, double *x, long int inc_x, double *y, long int inc_y)
{
	long int i=0;
	long int ix=0,iy=0;
	double  dot = 0.0 ;
	
	if ( n <= 0 )  return(dot);

	if ( (inc_x == 1) && (inc_y == 1) )
	{
		long int n1 = n & -32;
		if ( n1 )
		{
			long int err_num = ft_ddot_kernel(n1, x, y , &dot );
			if (err_num) printf("detected error number: %ld\n", err_num);
		}
		i = n1;
		while(i < n)
		{
			dot += y[i] * x[i] ;
			i++ ;
		}
		return(dot);
	}

	double temp1 = 0.0;
	double temp2 = 0.0;
    long int n1 = n & -4;

	while(i < n1)
	{
		double m1 = y[iy]       * x[ix] ;
		double m2 = y[iy+inc_y] * x[ix+inc_x] ;
		double m3 = y[iy+2*inc_y] * x[ix+2*inc_x] ;
		double m4 = y[iy+3*inc_y] * x[ix+3*inc_x] ;
		ix  += inc_x*4 ;
		iy  += inc_y*4 ;
		temp1 += m1+m3;
		temp2 += m2+m4;
		i+=4 ;
	}

	while(i < n)
	{
		temp1 += y[iy] * x[ix] ;
		ix  += inc_x ;
		iy  += inc_y ;
		i++ ;
	}
	dot = temp1 + temp2;
	return(dot);
}

double my_ori_dot_compute(long int n, double *x, long int inc_x, double *y, long int inc_y)
{
	long int i=0;
	long int ix=0,iy=0;
	double  dot = 0.0 ;
	if ( n <= 0 )  return(dot);
	if ( (inc_x == 1) && (inc_y == 1) )
	{
		long int n1 = n & -32;
		if ( n1 )
			ori_ddot_kernel(n1, x, y , &dot );
		i = n1;
		while(i < n)
		{
			dot += y[i] * x[i] ;
			i++ ;
		}
		return(dot);
	}

	double temp1 = 0.0;
	double temp2 = 0.0;
    long int n1 = n & -4;

	while(i < n1)
	{

		double m1 = y[iy]       * x[ix] ;
		double m2 = y[iy+inc_y] * x[ix+inc_x] ;
		double m3 = y[iy+2*inc_y] * x[ix+2*inc_x] ;
		double m4 = y[iy+3*inc_y] * x[ix+3*inc_x] ;
		ix  += inc_x*4 ;
		iy  += inc_y*4 ;
		temp1 += m1+m3;
		temp2 += m2+m4;
		i+=4 ;
	}

	while(i < n)
	{
		temp1 += y[iy] * x[ix] ;
		ix  += inc_x ;
		iy  += inc_y ;
		i++ ;
	}
	dot = temp1 + temp2;
	return dot;

}

double ft_ddot(long int n, double *x, long int inc_x, double *y, long int inc_y, bool is_ft)
{
    double res=0.;
    int tid,TOTAL_THREADS=atoi(getenv("OMP_NUM_THREADS"));
	if (TOTAL_THREADS<=1)
	{
		res+=my_ft_dot_compute(n,x,inc_x,y,inc_y);
		return res;
	}
	int max_cpu_num=(int)sysconf(_SC_NPROCESSORS_ONLN);
	if (TOTAL_THREADS>max_cpu_num) TOTAL_THREADS=max_cpu_num;
    #pragma omp parallel for schedule(static) reduction(+:res)
    for(tid=0;tid<TOTAL_THREADS;tid++)
    {
        long int NUM_DIV_NUM_THREADS=n/TOTAL_THREADS*TOTAL_THREADS;
        long int DIM_LEN=n/TOTAL_THREADS;
        long int EDGE_LEN = (NUM_DIV_NUM_THREADS == n)?n/TOTAL_THREADS:n-NUM_DIV_NUM_THREADS+DIM_LEN;
        if (tid==0) res+=my_ft_dot_compute(EDGE_LEN,x,inc_x,y,inc_y);
        else res+=my_ft_dot_compute(DIM_LEN,x+EDGE_LEN+(tid-1)*DIM_LEN,inc_x,y+EDGE_LEN+(tid-1)*DIM_LEN,inc_y);
	}
	return res;
}

double ori_ddot(long int n, double *x, long int inc_x, double *y, long int inc_y, bool is_ft)
{
    double res=0.;
    int tid,TOTAL_THREADS=atoi(getenv("OMP_NUM_THREADS"));
	if (TOTAL_THREADS<=1)
	{
		res+=my_ori_dot_compute(n,x,inc_x,y,inc_y);
		return res;
	}
	int max_cpu_num=(int)sysconf(_SC_NPROCESSORS_ONLN);
	if (TOTAL_THREADS>max_cpu_num) TOTAL_THREADS=max_cpu_num;
    #pragma omp parallel for schedule(static) reduction(+:res)
    for(tid=0;tid<TOTAL_THREADS;tid++)
    {
        long int NUM_DIV_NUM_THREADS=n/TOTAL_THREADS*TOTAL_THREADS;
        long int DIM_LEN=n/TOTAL_THREADS;
        long int EDGE_LEN = (NUM_DIV_NUM_THREADS == n)?n/TOTAL_THREADS:n-NUM_DIV_NUM_THREADS+DIM_LEN;
        if (tid==0) res+=my_ori_dot_compute(EDGE_LEN,x,inc_x,y,inc_y);
        else res+=my_ori_dot_compute(DIM_LEN,x+EDGE_LEN+(tid-1)*DIM_LEN,inc_x,y+EDGE_LEN+(tid-1)*DIM_LEN,inc_y);
	}
	return res;
}

double ftblas_ddot(const int n, const double *x, const int inc_x, const double *y, const int inc_y, bool is_ft)
{
  double res;
  if(is_ft == false)
  {
    res = ori_ddot((long int)n, (double *)x, (long int)inc_x, (double *)y, (long int)inc_y, is_ft);
    return res;
  }
  else
  {
    res = ft_ddot((long int)n, (double *)x, (long int)inc_x, (double *)y, (long int)inc_y, is_ft);
    return res;
  }
}