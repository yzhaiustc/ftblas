#include <stdio.h>
#include <stdlib.h>
#include "../include/ftblas.h"

static void ori_drotm_zero_kernel(long n, double *x, double *y, double *h12, double *h21)
{

	long register i = 0;

	__asm__ __volatile__(
		"vbroadcastsd		(%2), %%zmm0	\n\t" // h12
		"vbroadcastsd		(%3), %%zmm1	\n\t" // h21

		".p2align 4				             \n\t"
		"1:				             \n\t"
		"vmovupd                  (%4,%0,8), %%zmm16         \n\t" // 8 * x1
		"vmovupd                64(%4,%0,8), %%zmm17         \n\t" // 8 * x2
		"vmovupd               128(%4,%0,8), %%zmm18         \n\t" // 8 * x3
		"vmovupd               192(%4,%0,8), %%zmm19         \n\t" // 8 * x4

		"vmovupd                  (%5,%0,8), %%zmm20         \n\t" // 8 * y1
		"vmovupd                64(%5,%0,8), %%zmm21         \n\t" // 8 * y2
		"vmovupd               128(%5,%0,8), %%zmm22         \n\t" // 8 * y3
		"vmovupd               192(%5,%0,8), %%zmm23         \n\t" // 8 * y4

		"vmovupd 	 				%%zmm16, %%zmm24		 \n\t"
		"vmovupd 	 				%%zmm17, %%zmm25		 \n\t"
		"vmovupd 	 				%%zmm18, %%zmm26		 \n\t"
		"vmovupd 	 				%%zmm19, %%zmm27		 \n\t"

		"vmovupd 	 				%%zmm20, %%zmm28		 \n\t"
		"vmovupd 	 				%%zmm21, %%zmm29		 \n\t"
		"vmovupd 	 				%%zmm22, %%zmm30		 \n\t"
		"vmovupd 	 				%%zmm23, %%zmm31		 \n\t"

		"vfmadd231pd %%zmm0 , %%zmm28,  %%zmm16 \n\t" // x1 = x1 + %0 * y1
		"vfmadd231pd %%zmm0 , %%zmm29,  %%zmm17 \n\t" // x2 = x2 + %0 * y2
		"vfmadd231pd %%zmm0 , %%zmm30,  %%zmm18 \n\t" // x3 = x3 + %0 * y3
		"vfmadd231pd %%zmm0 , %%zmm31,  %%zmm19 \n\t" // x4 = x4 + %0 * y4

		"vfmadd231pd %%zmm1 , %%zmm24,  %%zmm20 \n\t" // y1 = %1 * x1 + y1
		"vfmadd231pd %%zmm1 , %%zmm25,  %%zmm21 \n\t" // y2 = %1 * x2 + y2
		"vfmadd231pd %%zmm1 , %%zmm26,  %%zmm22 \n\t" // y3 = %1 * x3 + y3
		"vfmadd231pd %%zmm1 , %%zmm27,  %%zmm23 \n\t" // y4 = %1 * x4 + y4

		"vmovupd                  %%zmm16,    (%4,%0,8)       \n\t" // 8 * x1
		"vmovupd                  %%zmm17,  64(%4,%0,8)       \n\t" // 8 * x2
		"vmovupd                  %%zmm18, 128(%4,%0,8)       \n\t" // 8 * x3
		"vmovupd                  %%zmm19, 192(%4,%0,8)       \n\t" // 8 * x4

		"vmovupd                  %%zmm20,    (%5,%0,8)       \n\t" // 8 * x1
		"vmovupd                  %%zmm21,  64(%5,%0,8)       \n\t" // 8 * x2
		"vmovupd                  %%zmm22, 128(%5,%0,8)       \n\t" // 8 * x3
		"vmovupd                  %%zmm23, 192(%5,%0,8)       \n\t" // 8 * x4

		"addq		$32 , %0	  	     \n\t"
		"subq	    $32 , %1		     \n\t"
		"jnz		1b		             \n\t"

		"vzeroupper				\n\t"
		"ret				         \n\t"

		: "+r"(i),   // 0
		  "+r"(n),   // 1
		  "+r"(h12), // 2
		  "+r"(h21)  // 3
		: "r"(x),    // 4
		  "r"(y)     // 5
		: "cc",
		  "%xmm0", "%xmm1", "%xmm2", "%xmm3",
//		  "%xmm4", "%xmm5", "%xmm6", "%xmm7",
//		  "%xmm8", "%xmm9", "%xmm10", "%xmm11",
//		  "%xmm12", "%xmm13", "%xmm14", "%xmm15",
		  "memory");
}

static void ori_drotm_kernel(long n, double *x, double *y, double *h11, double *h12, double *h21, double *h22)
{

	long register i = 0;

	__asm__ __volatile__(
		"vbroadcastsd		(%2), %%zmm0	\n\t" // h11
		"vbroadcastsd		(%3), %%zmm1	\n\t" // h12
		"vbroadcastsd		(%4), %%zmm2	\n\t" // h21
		"vbroadcastsd		(%5), %%zmm3	\n\t" // h22

		".p2align 4				             \n\t"
		"1:				             \n\t"
		"vmovupd                  (%7,%0,8), %%zmm20         \n\t" // 8 * y1
		"vmovupd                64(%7,%0,8), %%zmm21         \n\t" // 8 * y2
		"vmovupd               128(%7,%0,8), %%zmm22         \n\t" // 8 * y3
		"vmovupd               192(%7,%0,8), %%zmm23         \n\t" // 8 * y4

		"vmovupd                  (%6,%0,8), %%zmm16         \n\t" // 8 * x1
		"vmovupd                64(%6,%0,8), %%zmm17         \n\t" // 8 * x2
		"vmovupd               128(%6,%0,8), %%zmm18         \n\t" // 8 * x3
		"vmovupd               192(%6,%0,8), %%zmm19         \n\t" // 8 * x4

		"vmulpd      %%zmm1 , %%zmm20, %%zmm24  \n\t" // %1 * y1
		"vmulpd      %%zmm1 , %%zmm21, %%zmm25  \n\t" // %1 * y2
		"vmulpd      %%zmm1 , %%zmm22, %%zmm26  \n\t" // %1 * y3
		"vmulpd      %%zmm1 , %%zmm23, %%zmm27  \n\t" // %1 * y4

		"vmulpd      %%zmm3 , %%zmm20, %%zmm28  \n\t" // %3 * y1
		"vmulpd      %%zmm3 , %%zmm21, %%zmm29  \n\t" // %3 * y2
		"vmulpd      %%zmm3 , %%zmm22, %%zmm30  \n\t" // %3 * y3
		"vmulpd      %%zmm3 , %%zmm23, %%zmm31  \n\t" // %3 * y4

		"vfmadd231pd %%zmm16, %%zmm0 ,  %%zmm24 \n\t" // x1 = %0 * x1 + %1 * y1
		"vfmadd231pd %%zmm17, %%zmm0 ,  %%zmm25 \n\t" // x2 = %0 * x2 + %1 * y2
		"vfmadd231pd %%zmm18, %%zmm0 ,  %%zmm26 \n\t" // x3 = %0 * x3 + %1 * y3
		"vfmadd231pd %%zmm19, %%zmm0 ,  %%zmm27 \n\t" // x4 = %0 * x4 + %1 * y4

		"vfmadd231pd %%zmm20, %%zmm2 ,  %%zmm28 \n\t" // y1 = %2 * x1 + %3 * y1
		"vfmadd231pd %%zmm21, %%zmm2 ,  %%zmm29 \n\t" // y2 = %2 * x2 + %3 * y2
		"vfmadd231pd %%zmm22, %%zmm2 ,  %%zmm30 \n\t" // y3 = %2 * x3 + %3 * y3
		"vfmadd231pd %%zmm23, %%zmm2 ,  %%zmm31 \n\t" // y4 = %2 * x4 + %3 * y4

		"vmovupd                  %%zmm24,    (%6,%0,8)       \n\t" // 8 * x1
		"vmovupd                  %%zmm25,  64(%6,%0,8)       \n\t" // 8 * x2
		"vmovupd                  %%zmm26, 128(%6,%0,8)       \n\t" // 8 * x3
		"vmovupd                  %%zmm27, 192(%6,%0,8)       \n\t" // 8 * x4

		"vmovupd                  %%zmm28,    (%7,%0,8)       \n\t" // 8 * y1
		"vmovupd                  %%zmm29,  64(%7,%0,8)       \n\t" // 8 * y2
		"vmovupd                  %%zmm30, 128(%7,%0,8)       \n\t" // 8 * y3
		"vmovupd                  %%zmm31, 192(%7,%0,8)       \n\t" // 8 * y4

		"addq		$32 , %0	  	     \n\t"
		"subq	    $32 , %1		     \n\t"
		"jnz		1b		             \n\t"

		"vzeroupper				\n\t"
		"ret				         \n\t"

		: "+r"(i),   // 0
		  "+r"(n),   // 1
		  "+r"(h11), // 2
		  "+r"(h12), // 3
		  "+r"(h21), // 4
		  "+r"(h22)  // 5
		: "r"(x),    // 6
		  "r"(y)     // 7
		: "cc",
		  "%xmm0", "%xmm1", "%xmm2", "%xmm3",
//		  "%xmm4", "%xmm5", "%xmm6", "%xmm7",
//		  "%xmm8", "%xmm9", "%xmm10", "%xmm11",
//		  "%xmm12", "%xmm13", "%xmm14", "%xmm15",
		  "memory");
}

static void ori_drotm_one_kernel(long n, double *x, double *y, double *h11, double *h22)
{

	long register i = 0;

	__asm__ __volatile__(
		"vbroadcastsd		(%2), %%zmm0	\n\t" // h11
		"vbroadcastsd		(%3), %%zmm1	\n\t" // h22

		".p2align 4				             \n\t"
		"1:				             \n\t"
		"vmovupd                  (%4,%0,8), %%zmm16         \n\t" // 8 * x1
		"vmovupd                64(%4,%0,8), %%zmm17         \n\t" // 8 * x2
		"vmovupd               128(%4,%0,8), %%zmm18         \n\t" // 8 * x3
		"vmovupd               192(%4,%0,8), %%zmm19         \n\t" // 8 * x4

		"vmovupd                  (%5,%0,8), %%zmm20         \n\t" // 8 * y1
		"vmovupd                64(%5,%0,8), %%zmm21         \n\t" // 8 * y2
		"vmovupd               128(%5,%0,8), %%zmm22         \n\t" // 8 * y3
		"vmovupd               192(%5,%0,8), %%zmm23         \n\t" // 8 * y4

		"vmovupd 	 				%%zmm16, %%zmm24		 \n\t"
		"vmovupd 	 				%%zmm17, %%zmm25		 \n\t"
		"vmovupd 	 				%%zmm18, %%zmm26		 \n\t"
		"vmovupd 	 				%%zmm19, %%zmm27		 \n\t"

		"vmovupd 	 				%%zmm20, %%zmm28		 \n\t"
		"vmovupd 	 				%%zmm21, %%zmm29		 \n\t"
		"vmovupd 	 				%%zmm22, %%zmm30		 \n\t"
		"vmovupd 	 				%%zmm23, %%zmm31		 \n\t"

		"vfmadd231pd %%zmm0 , %%zmm24,  %%zmm20 \n\t" // x1 = %0 * x1 + y1
		"vfmadd231pd %%zmm0 , %%zmm25,  %%zmm21 \n\t" // x2 = %0 * x2 + y2
		"vfmadd231pd %%zmm0 , %%zmm26,  %%zmm22 \n\t" // x3 = %0 * x3 + y3
		"vfmadd231pd %%zmm0 , %%zmm27,  %%zmm23 \n\t" // x4 = %0 * x4 + y4

		"vfmsub231pd %%zmm1 , %%zmm28,  %%zmm16 \n\t" // y1 = -x1 + %1 * y1
		"vfmsub231pd %%zmm1 , %%zmm29,  %%zmm17 \n\t" // y2 = -x2 + %1 * y2
		"vfmsub231pd %%zmm1 , %%zmm30,  %%zmm18 \n\t" // y3 = x3 + %1 * y3
		"vfmsub231pd %%zmm1 , %%zmm31,  %%zmm19 \n\t" // y4 = x4 + %1 * y4

		"vmovupd                  %%zmm20,    (%4,%0,8)       \n\t" // 8 * x1
		"vmovupd                  %%zmm21,  64(%4,%0,8)       \n\t" // 8 * x2
		"vmovupd                  %%zmm22, 128(%4,%0,8)       \n\t" // 8 * x3
		"vmovupd                  %%zmm23, 192(%4,%0,8)       \n\t" // 8 * x4
/*
		"nop \n\t"
		"nop \n\t"
		"nop \n\t"
		"nop \n\t"
*/
		"vmovupd                  %%zmm16,    (%5,%0,8)       \n\t" // 8 * y1
		"vmovupd                  %%zmm17,  64(%5,%0,8)       \n\t" // 8 * y2
		"vmovupd                  %%zmm18, 128(%5,%0,8)       \n\t" // 8 * y3
		"vmovupd                  %%zmm19, 192(%5,%0,8)       \n\t" // 8 * y4
/*
		"nop \n\t"
		"nop \n\t"
		"nop \n\t"
		"nop \n\t"
*/
		"addq		$32 , %0	  	     \n\t"
		"subq	    $32 , %1		     \n\t"
		"jnz		1b		             \n\t"

		"vzeroupper				\n\t"
		"ret				         \n\t"

		: "+r"(i),   // 0
		  "+r"(n),   // 1
		  "+r"(h11), // 2
		  "+r"(h22)  // 3
		: "r"(x),    // 4
		  "r"(y)     // 5
		: "cc",
		  "%xmm0", "%xmm1", "%xmm2", "%xmm3",
		  "%xmm4", "%xmm5", "%xmm6", "%xmm7",
		  "%xmm8", "%xmm9", "%xmm10", "%xmm11",
		  "%xmm12", "%xmm13", "%xmm14", "%xmm15",
		  "memory");
}

int ori_drotm(long n, double *x, long inc_x, double *y, long inc_y, double *param)
{
  long i = 0;
  long ix = 0, iy = 0;

  double temp;
  double flag = param[0], h11, h12, h21, h22;
  if (n <= 0)
    return (0);

  if (flag != 0.0 && flag != 1.0 && flag != -1.0 && flag != -2.0)
  {
    return 0;
  }

  if (flag == -2.0)
  {
    return 0;
  }

  if ((inc_x == 1) && (inc_y == 1))
  {

    long n1 = n & -32;

    flag = param[0];
    h11 = param[1];
    h21 = param[2];
    h12 = param[3];
    h22 = param[4];

//    printf("flag = %f\n", flag);

    if (flag == 0.0)
    {
      h11 = h22 = 1.0;
    }
    else 
    {
      if (flag == 1.0)
      {
        h12 = 1.0;
        h21 = -1.0;
      }
    }

    if (n1 > 0)
    {
      if (flag == -1.0)
      {
        ori_drotm_kernel(n1, x, y, &h11, &h12, &h21, &h22);
      }
      else if (flag == 0.0)
      {
        ori_drotm_zero_kernel(n1, x, y, &h12, &h21);
      }
      else if (flag == 1.0)
      {
        ori_drotm_one_kernel(n1, x, y, &h11, &h22);
      }

      i = n1;
    }

//    printf("h11 = %f, h12 = %f \n h21 = %f, h22 = %f\n", h11, h12, h21, h22);

    while (i < n)
    {
//            printf("BEFORE: i = %d, n = %d, x[%d] = %f, y[%d] = %f\n", i, n, i, x[i], i, y[i]);
      temp = h11 * x[i] + h12 * y[i];
      y[i] = h21 * x[i] + h22 * y[i];
      x[i] = temp;
//            printf("AFTER: i = %d, n = %d, x[%d] = %f, y[%d] = %f\n", i, n, i, x[i], i, y[i]);

      i++;
    }
  }
  else
  {
    long inc = -1 * inc_x;
    long n1 = n & inc;
    flag = param[0];
    h11 = param[1];
    h21 = param[2];
    h12 = param[3];
    h22 = param[4];

    if (flag == 0.0)
    {
      h11 = h22 = 1.0;
    }
    else if (flag = 1.0)
    {
      h12 = 1.0;
      h21 = -1.0;
    }
    else
    {
      return 0;
    }

    while (i < n1)
    {
      temp = h11 * x[ix] + h12 * y[iy];
      y[iy] = h21 * y[iy] + h22 * x[ix];
      x[ix] = temp;
      ix += inc_x;
      iy += inc_y;
      i -= inc;
    }
  }
  return (0);
}

void ftblas_drotm_ori(const long int n, const double *x, const long int inc_x, const double *y, const long int inc_y, const double *param){
    ori_drotm(n, (double*)x, inc_x, (double*)y, inc_y, (double*)param);
}