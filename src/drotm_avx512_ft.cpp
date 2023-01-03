#include <stdio.h>
#include <stdlib.h>
#include "../include/ftblas.h"


static long ft_drotm_zero_kernel(long n, double *x, double *y, double *h12, double *h21)
{

	long register i = 0;
	long register err_num = 0;

	__asm__ __volatile__(
		"vbroadcastsd           (%2), %%zmm0      \n\t" // h12
		"vbroadcastsd           (%3), %%zmm1      \n\t" // h21

		/* Prologue */
		"vmovupd                  (%5,%0,8), %%zmm16         \n\t" // LX1

		"vmovupd                            %%zmm16, %%zmm24         \n\t" // BX1
		"vmovupd                            %%zmm16, %%zmm8          \n\t" // BX1'
		"vmovupd                  (%6,%0,8), %%zmm20         \n\t"		   // LY1
		"vmovupd                64(%5,%0,8), %%zmm17         \n\t"		   // LX2

		"vmovupd                            %%zmm20, %%zmm28         \n\t" // BY1
		"vfmadd231pd         %%zmm0 , %%zmm20, %%zmm16         \n\t"	   // FX1
		"vfmadd231pd         %%zmm0 , %%zmm20, %%zmm24         \n\t"	   // FX1 - dup - 16 - 24
		"vmovupd                            %%zmm17, %%zmm25         \n\t" // BX2
		"vmovupd                            %%zmm17, %%zmm9          \n\t" // BX2'
		"vmovupd                64(%6,%0,8), %%zmm21         \n\t"		   // LY2
		"vmovupd               128(%5,%0,8), %%zmm18         \n\t"		   // LX3

		"vpcmpeqd            %%zmm24, %%zmm16, %%k0            \n\t"	   // check - 0
		"vmovupd                    %%zmm16,     (%5,%0,8)   \n\t"		   // SX1
		"vmovupd                            %%zmm21, %%zmm29         \n\t" // BY2
		"vfmadd231pd       %%zmm0 , %%zmm21,  %%zmm17        \n\t"		   // FX2
		"vfmadd231pd       %%zmm0 , %%zmm21,  %%zmm25        \n\t"		   // FX2 - dup - 17 - 25
		"vmovupd                            %%zmm18, %%zmm26         \n\t" // BX3
		"vmovupd                            %%zmm18, %%zmm10         \n\t" // BX3'
		"vmovupd               128(%6,%0,8), %%zmm22         \n\t"		   // LY3
		"vmovupd               192(%5,%0,8), %%zmm19         \n\t"		   // LX4

		"vfmadd231pd         %%zmm1 , %%zmm8 , %%zmm20         \n\t"	   // FY1
		"vfmadd231pd         %%zmm1 , %%zmm8 , %%zmm28         \n\t"	   // FY1 - dup - 20 - 28
		"vpcmpeqd            %%zmm25, %%zmm17, %%k1            \n\t"	   // check - 1
		"vmovupd                    %%zmm17,   64(%5,%0,8)   \n\t"		   // SX2
		"vmovupd                            %%zmm22, %%zmm30         \n\t" // BY3
		"vfmadd231pd       %%zmm0 , %%zmm22,  %%zmm18        \n\t"		   // FX3
		"vfmadd231pd       %%zmm0 , %%zmm22,  %%zmm26        \n\t"		   // FX3 - dup - 18 - 26
		"vmovupd                            %%zmm19, %%zmm27         \n\t" // BX4
		"vmovupd                            %%zmm19, %%zmm11         \n\t" // BX4'
		"vmovupd               192(%6,%0,8), %%zmm23         \n\t"		   // LY4
		"vmovupd               256(%5,%0,8), %%zmm16         \n\t"		   // LX5

		"addq       $40 , %0               \n\t"
		"subq     $40 , %1                 \n\t"
		"jz             2f                         \n\t"

		/* Loop Body */
		".p2align 4                                \n\t"
		"1:                                  \n\t"
		"vpcmpeqd            %%zmm28, %%zmm20, %%k4            \n\t"	   // check - 4
		"vmovupd                    %%zmm20, -320(%6,%0,8)   \n\t"		   // SY1
		"vfmadd231pd       %%zmm1 , %%zmm9 ,  %%zmm21        \n\t"		   // FY2
		"vfmadd231pd       %%zmm1 , %%zmm9 ,  %%zmm29        \n\t"		   // FY2 - dup - 21 - 29
		"vpcmpeqd            %%zmm26, %%zmm18, %%k2            \n\t"	   // check - 2
		"vmovupd                    %%zmm18, -192(%5,%0,8)   \n\t"		   // SX3
		"vmovupd                            %%zmm23, %%zmm31         \n\t" // BY4
		"vfmadd231pd       %%zmm0 , %%zmm23, %%zmm19         \n\t"		   // FX4
		"vfmadd231pd       %%zmm0 , %%zmm23, %%zmm27         \n\t"		   // FX4 - dup - 19 - 27
		"vmovupd                            %%zmm16, %%zmm24         \n\t" // BX5
		"vmovupd                            %%zmm16, %%zmm8          \n\t" // BX5'
		"vmovupd               -64(%6,%0,8), %%zmm20         \n\t"		   // LY5
		"vmovupd                  (%5,%0,8), %%zmm17         \n\t"		   // LX6

		"kandw               %%k4   , %%k2   , %%k4            \n\t"	   // reduce - 4 - 2 -> 4
		"vpcmpeqd            %%zmm29, %%zmm21, %%k5            \n\t"	   // check - 5
		"vmovupd                    %%zmm21, -256(%6,%0,8)   \n\t"		   // SY2
		"vfmadd231pd       %%zmm1 , %%zmm10,  %%zmm22        \n\t"		   // FY3
		"vfmadd231pd       %%zmm1 , %%zmm10,  %%zmm30        \n\t"		   // FY3 - dup - 22 - 30
		"vpcmpeqd            %%zmm27, %%zmm19, %%k3            \n\t"	   // check - 3
		"vmovupd                    %%zmm19, -128(%5,%0,8)   \n\t"		   // SX4
		"vmovupd                            %%zmm20, %%zmm28         \n\t" // BY5
		"vfmadd231pd       %%zmm0 , %%zmm20, %%zmm16         \n\t"		   // FX5
		"vfmadd231pd       %%zmm0 , %%zmm20, %%zmm24         \n\t"		   // FX5 - dup - 16 - 24
		"vmovupd                            %%zmm17, %%zmm25         \n\t" // BX6
		"vmovupd                            %%zmm17, %%zmm9          \n\t" // BX6'
		"vmovupd                  (%6,%0,8), %%zmm21         \n\t"		   // LY6
		"kandw               %%k5   , %%k3   , %%k5            \n\t"	   // reduce - 5 - 3 -> 5
		"vmovupd                64(%5,%0,8), %%zmm18         \n\t"		   // LX7

		"kxnorw              %%k2   , %%k2   , %%k2            \n\t"	   // reg-reuse
		"kandw               %%k5   , %%k4   , %%k5            \n\t"	   // reduce - 5 - 4 -> 5
		"vpcmpeqd            %%zmm30, %%zmm22, %%k6            \n\t"	   // check - 6
		"vmovupd                    %%zmm22, -192(%6,%0,8)   \n\t"		   // SY3
		"vfmadd231pd       %%zmm1 , %%zmm11,  %%zmm23        \n\t"		   // FY4
		"vfmadd231pd       %%zmm1 , %%zmm11,  %%zmm31        \n\t"		   // FY4 - dup - 23 - 31
		"vpcmpeqd            %%zmm24, %%zmm16, %%k0            \n\t"	   // check - 0
		"vmovupd                    %%zmm16,  -64(%5,%0,8)   \n\t"		   // SX5
		"vmovupd                            %%zmm21, %%zmm29         \n\t" // BY6
		"vfmadd231pd       %%zmm0 , %%zmm21, %%zmm17         \n\t"		   // FX6
		"vfmadd231pd       %%zmm0 , %%zmm21, %%zmm25         \n\t"		   // FX6 - dup - 17 - 25
		"vmovupd                            %%zmm18, %%zmm26         \n\t" // BX7
		"vmovupd                            %%zmm18, %%zmm10         \n\t" // BX7'
		"vmovupd                64(%6,%0,8), %%zmm22         \n\t"		   // LY7
		"kandw               %%k6   , %%k0   , %%k6            \n\t"	   // reduce - 6 - 0 -> 6
		"vmovupd               128(%5,%0,8), %%zmm19         \n\t"		   // LX8

		"kandw               %%k6   , %%k5   , %%k6            \n\t"	   // reduce - 6 - 5 -> 6
		"vpcmpeqd          %%zmm31, %%zmm23, %%k7            \n\t"		   // check - 7
		"vmovupd                    %%zmm23, -128(%6,%0,8)   \n\t"		   // SY4
		"vfmadd231pd       %%zmm1 , %%zmm8 ,  %%zmm20        \n\t"		   // FY5
		"vfmadd231pd       %%zmm1 , %%zmm8 ,  %%zmm28        \n\t"		   // FY5 - dup - 20 - 28
		"vpcmpeqd            %%zmm25, %%zmm17, %%k1            \n\t"	   // check - 1
		"vmovupd                    %%zmm17,     (%5,%0,8)   \n\t"		   // SX6
		"vmovupd                            %%zmm22, %%zmm30         \n\t" // BY7
		"vfmadd231pd       %%zmm0 , %%zmm22, %%zmm18         \n\t"		   // FX7
		"vfmadd231pd       %%zmm0 , %%zmm22, %%zmm26         \n\t"		   // FX7 - dup - 18 - 26
		"vmovupd                            %%zmm19, %%zmm27         \n\t" // BX8
		"vmovupd                            %%zmm19, %%zmm11         \n\t" // BX8'
		"vmovupd               128(%6,%0,8), %%zmm23         \n\t"		   // LY8
		"kandw               %%k7   , %%k1   , %%k7            \n\t"	   // reduce - 7 - 1 -> 7
		"vmovupd               192(%5,%0,8), %%zmm16         \n\t"		   // LX9

		"kandw               %%k7   , %%k6   , %%k7            \n\t" // reduce - 7 - 6 -> 7

		"addq       $32 , %0               \n\t"

		"kxorw               %%k7   , %%k2   , %%k7            \n\t"
		"ktestw              %%k7   , %%k2                     \n\t"
		"jnz            4f           \n\t"

		"subq     $32 , %1                 \n\t"
		"jnz        1b                       \n\t"

		/* Epilogue */
		"2:                                  \n\t"
		"vpcmpeqd            %%zmm28, %%zmm20, %%k4            \n\t"	   // check - 4
		"vmovupd                    %%zmm20, -320(%6,%0,8)   \n\t"		   // SY1
		"vfmadd231pd       %%zmm1 , %%zmm9 ,  %%zmm21        \n\t"		   // FY2
		"vfmadd231pd       %%zmm1 , %%zmm9 ,  %%zmm29        \n\t"		   // FY2 - dup - 21 - 29
		"vpcmpeqd            %%zmm26, %%zmm18, %%k2            \n\t"	   // check - 2
		"vmovupd                    %%zmm18, -192(%5,%0,8)   \n\t"		   // SX3
		"vmovupd                            %%zmm23, %%zmm31         \n\t" // BY4
		"vfmadd231pd       %%zmm0 , %%zmm23, %%zmm19         \n\t"		   // FX4
		"vfmadd231pd       %%zmm0 , %%zmm23, %%zmm27         \n\t"		   // FX4 - dup - 19 - 27
		"vmovupd                            %%zmm16, %%zmm24         \n\t" // BX5
		"vmovupd                            %%zmm16, %%zmm8          \n\t" // BX5'
		"vmovupd               -64(%6,%0,8), %%zmm20         \n\t"		   // LY5

		"kandw               %%k4   , %%k2   , %%k4            \n\t"	   // reduce - 4 - 2 -> 4
		"vpcmpeqd            %%zmm29, %%zmm21, %%k5            \n\t"	   // check - 5
		"vmovupd                    %%zmm21, -256(%6,%0,8)   \n\t"		   // SY2
		"vfmadd231pd       %%zmm1 , %%zmm10,  %%zmm22        \n\t"		   // FY3
		"vfmadd231pd       %%zmm1 , %%zmm10,  %%zmm30        \n\t"		   // FY3 - dup - 22 - 30
		"vpcmpeqd            %%zmm27, %%zmm19, %%k3            \n\t"	   // check - 3
		"vmovupd                    %%zmm19, -128(%5,%0,8)   \n\t"		   // SX4
		"vmovupd                            %%zmm20, %%zmm28         \n\t" // BY5
		"vfmadd231pd       %%zmm0 , %%zmm20, %%zmm16         \n\t"		   // FX5
		"vfmadd231pd       %%zmm0 , %%zmm20, %%zmm24         \n\t"		   // FX5 - dup - 16 - 24

		"kandw               %%k5   , %%k3   , %%k5            \n\t" // reduce - 5 - 3 -> 5
		"vpcmpeqd            %%zmm30, %%zmm22, %%k6            \n\t" // check - 6
		"vmovupd                    %%zmm22, -192(%6,%0,8)   \n\t"   // SY3
		"vfmadd231pd       %%zmm1 , %%zmm11,  %%zmm23        \n\t"   // FY4
		"vfmadd231pd       %%zmm1 , %%zmm11,  %%zmm31        \n\t"   // FY4 - dup - 23 - 31
		"vpcmpeqd            %%zmm24, %%zmm16, %%k0            \n\t" // check - 0
		"vmovupd                    %%zmm16,  -64(%5,%0,8)   \n\t"   // SX5

		"kandw               %%k6   , %%k0   , %%k6            \n\t" // reduce - 6 - 0 -> 6
		"kandw               %%k5   , %%k4   , %%k5            \n\t" // reduce - 5 - 4 -> 5
		"vpcmpeqd          %%zmm31, %%zmm23, %%k7            \n\t"   // check - 7
		"vmovupd                    %%zmm23, -128(%6,%0,8)   \n\t"   // SY4
		"vfmadd231pd       %%zmm1 , %%zmm8 ,  %%zmm20        \n\t"   // FY5
		"vfmadd231pd       %%zmm1 , %%zmm8 ,  %%zmm28        \n\t"   // FY5 - dup - 20 - 28

		"kandw               %%k6   , %%k5   , %%k6            \n\t" // reduce - 6 - 5 -> 6
		"vpcmpeqd            %%zmm28, %%zmm20, %%k4            \n\t" // check - 4
		"vmovupd                    %%zmm20,  -64(%6,%0,8)   \n\t"   // SY5

		"kandw               %%k7   , %%k4   , %%k7            \n\t" // reduce - 7 - 4 -> 7
		"kxnorw              %%k2   , %%k2   , %%k2            \n\t" // reg-reuse
		"kandw               %%k7   , %%k6   , %%k7            \n\t" // reduce - 7 - 6 -> 7

		"kxorw               %%k7   , %%k2   , %%k7            \n\t"
		"ktestw              %%k7   , %%k2                     \n\t"
		"jnz               5f           \n\t"

		"jmp                 3f                                            \n\t"

		"5:                                      \n\t"
		"addq     $1 , %4                 \n\t"
		"jmp        3f                             \n\t"

		"4:                                      \n\t"
		"addq     $1 , %4                 \n\t"
		"subq     $32 , %1                 \n\t"
		"jnz        1b                       \n\t"
		"jmp        2b                             \n\t"

		"3:                                      \n\t"
		"vzeroupper                            \n\t"
		: "+r"(i),		// 0
		  "+r"(n),		// 1
		  "+r"(h12),	// 2
		  "+r"(h21),	// 3
		  "+r"(err_num) // 4
		: "r"(x),		// 5
		  "r"(y)		// 6
		: "cc",
		  "%xmm0", "%xmm1", "%xmm2", "%xmm3",
		  "%xmm4", "%xmm5", "%xmm6", "%xmm7",
		  "%xmm8", "%xmm9", "%xmm10", "%xmm11",
		  "%xmm12", "%xmm13", "%xmm14", "%xmm15",
		  "memory");
	return err_num;
}

static long ft_drotm_kernel(long n, double *x, double *y, double *h11, double *h12, double *h21, double *h22)
{

      long register i = 0;
      long register err_num = 0;
      __asm__ __volatile__(
            "vbroadcastsd           (%2), %%zmm0      \n\t" // h11
            "vbroadcastsd           (%3), %%zmm1      \n\t" // h12
            "vbroadcastsd           (%4), %%zmm2      \n\t" // h21
            "vbroadcastsd           (%5), %%zmm3      \n\t" // h22

            /* Prologue */
            "vmovupd                  (%8,%0,8), %%zmm20         \n\t" // LY1

            "vmulpd      %%zmm1 , %%zmm20, %%zmm24  \n\t" // MY1
            "vmulpd      %%zmm1 , %%zmm20, %%zmm4   \n\t" // MY1, dup
            "vmulpd      %%zmm3 , %%zmm20, %%zmm8   \n\t" // MY1'
            "vmulpd      %%zmm3 , %%zmm20, %%zmm28  \n\t" // MY1' - dup
            "vmovupd                  (%7,%0,8), %%zmm16         \n\t" // LX1
            "vmovupd                64(%8,%0,8), %%zmm21         \n\t" // LY2

            "vfmadd231pd %%zmm16, %%zmm0 ,  %%zmm24 \n\t" // FX1
            "vfmadd231pd %%zmm16, %%zmm0 ,  %%zmm4  \n\t" // FX1 - dup - 24 - 4
            "vmulpd      %%zmm1 , %%zmm21, %%zmm25  \n\t" // MY2
            "vmulpd      %%zmm1 , %%zmm21, %%zmm5   \n\t" // MY2 - dup
            "vmulpd      %%zmm3 , %%zmm21, %%zmm29  \n\t" // MY2'
            "vmulpd      %%zmm3 , %%zmm21, %%zmm9   \n\t" // MY2' - dup
            "vmovupd                64(%7,%0,8), %%zmm17         \n\t" // LX2
            "vmovupd               128(%8,%0,8), %%zmm22         \n\t" // LY3

            "vpcmpeqd                  %%zmm24, %%zmm4 , %%k0         \n\t" // check - 0
            "vmovupd                  %%zmm24,    (%7,%0,8)       \n\t" // SX1
            "vfmadd231pd %%zmm17, %%zmm0 ,  %%zmm25 \n\t" // FX2
            "vfmadd231pd %%zmm17, %%zmm0 ,  %%zmm5  \n\t" // FX2 - dup - 25 - 5
            "vmulpd      %%zmm1 , %%zmm22, %%zmm26  \n\t" // MY3
            "vmulpd      %%zmm1 , %%zmm22, %%zmm6   \n\t" // MY3 - dup
            "vmulpd      %%zmm3 , %%zmm22, %%zmm30  \n\t" // MY3'
            "vmulpd      %%zmm3 , %%zmm22, %%zmm10  \n\t" // MY3' - dup
            "vmovupd               128(%7,%0,8), %%zmm18         \n\t" // LX3
            "vmovupd               192(%8,%0,8), %%zmm23         \n\t" // LY4

            "vfmadd231pd %%zmm20, %%zmm2 ,  %%zmm28 \n\t" // FY1
            "vfmadd231pd %%zmm20, %%zmm2 ,  %%zmm8  \n\t" // FY1 - dup - 28 - 8
            "vpcmpeqd                  %%zmm25, %%zmm5 , %%k1         \n\t" // check - 1
            "vmovupd                  %%zmm25,  64(%7,%0,8)       \n\t" // SX2
            "vfmadd231pd %%zmm18, %%zmm0 ,  %%zmm26 \n\t" // FX3
            "vfmadd231pd %%zmm18, %%zmm0 ,  %%zmm6  \n\t" // FX3 - dup - 26 - 6
            "vmulpd      %%zmm1 , %%zmm23, %%zmm27  \n\t" // MY4
            "vmulpd      %%zmm1 , %%zmm23, %%zmm7   \n\t" // MY4 - dup
            "vmulpd      %%zmm3 , %%zmm23, %%zmm31  \n\t" // MY4'
            "vmulpd      %%zmm3 , %%zmm23, %%zmm11  \n\t" // MY4' - dup
            "vmovupd               192(%7,%0,8), %%zmm19         \n\t" // LX4
            "vmovupd               256(%8,%0,8), %%zmm20         \n\t" // LY5

            "addq       $40 , %0               \n\t"
            "subq     $40 , %1                 \n\t"
            "jz         2f                       \n\t"

            ".p2align 4                                \n\t"
            "1:                                  \n\t"

            "vpcmpeqd                  %%zmm28, %%zmm8 , %%k4         \n\t" // check - 4
            "vmovupd                  %%zmm28, -320(%8,%0,8)       \n\t" // SY1
            "vfmadd231pd %%zmm21, %%zmm2 ,  %%zmm29 \n\t" // FY2
            "vfmadd231pd %%zmm21, %%zmm2 ,  %%zmm9  \n\t" // FY2 - dup - 29 - 9
            "vpcmpeqd                  %%zmm26, %%zmm6 , %%k2         \n\t" // check - 2
            "vmovupd                  %%zmm26, -192(%7,%0,8)       \n\t" // SX3
            "vfmadd231pd %%zmm19, %%zmm0 ,  %%zmm27 \n\t" // FX4
            "vfmadd231pd %%zmm19, %%zmm0 ,  %%zmm7  \n\t" // FX4 - dup - 27 - 7
            "vmulpd      %%zmm1 , %%zmm20, %%zmm24  \n\t" // MY5
            "vmulpd      %%zmm1 , %%zmm20, %%zmm4   \n\t" // MY5 - dup
            "vmulpd      %%zmm3 , %%zmm20, %%zmm28  \n\t" // MY5'
            "vmulpd      %%zmm3 , %%zmm20, %%zmm8   \n\t" // MY5' - dup
            "vmovupd                 -64(%7,%0,8), %%zmm16         \n\t" // LX5
            "vmovupd                    (%8,%0,8), %%zmm21         \n\t" // LY6

            "kandw                     %%k4   , %%k2   , %%k4         \n\t" // red - 4 - 2 -> 4
            "kxnorw                    %%k2   , %%k2   , %%k2         \n\t" // reg-reuse
            "vpcmpeqd                  %%zmm29, %%zmm9 , %%k5         \n\t" // check - 5
            "vmovupd                  %%zmm29, -256(%8,%0,8)       \n\t" // SY2
            "vfmadd231pd %%zmm22, %%zmm2 ,  %%zmm30 \n\t" // FY3
            "vfmadd231pd %%zmm22, %%zmm2 ,  %%zmm10 \n\t" // FY3 - dup - 30 - 10
            "vpcmpeqd                  %%zmm27, %%zmm7 , %%k3         \n\t" // check - 3
            "vmovupd                  %%zmm27, -128(%7,%0,8)       \n\t" // SX4
            "vfmadd231pd %%zmm16, %%zmm0 ,  %%zmm24 \n\t" // FX5
            "vfmadd231pd %%zmm16, %%zmm0 ,  %%zmm4  \n\t" // FX5 - dup - 24 - 4
            "vmulpd      %%zmm1 , %%zmm21, %%zmm25  \n\t" // MY6
            "vmulpd      %%zmm1 , %%zmm21, %%zmm5   \n\t" // MY6 - dup 
            "vmulpd      %%zmm3 , %%zmm21, %%zmm29  \n\t" // MY6'
            "vmulpd      %%zmm3 , %%zmm21, %%zmm9   \n\t" // MY6' - dup
            "vmovupd                    (%7,%0,8), %%zmm17         \n\t" // LX6
            "vmovupd                  64(%8,%0,8), %%zmm22         \n\t" // LY7

            "kandw                     %%k5   , %%k3   , %%k5         \n\t" // red - 5 - 3 -> 5
            "vpcmpeqd                  %%zmm30, %%zmm10, %%k6         \n\t" // check - 6
            "vmovupd                  %%zmm30, -192(%8,%0,8)       \n\t" // SY3
            "vfmadd231pd %%zmm23, %%zmm2 ,  %%zmm31 \n\t" // FY4
            "vfmadd231pd %%zmm23, %%zmm2 ,  %%zmm11 \n\t" // FY4 - dup - 31 - 11
            "vpcmpeqd                  %%zmm24, %%zmm4 , %%k0         \n\t" // check - 0
            "vmovupd                  %%zmm24,  -64(%7,%0,8)       \n\t" // SX5
            "vfmadd231pd %%zmm17, %%zmm0 ,  %%zmm25 \n\t" // FX6
            "vfmadd231pd %%zmm17, %%zmm0 ,  %%zmm5  \n\t" // FX6 - dup - 25 - 5
            "vmulpd      %%zmm1 , %%zmm22, %%zmm26  \n\t" // MY7
            "vmulpd      %%zmm1 , %%zmm22, %%zmm6  \n\t" // MY7 - dup
            "vmulpd      %%zmm3 , %%zmm22, %%zmm30  \n\t" // MY7'
            "vmulpd      %%zmm3 , %%zmm22, %%zmm10  \n\t" // MY7' - dup
            "vmovupd                  64(%7,%0,8), %%zmm18         \n\t" // LX7
            "vmovupd                 128(%8,%0,8), %%zmm23         \n\t" // LY8

            "kandw                     %%k6   , %%k0   , %%k6         \n\t" // red - 6 - 0 -> 6
            "kandw                     %%k5   , %%k4   , %%k5         \n\t" // red - 5 - 4 -> 5
            "vpcmpeqd                  %%zmm31, %%zmm11, %%k7         \n\t" // check - 7
            "vmovupd                  %%zmm31, -128(%8,%0,8)       \n\t" // SY4
            "vfmadd231pd %%zmm20, %%zmm2 ,  %%zmm28 \n\t" // FY5
            "vfmadd231pd %%zmm20, %%zmm2 ,  %%zmm8  \n\t" // FY5 - dup - 28 - 8
            "vpcmpeqd                  %%zmm25, %%zmm5 , %%k1         \n\t" // check - 1
            "vmovupd                  %%zmm25,     (%7,%0,8)       \n\t" // SX6
            "vfmadd231pd %%zmm18, %%zmm0 ,  %%zmm26 \n\t" // FX7
            "vfmadd231pd %%zmm18, %%zmm0 ,  %%zmm6  \n\t" // FX7 - dup - 26 - 6
            "vmulpd      %%zmm1 , %%zmm23, %%zmm27  \n\t" // MY8
            "vmulpd      %%zmm1 , %%zmm23, %%zmm7   \n\t" // MY8 - dup
            "vmulpd      %%zmm3 , %%zmm23, %%zmm31  \n\t" // MY8'
            "vmulpd      %%zmm3 , %%zmm23, %%zmm11  \n\t" // MY8' - dup
            "vmovupd                 128(%7,%0,8), %%zmm19         \n\t" // LX8
            "vmovupd                 192(%8,%0,8), %%zmm20         \n\t" // LY9

            "kandw                     %%k7   , %%k1   , %%k7         \n\t" // red - 7 - 1 -> 7
            "kandw                     %%k6   , %%k5   , %%k6         \n\t" // red - 6 - 5 -> 6
            "kandw                     %%k7   , %%k6   , %%k7         \n\t" // red - 7 - 1 -> 7
            "addq       $32 , %0               \n\t"

            "kxorw                     %%k7   , %%k2   , %%k7         \n\t"
            "ktestw                    %%k7   , %%k2                  \n\t"
            "jnz                       4f           \n\t"

            "subq     $32 , %1                 \n\t"
            "jnz        1b                       \n\t"
      
            "2:                                  \n\t"
            "vpcmpeqd                  %%zmm28, %%zmm8 , %%k4         \n\t" // check - 4
            "vmovupd                  %%zmm28, -320(%8,%0,8)       \n\t" // SY1
            "vfmadd231pd %%zmm21, %%zmm2 ,  %%zmm29 \n\t" // FY2
            "vfmadd231pd %%zmm21, %%zmm2 ,  %%zmm9  \n\t" // FY2 - dup - 29 - 9
            "vpcmpeqd                  %%zmm26, %%zmm6 , %%k2         \n\t" // check - 2
            "vmovupd                  %%zmm26, -192(%7,%0,8)       \n\t" // SX3
            "vfmadd231pd %%zmm19, %%zmm0 ,  %%zmm27 \n\t" // FX4
            "vfmadd231pd %%zmm19, %%zmm0 ,  %%zmm7  \n\t" // FX4 - dup - 27 - 7
            "vmulpd      %%zmm1 , %%zmm20, %%zmm24  \n\t" // MY5
            "vmulpd      %%zmm1 , %%zmm20, %%zmm4   \n\t" // MY5 - dup - 24 - 4
            "vmulpd      %%zmm3 , %%zmm20, %%zmm28  \n\t" // MY5'
            "vmulpd      %%zmm3 , %%zmm20, %%zmm8   \n\t" // MY5' - dup - 28 - 8
            "vmovupd                 -64(%7,%0,8), %%zmm16         \n\t" // LX5

            "kandw                     %%k4   , %%k2   , %%k4         \n\t" // red - 4 - 2 -> 4
            "kxnorw                    %%k2   , %%k2   , %%k2         \n\t" // reg-reuse
            "vpcmpeqd                  %%zmm29, %%zmm9 , %%k5         \n\t" // check - 5
            "vmovupd                  %%zmm29, -256(%8,%0,8)       \n\t" // SY2
            "vfmadd231pd %%zmm22, %%zmm2 ,  %%zmm30 \n\t" // FY3
            "vfmadd231pd %%zmm22, %%zmm2 ,  %%zmm10 \n\t" // FY3 - dup - 30 - 10
            "vpcmpeqd                  %%zmm27, %%zmm7 , %%k3         \n\t" // check - 3
            "vmovupd                  %%zmm27, -128(%7,%0,8)       \n\t" // SX4
            "vfmadd231pd %%zmm16, %%zmm0 ,  %%zmm24 \n\t" // FX5
            "vfmadd231pd %%zmm16, %%zmm0 ,  %%zmm4  \n\t" // FX5 - dup - 24 - 4

            "kandw                     %%k5   , %%k3   , %%k5         \n\t" // red - 5 - 3 -> 5
            "vpcmpeqd                  %%zmm30, %%zmm10, %%k6         \n\t" // check - 6
            "vmovupd                  %%zmm30, -192(%8,%0,8)       \n\t" // SY3
            "vfmadd231pd %%zmm23, %%zmm2 ,  %%zmm31 \n\t" // FY4
            "vfmadd231pd %%zmm23, %%zmm2 ,  %%zmm11 \n\t" // FY4 - dup - 31 - 11
            "vpcmpeqd                  %%zmm24, %%zmm4 , %%k0         \n\t" // check - 0
            "vmovupd                  %%zmm24,  -64(%7,%0,8)       \n\t" // SX5

            "kandw                     %%k6   , %%k0   , %%k6         \n\t" // red - 6 - 0 -> 6
            "kandw                     %%k5   , %%k4   , %%k5         \n\t" // red - 5 - 4 -> 5
            "vpcmpeqd                  %%zmm31, %%zmm11, %%k7         \n\t" // check - 7
            "vmovupd                  %%zmm31, -128(%8,%0,8)       \n\t" // SY4
            "vfmadd231pd %%zmm20, %%zmm2 ,  %%zmm28 \n\t" // FY5
            "vfmadd231pd %%zmm20, %%zmm2 ,  %%zmm8  \n\t" // FY5 - dup - 28 - 8

            "vpcmpeqd                  %%zmm28, %%zmm8 , %%k4         \n\t" // check - 4
            "vmovupd                  %%zmm28,  -64(%8,%0,8)       \n\t" // SY5

            "kandw                     %%k7   , %%k4   , %%k7         \n\t" // red - 7 - 4 -> 7
            "kandw                     %%k6   , %%k5   , %%k6         \n\t" // red - 6 - 5 -> 6
            "kandw                     %%k7   , %%k6   , %%k7         \n\t" // red - 7 - 6 -> 7

            "kxorw                     %%k7   , %%k2   , %%k7         \n\t"
            "ktestw                    %%k7   , %%k2                  \n\t"
            "jnz                       5f           \n\t"
            "jmp                       3f           \n\t"


            "4:                                  \n\t"
            "addq       $1 , %6                  \n\t"
            "subq       $32 , %1                 \n\t"
            "jnz        1b                       \n\t"
            "jmp        2b                       \n\t"

            "5:                                  \n\t"
            "addq       $1 , %6                  \n\t"

            "3:                                  \n\t"
            "vzeroupper                          \n\t"

            : "+r"(i),   // 0
              "+r"(n),   // 1
              "+r"(h11), // 2
              "+r"(h12), // 3
              "+r"(h21), // 4
              "+r"(h22), // 5
              "+r"(err_num) // 6
            : "r"(x),    // 7
              "r"(y)     // 8
            : "cc",
              "%xmm0", "%xmm1", "%xmm2", "%xmm3",
              "%xmm4", "%xmm5", "%xmm6", "%xmm7",
              "%xmm8", "%xmm9", "%xmm10", "%xmm11",
              "%xmm12", "%xmm13", "%xmm14", "%xmm15",
              "memory");
	return err_num;
}

static long ft_drotm_one_kernel(long n, double *x, double *y, double *h11, double *h22)
{

      long register i = 0;
      long register err_num = 0;

      __asm__ __volatile__(
            "vbroadcastsd           (%2), %%zmm1      \n\t" // h12
            "vbroadcastsd           (%3), %%zmm0      \n\t" // h21

            /* Prologue */
            "vmovupd                  (%5,%0,8), %%zmm16           \n\t" // LX1

            "vmovupd                    %%zmm16, %%zmm24           \n\t" // BX1
            "vmovupd                    %%zmm16, %%zmm8            \n\t" // BX1'
            "vmovupd                  (%6,%0,8), %%zmm20           \n\t" // LY1
            "vmovupd                64(%5,%0,8), %%zmm17           \n\t"      // LX2

            "vmovupd             %%zmm20, %%zmm28                  \n\t" // BY1
            "vfmsub231pd         %%zmm0 , %%zmm20, %%zmm16         \n\t"         // FY1
            "vfmsub231pd         %%zmm0 , %%zmm20, %%zmm24         \n\t"         // FY1 - dup - 16 - 24
            "vmovupd                            %%zmm17, %%zmm25         \n\t" // BX2
            "vmovupd                            %%zmm17, %%zmm9          \n\t" // BX2'
            "vmovupd                64(%6,%0,8), %%zmm21         \n\t"           // LY2
            "vmovupd               128(%5,%0,8), %%zmm18         \n\t"           // LX3

            "vpcmpeqd            %%zmm24, %%zmm16, %%k0            \n\t"         // check - 0
            "vmovupd                    %%zmm16,     (%6,%0,8)   \n\t"           // SY1
            "vmovupd                            %%zmm21, %%zmm29         \n\t" // BY2
            "vfmsub231pd       %%zmm0 , %%zmm21,  %%zmm17        \n\t"           // FY2
            "vfmsub231pd       %%zmm0 , %%zmm21,  %%zmm25        \n\t"           // FY2 - dup - 17 - 25
            "vmovupd                            %%zmm18, %%zmm26         \n\t" // BX3
            "vmovupd                            %%zmm18, %%zmm10         \n\t" // BX3'
            "vmovupd               128(%6,%0,8), %%zmm22         \n\t"           // LY3
            "vmovupd               192(%5,%0,8), %%zmm19         \n\t"           // LX4

            "vfmadd231pd         %%zmm1 , %%zmm8 , %%zmm20         \n\t"         // FX1
            "vfmadd231pd         %%zmm1 , %%zmm8 , %%zmm28         \n\t"         // FX1 - dup - 20 - 28
            "vpcmpeqd            %%zmm25, %%zmm17, %%k1            \n\t"         // check - 1
            "vmovupd                    %%zmm17,   64(%6,%0,8)   \n\t"           // SY2
            "vmovupd                            %%zmm22, %%zmm30         \n\t" // BY3
            "vfmsub231pd       %%zmm0 , %%zmm22,  %%zmm18        \n\t"           // FY3
            "vfmsub231pd       %%zmm0 , %%zmm22,  %%zmm26        \n\t"           // FY3 - dup - 18 - 26
            "vmovupd                            %%zmm19, %%zmm27         \n\t" // BX4
            "vmovupd                            %%zmm19, %%zmm11         \n\t" // BX4'
            "vmovupd               192(%6,%0,8), %%zmm23         \n\t"           // LY4
            "vmovupd               256(%5,%0,8), %%zmm16         \n\t"           // LX5

            "addq       $40 , %0               \n\t"
            "subq     $40 , %1                 \n\t"
            "jz             2f                         \n\t"

            /* Loop Body */
            ".p2align 4                                \n\t"
            "1:                                  \n\t"
            "vpcmpeqd            %%zmm28, %%zmm20, %%k4            \n\t"         // check - 4
            "vmovupd                    %%zmm20, -320(%5,%0,8)   \n\t"           // SX1
            "vfmadd231pd       %%zmm1 , %%zmm9 ,  %%zmm21        \n\t"           // FX2
            "vfmadd231pd       %%zmm1 , %%zmm9 ,  %%zmm29        \n\t"           // FX2 - dup - 21 - 29
            "vpcmpeqd            %%zmm26, %%zmm18, %%k2            \n\t"         // check - 2
            "vmovupd                    %%zmm18, -192(%6,%0,8)   \n\t"           // SY3
            "vmovupd                            %%zmm23, %%zmm31         \n\t" // BY4
            "vfmsub231pd       %%zmm0 , %%zmm23, %%zmm19         \n\t"           // FY4
            "vfmsub231pd       %%zmm0 , %%zmm23, %%zmm27         \n\t"           // FY4 - dup - 19 - 27
            "vmovupd                            %%zmm16, %%zmm24         \n\t" // BX5
            "vmovupd                            %%zmm16, %%zmm8          \n\t" // BX5'
            "vmovupd               -64(%6,%0,8), %%zmm20         \n\t"           // LY5
            "vmovupd                  (%5,%0,8), %%zmm17         \n\t"           // LX6

            "kandw               %%k4   , %%k2   , %%k4            \n\t"         // reduce - 4 - 2 -> 4
            "vpcmpeqd            %%zmm29, %%zmm21, %%k5            \n\t"         // check - 5
            "vmovupd                    %%zmm21, -256(%5,%0,8)   \n\t"           // SX2
            "vfmadd231pd       %%zmm1 , %%zmm10,  %%zmm22        \n\t"           // FX3
            "vfmadd231pd       %%zmm1 , %%zmm10,  %%zmm30        \n\t"           // FX3 - dup - 22 - 30
            "vpcmpeqd            %%zmm27, %%zmm19, %%k3            \n\t"         // check - 3
            "vmovupd                    %%zmm19, -128(%6,%0,8)   \n\t"           // SY4
            "vmovupd                            %%zmm20, %%zmm28         \n\t" // BY5
            "vfmsub231pd       %%zmm0 , %%zmm20, %%zmm16         \n\t"           // FY5
            "vfmsub231pd       %%zmm0 , %%zmm20, %%zmm24         \n\t"           // FY5 - dup - 16 - 24
            "vmovupd                            %%zmm17, %%zmm25         \n\t" // BX6
            "vmovupd                            %%zmm17, %%zmm9          \n\t" // BX6'
            "vmovupd                  (%6,%0,8), %%zmm21         \n\t"           // LY6
            "kandw               %%k5   , %%k3   , %%k5            \n\t"         // reduce - 5 - 3 -> 5
            "vmovupd                64(%5,%0,8), %%zmm18         \n\t"           // LX7

            "kxnorw              %%k2   , %%k2   , %%k2            \n\t"         // reg-reuse
            "kandw               %%k5   , %%k4   , %%k5            \n\t"         // reduce - 5 - 4 -> 5
            "vpcmpeqd            %%zmm30, %%zmm22, %%k6            \n\t"         // check - 6
            "vmovupd                    %%zmm22, -192(%5,%0,8)   \n\t"           // SX3
            "vfmadd231pd       %%zmm1 , %%zmm11,  %%zmm23        \n\t"           // FX4
            "vfmadd231pd       %%zmm1 , %%zmm11,  %%zmm31        \n\t"           // FX4 - dup - 23 - 31
            "vpcmpeqd            %%zmm24, %%zmm16, %%k0            \n\t"         // check - 0
            "vmovupd                    %%zmm16,  -64(%6,%0,8)   \n\t"           // SY5
            "vmovupd                            %%zmm21, %%zmm29         \n\t" // BY6
            "vfmsub231pd       %%zmm0 , %%zmm21, %%zmm17         \n\t"           // FY6
            "vfmsub231pd       %%zmm0 , %%zmm21, %%zmm25         \n\t"           // FY6 - dup - 17 - 25
            "vmovupd                            %%zmm18, %%zmm26         \n\t" // BX7
            "vmovupd                            %%zmm18, %%zmm10         \n\t" // BX7'
            "vmovupd                64(%6,%0,8), %%zmm22         \n\t"           // LY7
            "kandw               %%k6   , %%k0   , %%k6            \n\t"         // reduce - 6 - 0 -> 6
            "vmovupd               128(%5,%0,8), %%zmm19         \n\t"           // LX8

            "kandw               %%k6   , %%k5   , %%k6            \n\t"         // reduce - 6 - 5 -> 6
            "vpcmpeqd          %%zmm31, %%zmm23, %%k7            \n\t"           // check - 7
            "vmovupd                    %%zmm23, -128(%5,%0,8)   \n\t"           // SX4
            "vfmadd231pd       %%zmm1 , %%zmm8 ,  %%zmm20        \n\t"           // FX5
            "vfmadd231pd       %%zmm1 , %%zmm8 ,  %%zmm28        \n\t"           // FX5 - dup - 20 - 28
            "vpcmpeqd            %%zmm25, %%zmm17, %%k1            \n\t"         // check - 1
            "vmovupd                    %%zmm17,     (%6,%0,8)   \n\t"           // SY6
            "vmovupd                            %%zmm22, %%zmm30         \n\t" // BY7
            "vfmsub231pd       %%zmm0 , %%zmm22, %%zmm18         \n\t"           // FY7
            "vfmsub231pd       %%zmm0 , %%zmm22, %%zmm26         \n\t"           // FY7 - dup - 18 - 26
            "vmovupd                            %%zmm19, %%zmm27         \n\t" // BX8
            "vmovupd                            %%zmm19, %%zmm11         \n\t" // BX8'
            "vmovupd               128(%6,%0,8), %%zmm23         \n\t"           // LY8
            "kandw               %%k7   , %%k1   , %%k7            \n\t"         // reduce - 7 - 1 -> 7
            "vmovupd               192(%5,%0,8), %%zmm16         \n\t"           // LX9

            "kandw               %%k7   , %%k6   , %%k7            \n\t" // reduce - 7 - 6 -> 7

            "addq       $32 , %0               \n\t"

            "kxorw               %%k7   , %%k2   , %%k7            \n\t"
            "ktestw              %%k7   , %%k2                     \n\t"
            "jnz            4f           \n\t"

            "subq     $32 , %1                 \n\t"
            "jnz        1b                       \n\t"

            /* Epilogue */
            "2:                                  \n\t"
            "vpcmpeqd            %%zmm28, %%zmm20, %%k4            \n\t"         // check - 4
            "vmovupd                    %%zmm20, -320(%5,%0,8)   \n\t"           // SX1
            "vfmadd231pd       %%zmm1 , %%zmm9 ,  %%zmm21        \n\t"           // FX2
            "vfmadd231pd       %%zmm1 , %%zmm9 ,  %%zmm29        \n\t"           // FX2 - dup - 21 - 29
            "vpcmpeqd            %%zmm26, %%zmm18, %%k2            \n\t"         // check - 2
            "vmovupd                    %%zmm18, -192(%6,%0,8)   \n\t"           // SY3
            "vmovupd                            %%zmm23, %%zmm31         \n\t" // BY4
            "vfmsub231pd       %%zmm0 , %%zmm23, %%zmm19         \n\t"           // FY4
            "vfmsub231pd       %%zmm0 , %%zmm23, %%zmm27         \n\t"           // FY4 - dup - 19 - 27
            "vmovupd                            %%zmm16, %%zmm24         \n\t" // BX5
            "vmovupd                            %%zmm16, %%zmm8          \n\t" // BX5'
            "vmovupd               -64(%6,%0,8), %%zmm20         \n\t"           // LY5

            "kandw               %%k4   , %%k2   , %%k4            \n\t"         // reduce - 4 - 2 -> 4
            "vpcmpeqd            %%zmm29, %%zmm21, %%k5            \n\t"         // check - 5
            "vmovupd                    %%zmm21, -256(%5,%0,8)   \n\t"           // SX2
            "vfmadd231pd       %%zmm1 , %%zmm10,  %%zmm22        \n\t"           // FX3
            "vfmadd231pd       %%zmm1 , %%zmm10,  %%zmm30        \n\t"           // FX3 - dup - 22 - 30
            "vpcmpeqd            %%zmm27, %%zmm19, %%k3            \n\t"         // check - 3
            "vmovupd                    %%zmm19, -128(%6,%0,8)   \n\t"           // SY4
            "vmovupd                            %%zmm20, %%zmm28         \n\t" // BY5
            "vfmsub231pd       %%zmm0 , %%zmm20, %%zmm16         \n\t"           // FY5
            "vfmsub231pd       %%zmm0 , %%zmm20, %%zmm24         \n\t"           // FY5 - dup - 16 - 24

            "kandw               %%k5   , %%k3   , %%k5            \n\t" // reduce - 5 - 3 -> 5
            "vpcmpeqd            %%zmm30, %%zmm22, %%k6            \n\t" // check - 6
            "vmovupd                    %%zmm22, -192(%5,%0,8)   \n\t"   // SX3
            "vfmadd231pd       %%zmm1 , %%zmm11,  %%zmm23        \n\t"   // FX4
            "vfmadd231pd       %%zmm1 , %%zmm11,  %%zmm31        \n\t"   // FX4 - dup - 23 - 31
            "vpcmpeqd            %%zmm24, %%zmm16, %%k0            \n\t" // check - 0
            "vmovupd                    %%zmm16,  -64(%6,%0,8)   \n\t"   // SY5

            "kandw               %%k6   , %%k0   , %%k6            \n\t" // reduce - 6 - 0 -> 6
            "kandw               %%k5   , %%k4   , %%k5            \n\t" // reduce - 5 - 4 -> 5
            "vpcmpeqd          %%zmm31, %%zmm23, %%k7            \n\t"   // check - 7
            "vmovupd                    %%zmm23, -128(%5,%0,8)   \n\t"   // SX4
            "vfmadd231pd       %%zmm1 , %%zmm8 ,  %%zmm20        \n\t"   // FX5
            "vfmadd231pd       %%zmm1 , %%zmm8 ,  %%zmm28        \n\t"   // FX5 - dup - 20 - 28

            "kandw               %%k6   , %%k5   , %%k6            \n\t" // reduce - 6 - 5 -> 6
            "vpcmpeqd            %%zmm28, %%zmm20, %%k4            \n\t" // check - 4
            "vmovupd                    %%zmm20,  -64(%5,%0,8)   \n\t"   // SX5

            "kandw               %%k7   , %%k4   , %%k7            \n\t" // reduce - 7 - 4 -> 7
            "kxnorw              %%k2   , %%k2   , %%k2            \n\t" // reg-reuse
            "kandw               %%k7   , %%k6   , %%k7            \n\t" // reduce - 7 - 6 -> 7

            "kxorw               %%k7   , %%k2   , %%k7            \n\t"
            "ktestw              %%k7   , %%k2                     \n\t"
            "jnz               5f           \n\t"

            "jmp                 3f                                            \n\t"

            "5:                                      \n\t"
            "addq     $1 , %4                 \n\t"
            "jmp        3f                             \n\t"

            "4:                                      \n\t"
            "addq     $1 , %4                 \n\t"
            "subq     $32 , %1                 \n\t"
            "jnz        1b                       \n\t"
            "jmp        2b                             \n\t"

            "3:                                      \n\t"
            "vzeroupper                            \n\t"
            : "+r"(i),        // 0
              "+r"(n),        // 1
                  "+r"(h11),      // 2
                  "+r"(h22),      // 3
              "+r"(err_num)   // 4
            : "r"(x),         // 5
              "r"(y)          // 6
            : "cc",
              "%xmm0", "%xmm1", "%xmm2", "%xmm3",
              "%xmm4", "%xmm5", "%xmm6", "%xmm7",
              "%xmm8", "%xmm9", "%xmm10", "%xmm11",
              "%xmm12", "%xmm13", "%xmm14", "%xmm15",
              "memory");
      return err_num;
}

int ft_drotm(long n, double *x, long inc_x, double *y, long inc_y, double *param)
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
    n1 = (n1 >= 64) ? n1 - 24 : 0;

    flag = param[0];
    h11 = param[1];
    h21 = param[2];
    h12 = param[3];
    h22 = param[4];

//    printf("flag = %f, n1 = %ld\n", flag, n1);

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
        long err_num = ft_drotm_kernel(n1, x, y, &h11, &h12, &h21, &h22);
        if (err_num) printf("detected error number: %ld\n", err_num);
      }
      else if (flag == 0.0)
      {
        long err_num = ft_drotm_zero_kernel(n1, x, y, &h12, &h21);
        if (err_num) printf("detected error number: %ld\n", err_num);
      }
      else if (flag == 1.0)
      {
        long err_num = ft_drotm_one_kernel(n1, x, y, &h11, &h22);
        if (err_num) printf("detected error number: %ld\n", err_num);
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

void ftblas_drotm_ft(const long int n, const double *x, const long int inc_x, const double *y, const long int inc_y, const double *param){
    ft_drotm(n, (double*)x, inc_x, (double*)y, inc_y, (double*)param);
}