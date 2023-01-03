#include <stdio.h>
#include <stdlib.h>
#include "../include/ftblas.h"

static long drot_kernel(long n, double *x, double *y, double *c, double *s)
{
        long register i = 0;
        long err_num = 0;
        __asm__ __volatile__(
            "vbroadcastsd           (%2), %%zmm0      \n\t" // cos
            "vbroadcastsd           (%3), %%zmm1      \n\t" // sin

            "vmovups                   (%5,%0,8), %%zmm16          \n\t" // LX1

            "vmulpd                 %%zmm1 , %%zmm16, %%zmm24      \n\t" // MX1
			"vmulpd                 %%zmm1 , %%zmm16, %%zmm24      \n\t" // MX1 - dup - 24 - 8
            "vmovups                   (%6,%0,8), %%zmm20          \n\t" // LY1
            "vmovups                 64(%5,%0,8), %%zmm17          \n\t" // LX2
            
            "vfmsub231pd            %%zmm0 , %%zmm20, %%zmm24      \n\t" // FS1
			"vfmsub231pd            %%zmm0 , %%zmm20, %%zmm8       \n\t" // FS1 - dup - 24 - 8
            "vmulpd                 %%zmm1 , %%zmm20, %%zmm28      \n\t" // MY1
			"vmulpd                 %%zmm1 , %%zmm20, %%zmm4       \n\t" // MY1 - dup - 28 - 4
            "vmulpd                 %%zmm1 , %%zmm17, %%zmm25      \n\t" // MX2
			"vmulpd                 %%zmm1 , %%zmm17, %%zmm9       \n\t" // MX2 - dup - 25 - 9
            "vmovups                 64(%6,%0,8), %%zmm21          \n\t" // LY2
            "vmovups                128(%5,%0,8), %%zmm18          \n\t" // LX3

			"vpcmpeqd     		    %%zmm24, %%zmm8 , %%k0         \n\t" // check - 0
            "vmovups                %%zmm24, (%6,%0,8)             \n\t" // SY1
            "vfmsub231pd            %%zmm0 , %%zmm21, %%zmm25      \n\t" // FS2
			"vfmsub231pd            %%zmm0 , %%zmm21, %%zmm9       \n\t" // FS2 - dup - 25 - 9
            "vmulpd                 %%zmm1 , %%zmm21, %%zmm29      \n\t" // MY2
			"vmulpd                 %%zmm1 , %%zmm21, %%zmm5       \n\t" // MY2 - dup - 29 - 5
            "vmulpd                 %%zmm1 , %%zmm18, %%zmm26      \n\t" // MX3
			"vmulpd                 %%zmm1 , %%zmm18, %%zmm10      \n\t" // MX3 - dup - 26 - 10
            "vmovups                128(%6,%0,8), %%zmm22          \n\t" // LY3
            "vmovups                192(%5,%0,8), %%zmm19          \n\t" // LX4
            
            "vfmadd231pd            %%zmm0 , %%zmm16, %%zmm28      \n\t" // FA1
			"vfmadd231pd            %%zmm0 , %%zmm16, %%zmm4       \n\t" // FA1 - dup - 28 - 4
			"vpcmpeqd     		    %%zmm25, %%zmm9 , %%k1         \n\t" // check - 1
            "vmovups                %%zmm25, 64(%6,%0,8)           \n\t" // SY2
            "vfmsub231pd            %%zmm0 , %%zmm22, %%zmm26      \n\t" // FS3
			"vfmsub231pd            %%zmm0 , %%zmm22, %%zmm10      \n\t" // FS3 - dup - 26 - 10
            "vmulpd                 %%zmm1 , %%zmm22, %%zmm30      \n\t" // MY3
			"vmulpd                 %%zmm1 , %%zmm22, %%zmm6       \n\t" // MY3 - dup - 30 - 6
            "vmulpd                 %%zmm1 , %%zmm19, %%zmm27      \n\t" // MX4
			"vmulpd                 %%zmm1 , %%zmm19, %%zmm11      \n\t" // MX4 - dup - 27 - 11
            "vmovups                192(%6,%0,8), %%zmm23          \n\t" // LY4
            "vmovups                256(%5,%0,8), %%zmm16          \n\t" // LX5

            "addq       $40 , %0                 \n\t"
            "subq       $40 , %1                 \n\t"
            "jz         2f                       \n\t"            

            ".p2align 4                                \n\t"
            "1:                                  \n\t"
			"vpcmpeqd         %%zmm28, %%zmm4 , %%k4          \n\t" // check - 4
            "vmovups          %%zmm28, -320(%5,%0,8)          \n\t" // SX1
            "vfmadd231pd      %%zmm0 , %%zmm17, %%zmm29       \n\t" // FA2
			"vfmadd231pd      %%zmm0 , %%zmm17, %%zmm5        \n\t" // FA2 - dup - 29 - 5
            "vpcmpeqd     	  %%zmm26, %%zmm10, %%k2          \n\t" // check - 2
			"vmovups          %%zmm26, -192(%6,%0,8)          \n\t" // SY3
            "vfmsub231pd      %%zmm0 , %%zmm23, %%zmm27       \n\t" // FS4
			"vfmsub231pd      %%zmm0 , %%zmm23, %%zmm11       \n\t" // FS4 - dup - 27 - 11
            "prefetcht0        960(%6, %0, 8)                 \n\t" // PY
            "vmulpd           %%zmm1 , %%zmm23, %%zmm31       \n\t" // MY4
			"vmulpd           %%zmm1 , %%zmm23, %%zmm7        \n\t" // MY4 - dup - 31 - 7
            "prefetcht0       1024(%5, %0, 8)                 \n\t" // PX
            "vmulpd           %%zmm1 , %%zmm16, %%zmm24       \n\t" // MX5
			"vmulpd           %%zmm1 , %%zmm16, %%zmm8        \n\t" // MX5 - dup - 24 - 8
            "vmovups          -64(%6,%0,8), %%zmm20           \n\t" // LY5
			"kandw        	  %%k4   , %%k2   , %%k4          \n\t" // reduce - 4 - 2 -> 4
			"kxnorw   		  %%k2   , %%k2   , %%k2          \n\t" // reg-reuse
            "vmovups             (%5,%0,8), %%zmm17           \n\t" // LX6

			"vpcmpeqd     	  %%zmm29, %%zmm5 , %%k5          \n\t" // check - 5
            "vmovups          %%zmm29, -256(%5,%0,8)          \n\t" // SX2
            "vfmadd231pd      %%zmm0 , %%zmm18, %%zmm30       \n\t" // FA3
			"vfmadd231pd      %%zmm0 , %%zmm18, %%zmm6        \n\t" // FA3 - dup - 30 - 6
            "vpcmpeqd     	  %%zmm27, %%zmm11, %%k3          \n\t" // check - 3
            "vmovups          %%zmm27, -128(%6,%0,8)          \n\t" // SY4
            "vfmsub231pd      %%zmm0 , %%zmm20, %%zmm24       \n\t" // FS5
			"vfmsub231pd      %%zmm0 , %%zmm20, %%zmm8        \n\t" // FS5 - dup - 24 - 8
            "vmulpd           %%zmm1 , %%zmm20, %%zmm28       \n\t" // MY5
			"vmulpd           %%zmm1 , %%zmm20, %%zmm4        \n\t" // MY5 - dup - 28 - 4
            "vmulpd           %%zmm1 , %%zmm17, %%zmm25       \n\t" // MX6
			"vmulpd           %%zmm1 , %%zmm17, %%zmm9        \n\t" // MX6 - dup - 25 - 9
            "vmovups             (%6,%0,8), %%zmm21           \n\t" // LY6
			"kandw        	  %%k5   , %%k3   , %%k5          \n\t" // reduce - 5 - 3 -> 5
            "vmovups           64(%5,%0,8), %%zmm18           \n\t" // LX7

			"vpcmpeqd         %%zmm30, %%zmm6 , %%k6          \n\t" // check - 6
            "vmovups          %%zmm30, -192(%5,%0,8)          \n\t" // SX3
			"kandw        	  %%k5   , %%k4   , %%k5          \n\t" // reduce - 5 - 4 -> 5
            "vfmadd231pd      %%zmm0 , %%zmm19, %%zmm31       \n\t" // FA4
			"vfmadd231pd      %%zmm0 , %%zmm19, %%zmm7        \n\t" // FA4 - dup - 31 - 7
			"vpcmpeqd         %%zmm24, %%zmm8 , %%k0          \n\t" // check - 0
            "vmovups          %%zmm24, -64(%6,%0,8)           \n\t" // SY5
            "vfmsub231pd      %%zmm0 , %%zmm21, %%zmm25       \n\t" // FS6
			"vfmsub231pd      %%zmm0 , %%zmm21, %%zmm9        \n\t" // FS6 - dup - 25 - 9
            "prefetcht0       1088(%6, %0, 8)                 \n\t" // PY
            "vmulpd           %%zmm1 , %%zmm21, %%zmm29       \n\t" // MY6
			"vmulpd           %%zmm1 , %%zmm21, %%zmm5        \n\t" // MY6 - dup - 29 - 5
            "prefetcht0       1152(%5, %0, 8)                 \n\t" // PX
            "vmulpd           %%zmm1 , %%zmm18, %%zmm26       \n\t" // MX7
			"vmulpd           %%zmm1 , %%zmm18, %%zmm10       \n\t" // MX7 - dup - 26 - 10
            "vmovups           64(%6,%0,8), %%zmm22           \n\t" // LY7
			"kandw        	  %%k6   , %%k0   , %%k6          \n\t" // reduce - 6 - 0 -> 6
            "vmovups          128(%5,%0,8), %%zmm19           \n\t" // LX8

			"vpcmpeqd     	  %%zmm31, %%zmm7 , %%k7          \n\t" // check - 7
            "vmovups          %%zmm31, -128(%5,%0,8)          \n\t" // SX4
            "vfmadd231pd      %%zmm0 , %%zmm16, %%zmm28       \n\t" // FA5
			"vfmadd231pd      %%zmm0 , %%zmm16, %%zmm4        \n\t" // FA5 - dup - 28 - 4
			"vpcmpeqd     	  %%zmm25, %%zmm9 , %%k1          \n\t" // check - 1
            "vmovups          %%zmm25,    (%6,%0,8)           \n\t" // SY6
            "vfmsub231pd      %%zmm0 , %%zmm22, %%zmm26       \n\t" // FS7
			"vfmsub231pd      %%zmm0 , %%zmm22, %%zmm10       \n\t" // FS7 - dup - 26 - 10
			"kandw        	  %%k7   , %%k1   , %%k7          \n\t" // reduce - 7 - 1 -> 7
            "vmulpd             %%zmm1 , %%zmm22, %%zmm30     \n\t" // MY7
			"vmulpd             %%zmm1 , %%zmm22, %%zmm6      \n\t" // MY7 - dup - 30 - 6
			"kandw        	  %%k7   , %%k6   , %%k7          \n\t" // reduce - 7 - 6 -> 7
            "vmulpd           %%zmm1 , %%zmm19, %%zmm27       \n\t" // MX8
			"vmulpd           %%zmm1 , %%zmm19, %%zmm11       \n\t" // MX8 - dup - 27 - 11
            "vmovups          128(%6,%0,8), %%zmm23           \n\t" // LY8
			"kandw        	  %%k7   , %%k5   , %%k7          \n\t" // reduce - 7 - 5 -> 7
            "vmovups          192(%5,%0,8), %%zmm16           \n\t" // LX9

			"kxorw        	  %%k7   , %%k2   , %%k7          \n\t"
            "addq                $32 , %0                     \n\t"
			"ktestw       	  %%k7   , %%k2                   \n\t"
			
			"jnz              4f                              \n\t"
            "subq                $32 , %1                     \n\t"
            "jnz              1b                              \n\t"

            "2:                                               \n\t"
			"vpcmpeqd         %%zmm28, %%zmm4 , %%k4          \n\t" // check - 4
            "vmovups          %%zmm28, -320(%5,%0,8)          \n\t" // SX1
            "vfmadd231pd      %%zmm0 , %%zmm17, %%zmm29       \n\t" // FA2
			"vfmadd231pd      %%zmm0 , %%zmm17, %%zmm5        \n\t" // FA2 - dup - 29 - 5
            "vpcmpeqd     	  %%zmm26, %%zmm10, %%k2          \n\t" // check - 2
			"vmovups          %%zmm26, -192(%6,%0,8)          \n\t" // SY3
            "vfmsub231pd      %%zmm0 , %%zmm23, %%zmm27       \n\t" // FS4
			"vfmsub231pd      %%zmm0 , %%zmm23, %%zmm11       \n\t" // FS4 - dup - 27 - 11
            "kxnorw   		  %%k1   , %%k1   , %%k1          \n\t" // reg-reuse
            "vmulpd             %%zmm1 , %%zmm23, %%zmm31     \n\t" // MY4
			"vmulpd             %%zmm1 , %%zmm23, %%zmm7      \n\t" // MY4 - dup - 31 - 7
            "kandw        	  %%k4   , %%k2   , %%k4          \n\t" // reduce - 4 - 2 -> 4
            "vmulpd           %%zmm1 , %%zmm16, %%zmm24       \n\t" // MX5
			"vmulpd           %%zmm1 , %%zmm16, %%zmm8        \n\t" // MX5 - dup - 24 - 8
            "vmovups          -64(%6,%0,8), %%zmm20           \n\t" // LY5
            
			"vpcmpeqd     	  %%zmm29, %%zmm5 , %%k5          \n\t" // check - 5
            "vmovups          %%zmm29, -256(%5,%0,8)          \n\t" // SX2
            "vfmadd231pd      %%zmm0 , %%zmm18, %%zmm30       \n\t" // FA3
			"vfmadd231pd      %%zmm0 , %%zmm18, %%zmm6        \n\t" // FA3 - dup - 30 - 6
            "vpcmpeqd     	  %%zmm27, %%zmm11, %%k3          \n\t" // check - 3
			"vmovups          %%zmm27, -128(%6,%0,8)          \n\t" // SY4
            "vfmsub231pd      %%zmm0 , %%zmm20, %%zmm24       \n\t" // FS5
			"vfmsub231pd      %%zmm0 , %%zmm20, %%zmm8        \n\t" // FS5 - dup - 24 - 8
            "kandw        	  %%k5   , %%k3   , %%k5          \n\t" // reduce - 5 - 3 -> 5
            "vmulpd             %%zmm1 , %%zmm20, %%zmm28     \n\t" // MY5
			"vmulpd             %%zmm1 , %%zmm20, %%zmm4      \n\t" // MY5 - dup - 28 - 4

			"vpcmpeqd         %%zmm30, %%zmm6 , %%k6          \n\t" // check - 6
            "vmovups          %%zmm30, -192(%5,%0,8)          \n\t" // SX3
            "kandw        	  %%k5   , %%k4   , %%k5          \n\t" // reduce - 5 - 4 -> 5
            "vfmadd231pd      %%zmm0 , %%zmm19, %%zmm31       \n\t" // FA4
			"vfmadd231pd      %%zmm0 , %%zmm19, %%zmm7        \n\t" // FA4 - dup - 31 - 7
			"vpcmpeqd         %%zmm24, %%zmm8 , %%k0          \n\t" // check - 0
            "vmovups          %%zmm24,  -64(%6,%0,8)          \n\t" // SY5
            "kandw        	  %%k6   , %%k0   , %%k6          \n\t" // reduce - 6 - 0 -> 6
			"vpcmpeqd     	  %%zmm31, %%zmm7 , %%k7          \n\t" // check - 7
            "vmovups          %%zmm31, -128(%5,%0,8)          \n\t" // SX4
            "kandw        	  %%k6   , %%k7   , %%k6          \n\t" // reduce - 6 - 7 -> 6
            "vfmadd231pd      %%zmm0 , %%zmm16, %%zmm28       \n\t" // FA5
			"vfmadd231pd      %%zmm0 , %%zmm16, %%zmm4        \n\t" // FA5 - dup - 28 - 4

			"vpcmpeqd         %%zmm28, %%zmm4 , %%k4          \n\t" // check - 4
            "vmovups          %%zmm28,  -64(%5,%0,8)          \n\t" // SX5
            "kandw        	  %%k6   , %%k4   , %%k6          \n\t" // reduce - 6 - 4 -> 6

			"kxorw        	  %%k6   , %%k1   , %%k6          \n\t"
			"ktestw       	  %%k6   , %%k1                   \n\t"
			"jnz              5f                              \n\t"

            "jmp              3f                              \n\t"

            "4:                                               \n\t"
            "addq                $1  , %4                     \n\t"
            "subq                $32 , %1                     \n\t"
            "jnz              1b                              \n\t"
            "jmp              2b                              \n\t"

            "5:                                               \n\t"
            "addq                $1  , %4                     \n\t"

            "3:                                  \n\t"
            "vzeroupper                   \n\t"

            : "+r"(i), // 0
              "+r"(n), // 1
              "+r"(c), // 2
              "+r"(s), // 3
              "+r"(err_num) // 4
            : "r"(x),  // 5
              "r"(y)   // 6
            : "cc",
              "%xmm0", "%xmm1", "%xmm2", "%xmm3",
              "%xmm4", "%xmm5", "%xmm6", "%xmm7",
              "%xmm8", "%xmm9", "%xmm10", "%xmm11",
              "%xmm12", "%xmm13", "%xmm14", "%xmm15",
              "memory");
        return err_num;
}

int ft_drot(long n, double *x, long inc_x, double *y, long inc_y, double c, double s)
{
  long i = 0;
  long ix = 0, iy = 0;

  double temp;

  if (n <= 0)
    return (0);

  if ((inc_x == 1) && (inc_y == 1)) {

    long n1 = n & -32;
    n1 = (n1 >= 64) ? n1 - 24 : 0;
    if (n1 > 0) {
      //printf("n1 = %d, n = %d\n", n1, n);
      double cosa, sina;
      cosa = c;
      sina = s;
      long err_num = drot_kernel(n1, x, y, &cosa, &sina);
      if (err_num) printf("detected %ld error\n", err_num);
      
    }
    i = n1;

    double old_x, old_y;

    while (i < n) {
      old_x = x[i];
      old_y = y[i];
      x[i] = c * old_x + s * old_y;
      y[i] = c * old_y - s * old_x;
//      printf("%f = %5.2f * %5.2f + %5.2f * %5.2f\n", x[i], c, old_x, s, old_y);
//      printf("%f = %5.2f * %5.2f - %5.2f * %5.2f\n", y[i], c, old_y, s, old_x);

      i++;

    }

  } else {
    long inc = -1 * inc_x;
    long n1 = n & inc;
    while (i < n1) {
//      printf("n = %d, n1 = %d, i = %d, BEFORE: y[%d] = %f, x[%d] = %f, AFTER: ", n, n1, i, iy, y[iy], ix, x[ix]);
      temp = c * x[ix] + s * y[iy];
      y[iy] = c * y[iy] - s * x[ix];
      x[ix] = temp;
//      printf("y[%d] = %f, x[%d] = %f\n", iy, y[iy], ix, x[ix]);
      ix += inc_x;
      iy += inc_y;
//      i++;
      i -= inc;
    }

  }
  return (0);

}

void ftblas_drot_ft(const long int n, const double *x, const long int inc_x, const double *y, const long int inc_y, const double c, const double s){
    ft_drot(n, (double*)x, inc_x, (double*)y, inc_y, c, s);
}