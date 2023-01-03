#include <stdio.h>
#include <stdlib.h>
#include "../include/ftblas.h"

static void ori_drot_kernel(long n, double *x, double *y, double *c, double *s)
{

      long register i = 0;

      __asm__ __volatile__(
            "vbroadcastsd           (%2), %%zmm0      \n\t" // cos
            "vbroadcastsd           (%3), %%zmm1      \n\t" // sin

            "vmovups                   (%4,%0,8), %%zmm16          \n\t" // LX1

            "vmulpd                 %%zmm1 , %%zmm16, %%zmm24              \n\t" // MX1
            "vmovups                   (%5,%0,8), %%zmm20          \n\t" // LY1
            "vmovups                 64(%4,%0,8), %%zmm17          \n\t" // LX2
            
            "vfmsub231pd            %%zmm0 , %%zmm20, %%zmm24              \n\t" // FS1
            "vmulpd                 %%zmm1 , %%zmm20, %%zmm28              \n\t" // MY1
            "vmulpd                 %%zmm1 , %%zmm17, %%zmm25              \n\t" // MX2
            "vmovups                 64(%5,%0,8), %%zmm21          \n\t" // LY2
            "vmovups                128(%4,%0,8), %%zmm18          \n\t" // LX3

            "vmovups                             %%zmm24, (%5,%0,8)        \n\t" // SY1
            "vfmsub231pd            %%zmm0 , %%zmm21, %%zmm25              \n\t" // FS2
            "vmulpd                 %%zmm1 , %%zmm21, %%zmm29              \n\t" // MY2
            "vmulpd                 %%zmm1 , %%zmm18, %%zmm26              \n\t" // MX3
            "vmovups                128(%5,%0,8), %%zmm22          \n\t" // LY3
            "vmovups                192(%4,%0,8), %%zmm19          \n\t" // LX4
            
            "vfmadd231pd            %%zmm0 , %%zmm16, %%zmm28              \n\t" // FA1
            "vmovups                             %%zmm25, 64(%5,%0,8)      \n\t" // SY2
            "vfmsub231pd            %%zmm0 , %%zmm22, %%zmm26              \n\t" // FS3
            "vmulpd               %%zmm1 , %%zmm22, %%zmm30                \n\t" // MY3
            "vmulpd                 %%zmm1 , %%zmm19, %%zmm27              \n\t" // MX4
            "vmovups                192(%5,%0,8), %%zmm23          \n\t" // LY4
            "vmovups                256(%4,%0,8), %%zmm16          \n\t" // LX5

            "addq       $40 , %0               \n\t"
            "subq     $40 , %1                 \n\t"
            "jz         2f                       \n\t"            

            ".p2align 4                                \n\t"
            "1:                                  \n\t"
            "vmovups          %%zmm28, -320(%4,%0,8)          \n\t" // SX1
            "vfmadd231pd      %%zmm0 , %%zmm17, %%zmm29       \n\t" // FA2
            "vmovups          %%zmm26, -192(%5,%0,8)          \n\t" // SY3
            "vfmsub231pd      %%zmm0 , %%zmm23, %%zmm27       \n\t" // FS4
            "prefetcht0        960(%5, %0, 8)                 \n\t" // PY
            "vmulpd             %%zmm1 , %%zmm23, %%zmm31     \n\t" // MY4
            "prefetcht0       1024(%4, %0, 8)                 \n\t" // PX
            "vmulpd           %%zmm1 , %%zmm16, %%zmm24       \n\t" // MX5
            "vmovups          -64(%5,%0,8), %%zmm20           \n\t" // LY5
            "vmovups             (%4,%0,8), %%zmm17           \n\t" // LX6

            "vmovups          %%zmm29, -256(%4,%0,8)          \n\t" // SX2
            "vfmadd231pd      %%zmm0 , %%zmm18, %%zmm30       \n\t" // FA3
            "vmovups          %%zmm27, -128(%5,%0,8)          \n\t" // SY4
            "vfmsub231pd      %%zmm0 , %%zmm20, %%zmm24       \n\t" // FS5
            "vmulpd             %%zmm1 , %%zmm20, %%zmm28     \n\t" // MY5
            "vmulpd           %%zmm1 , %%zmm17, %%zmm25       \n\t" // MX6
            "vmovups             (%5,%0,8), %%zmm21           \n\t" // LY6
            "vmovups           64(%4,%0,8), %%zmm18           \n\t" // LX7

            "vmovups          %%zmm30, -192(%4,%0,8)          \n\t" // SX3
            "vfmadd231pd      %%zmm0 , %%zmm19, %%zmm31       \n\t" // FA4
            "vmovups          %%zmm24, -64(%5,%0,8)           \n\t" // SY5
            "vfmsub231pd      %%zmm0 , %%zmm21, %%zmm25       \n\t" // FS6
            "prefetcht0       1088(%5, %0, 8)                 \n\t" // PY
            "vmulpd             %%zmm1 , %%zmm21, %%zmm29     \n\t" // MY6
            "prefetcht0       1152(%5, %0, 8)                 \n\t" // PY
            "vmulpd           %%zmm1 , %%zmm18, %%zmm26       \n\t" // MX7
            "vmovups           64(%5,%0,8), %%zmm22           \n\t" // LY7
            "vmovups          128(%4,%0,8), %%zmm19           \n\t" // LX8

            "vmovups          %%zmm31, -128(%4,%0,8)          \n\t" // SX4
            "vfmadd231pd      %%zmm0 , %%zmm16, %%zmm28       \n\t" // FA5
            "vmovups          %%zmm25,    (%5,%0,8)           \n\t" // SY6
            "vfmsub231pd      %%zmm0 , %%zmm22, %%zmm26       \n\t" // FS7
            "vmulpd             %%zmm1 , %%zmm22, %%zmm30     \n\t" // MY7
            "vmulpd           %%zmm1 , %%zmm19, %%zmm27       \n\t" // MX8
            "vmovups          128(%5,%0,8), %%zmm23           \n\t" // LY8
            "vmovups          192(%4,%0,8), %%zmm16           \n\t" // LX9

            "addq       $32 , %0               \n\t"
            "subq     $32 , %1                 \n\t"
            "jnz        1b                       \n\t"

            "2:                                  \n\t"
            "vmovups          %%zmm28, -320(%4,%0,8)          \n\t" // SX1
            "vfmadd231pd      %%zmm0 , %%zmm17, %%zmm29       \n\t" // FA2
            "vmovups          %%zmm26, -192(%5,%0,8)          \n\t" // SY3
            "vfmsub231pd      %%zmm0 , %%zmm23, %%zmm27       \n\t" // FS4
            "vmulpd             %%zmm1 , %%zmm23, %%zmm31     \n\t" // MY4
            "vmulpd           %%zmm1 , %%zmm16, %%zmm24       \n\t" // MX5
            "vmovups          -64(%5,%0,8), %%zmm20           \n\t" // LY5
            
            "vmovups          %%zmm29, -256(%4,%0,8)          \n\t" // SX2
            "vfmadd231pd      %%zmm0 , %%zmm18, %%zmm30       \n\t" // FA3
            "vmovups          %%zmm27, -128(%5,%0,8)          \n\t" // SY4
            "vfmsub231pd      %%zmm0 , %%zmm20, %%zmm24       \n\t" // FS5
            "vmulpd             %%zmm1 , %%zmm20, %%zmm28     \n\t" // MY5

            "vmovups          %%zmm30, -192(%4,%0,8)          \n\t" // SX3
            "vfmadd231pd      %%zmm0 , %%zmm19, %%zmm31       \n\t" // FA4
            "vmovups          %%zmm24,  -64(%5,%0,8)          \n\t" // SY5

            "vmovups          %%zmm31, -128(%4,%0,8)          \n\t" // SX4
            "vfmadd231pd      %%zmm0 , %%zmm16, %%zmm28       \n\t" // FA5

            "vmovups          %%zmm28,  -64(%4,%0,8)          \n\t" // SX5

            "3:                                  \n\t"
            "vzeroupper                   \n\t"

            : "+r"(i), // 0
              "+r"(n), // 1
              "+r"(c), // 2
              "+r"(s)  // 3
            : "r"(x),  //4
              "r"(y)   // 5
            : "cc",
              "%xmm0", "%xmm1", "%xmm2", "%xmm3",
            "%xmm4", "%xmm5", "%xmm6", "%xmm7",
            "%xmm8", "%xmm9", "%xmm10", "%xmm11",
            "%xmm12", "%xmm13", "%xmm14", "%xmm15",
              "memory");
}

int ori_drot(long n, double *x, long inc_x, double *y, long inc_y, double c, double s)
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
      ori_drot_kernel(n1, x, y, &cosa, &sina);
      
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

void ftblas_drot_ori(const long int n, const double *x, const long int inc_x, const double *y, const long int inc_y, const double c, const double s){
    ori_drot(n, (double*)x, inc_x, (double*)y, inc_y, c, s);
}