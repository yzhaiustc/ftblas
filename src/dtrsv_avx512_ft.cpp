#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "immintrin.h"
#include <unistd.h>
#include "../include/ftblas.h"
#define MAX(a, b) (((a) > (b)) ? (a) : (b))

static long ft_dgemv_n_kernel_trsv(long int n, double *a1, double *a2, double *a3, double *a4, double *x, double *y)
{
    register long int i = 0;
    register long err_num = 0;
    __asm__ __volatile__(
        "vbroadcastsd           (%3) , %%zmm4                    \n\t" // x1
        "vbroadcastsd      8(%3) , %%zmm5                    \n\t" // x2
        "vbroadcastsd     16(%3) , %%zmm6                    \n\t" // x3
        "vbroadcastsd     24(%3) , %%zmm7                    \n\t" // x4
        "kxnorw             %%k0 , %%k0 , %%k0               \n\t"

        // Prologue
        "vmovups    (%4,%0,8), %%zmm8                       \n\t" // LY1

        "vmovups      %%zmm8 , %%zmm27                      \n\t" // BY1 - 8 - 27
        "vmovups      %%zmm8 , %%zmm22                      \n\t" // BY1 - 8 - 22
        "vmovups    (%5,%0,8), %%zmm16                      \n\t" // LA11
        "vmovups  64(%4,%0,8), %%zmm9                       \n\t" // LY2

        "vfmadd231pd  %%zmm16, %%zmm4 , %%zmm27             \n\t" // F11
        "vfmadd231pd  %%zmm16, %%zmm4 , %%zmm8              \n\t" // F11 - dup - 8 - 27
        "vmovups        (%6,%0,8), %%zmm17                      \n\t" // LA21
        "vmovups      %%zmm9 , %%zmm28                      \n\t" // BY2 - 9 - 28
        "vmovups      %%zmm9 , %%zmm23                      \n\t" // BY2 - 9 - 23
        "vmovups  64(%5,%0,8), %%zmm16                      \n\t" // LA12
        "vmovups 128(%4,%0,8), %%zmm10                      \n\t" // LY3

        "vfmadd231pd  %%zmm17, %%zmm5 , %%zmm27             \n\t" // F21
        "vfmadd231pd  %%zmm17, %%zmm5 , %%zmm8              \n\t" // F21 - dup - 8 - 27
        "vmovups        (%7,%0,8), %%zmm18                      \n\t" // LA31
        "vfmadd231pd  %%zmm16, %%zmm4 , %%zmm28             \n\t" // F12
        "vfmadd231pd  %%zmm16, %%zmm4 , %%zmm9              \n\t" // F12 - dup - 9 - 28
        "vmovups  64(%6,%0,8), %%zmm17                      \n\t" // LA22
        "vmovups      %%zmm10, %%zmm29                      \n\t" // BY3 - 10 - 29
        "vmovups      %%zmm10, %%zmm24                      \n\t" // BY3 - 10 - 24
        "vmovups 128(%5,%0,8), %%zmm16                      \n\t" // LA13
        "vmovups 192(%4,%0,8), %%zmm11                      \n\t" // LY4

        "vfmadd231pd  %%zmm18, %%zmm6 , %%zmm27             \n\t" // F31
        "vfmadd231pd  %%zmm18, %%zmm6 , %%zmm8              \n\t" // F31 - dup - 8 - 27
        "vmovups        (%8,%0,8), %%zmm19                      \n\t" // LA41
        "vfmadd231pd  %%zmm17, %%zmm5 , %%zmm28             \n\t" // F22
        "vfmadd231pd  %%zmm17, %%zmm5 , %%zmm9              \n\t" // F22 - dup - 9 - 28
        "vmovups  64(%7,%0,8), %%zmm18                      \n\t" // LA32
        "vfmadd231pd  %%zmm16, %%zmm4 , %%zmm29             \n\t" // F13
        "vfmadd231pd  %%zmm16, %%zmm4 , %%zmm10             \n\t" // F13 - dup - 10 - 29
        "vmovups 128(%6,%0,8), %%zmm17                      \n\t" // LA23
        "vmovups      %%zmm11, %%zmm30                      \n\t" // BY4 - 11 - 30
        "vmovups      %%zmm11, %%zmm25                      \n\t" // BY4 - 11 - 25
        "vmovups 192(%5,%0,8), %%zmm16                      \n\t" // LA14
        "vmovups 256(%4,%0,8), %%zmm12                      \n\t" // LY5

        "vfmadd231pd  %%zmm19, %%zmm7 , %%zmm27             \n\t" // F41
        "vfmadd231pd  %%zmm19, %%zmm7 , %%zmm8              \n\t" // F41 - dup - 8 - 27
        "vfmadd231pd  %%zmm18, %%zmm6 , %%zmm28             \n\t" // F32
        "vfmadd231pd  %%zmm18, %%zmm6 , %%zmm9              \n\t" // F32 - dup - 9 - 28
        "vmovups  64(%8,%0,8), %%zmm19                      \n\t" // LA42
        "vfmadd231pd  %%zmm17, %%zmm5 , %%zmm29             \n\t" // F23
        "vfmadd231pd  %%zmm17, %%zmm5 , %%zmm10             \n\t" // F23 - dup - 10 - 29
        "vmovups 128(%7,%0,8), %%zmm18                      \n\t" // LA33
        "vfmadd231pd  %%zmm16, %%zmm4 , %%zmm30             \n\t" // F14
        "vfmadd231pd  %%zmm16, %%zmm4 , %%zmm11             \n\t" // F14 - dup - 11 - 30
        "vmovups 192(%6,%0,8), %%zmm17                      \n\t" // LA24
        "vmovups      %%zmm12, %%zmm31                      \n\t" // BY5 - 12 - 31
        "vmovups      %%zmm12, %%zmm26                      \n\t" // BY5 - 12 - 26
        "vmovups 256(%5,%0,8), %%zmm16                      \n\t" // LA15
        "vpcmpeqd     %%zmm8 , %%zmm27, %%k1                \n\t" // C1
        "vmovups 320(%4,%0,8), %%zmm8                       \n\t" // LY6

        "addq           $48, %0                                         \n\t"
        "subq           $48, %1                                         \n\t"
        "jz             2f                                              \n\t"

        // Loop Body
        ".p2align 4                                         \n\t"
        "1:                                                             \n\t"
        "vmovups       %%zmm22, %%zmm0                      \n\t" // BY1 - 22 - 0
        "vmovups       %%zmm27, -384(%4,%0,8)               \n\t" // S1
        "vfmadd231pd   %%zmm19, %%zmm7 , %%zmm28            \n\t" // F42
        "vfmadd231pd   %%zmm19, %%zmm7 , %%zmm9             \n\t" // F42 - dup - 9 - 28
        "vfmadd231pd   %%zmm18, %%zmm6 , %%zmm29            \n\t" // F33
        "vfmadd231pd   %%zmm18, %%zmm6 , %%zmm10            \n\t" // F33 - dup - 10 - 29
        "vmovups -256(%8,%0,8), %%zmm19                     \n\t" // A43
        "vfmadd231pd   %%zmm17, %%zmm5 , %%zmm30            \n\t" // F24
        "vfmadd231pd   %%zmm17, %%zmm5 , %%zmm11            \n\t" // F24 - dup - 11 - 30
        "vmovups -192(%7,%0,8), %%zmm18                     \n\t" // A34
        "vfmadd231pd   %%zmm16, %%zmm4 , %%zmm31            \n\t" // F15
        "vfmadd231pd   %%zmm16, %%zmm4 , %%zmm12            \n\t" // F15 - dup - 12 - 31
        "vmovups -128(%6,%0,8), %%zmm17                     \n\t" // A25
        "vmovups       %%zmm8 , %%zmm27                     \n\t" // BY6 - 8 - 27
        "vmovups       %%zmm8 , %%zmm22                     \n\t" // BY6 - 8 - 22
        "vmovups  -64(%5,%0,8), %%zmm16                     \n\t" // LA16
        "vpcmpeqd     %%zmm9 , %%zmm28, %%k2                \n\t" // C2
        "vmovups     (%4,%0,8), %%zmm9                      \n\t" // LY7

        "vmovups       %%zmm23, %%zmm1                      \n\t" // BY2 - 23 - 1
        "vmovups       %%zmm28, -320(%4,%0,8)               \n\t" // S2
        "vfmadd231pd   %%zmm19, %%zmm7 , %%zmm29            \n\t" // F43
        "vfmadd231pd   %%zmm19, %%zmm7 , %%zmm10            \n\t" // F43 - dup - 10 - 29
        "kandw         %%k1   , %%k2   , %%k2               \n\t" // red - 1 - 2 -> 2
        "vfmadd231pd   %%zmm18, %%zmm6 , %%zmm30            \n\t" // F34
        "vfmadd231pd   %%zmm18, %%zmm6 , %%zmm11            \n\t" // F34 - dup - 11 - 30
        "vmovups -192(%8,%0,8), %%zmm19                     \n\t" // A44
        "vfmadd231pd   %%zmm17, %%zmm5 , %%zmm31            \n\t" // F25
        "vfmadd231pd   %%zmm17, %%zmm5 , %%zmm12            \n\t" // F25 - dup - 12 - 31
        "vmovups -128(%7,%0,8), %%zmm18                     \n\t" // A35
        "vfmadd231pd   %%zmm16, %%zmm4 , %%zmm27            \n\t" // F16
        "vfmadd231pd   %%zmm16, %%zmm4 , %%zmm8             \n\t" // F16 - dup - 8 - 27
        "vmovups  -64(%6,%0,8), %%zmm17                     \n\t" // A26
        "vmovups       %%zmm9 , %%zmm28                     \n\t" // BY7 - 9 - 28
        "vmovups       %%zmm9 , %%zmm23                     \n\t" // BY7 - 9 - 23
        "vmovups     (%5,%0,8), %%zmm16                     \n\t" // LA17
        "vpcmpeqd      %%zmm10, %%zmm29, %%k3               \n\t" // C3
        "vmovups   64(%4,%0,8), %%zmm10                     \n\t" // LY8

        "vmovups       %%zmm24, %%zmm2                      \n\t" // BY3 - 24 - 2
        "vmovups       %%zmm29, -256(%4,%0,8)               \n\t" // S3
        "vfmadd231pd   %%zmm19, %%zmm7 , %%zmm30            \n\t" // F44
        "vfmadd231pd   %%zmm19, %%zmm7 , %%zmm11            \n\t" // F44 - dup - 11 - 30
        "kandw         %%k2   , %%k3   , %%k3               \n\t" // red - 2 - 3 -> 3
        "vfmadd231pd   %%zmm18, %%zmm6 , %%zmm31            \n\t" // F35
        "vfmadd231pd   %%zmm18, %%zmm6 , %%zmm12            \n\t" // F35 - dup - 12 - 31
        "vmovups -128(%8,%0,8), %%zmm19                     \n\t" // A45
        "vfmadd231pd   %%zmm17, %%zmm5 , %%zmm27            \n\t" // F26
        "vfmadd231pd   %%zmm17, %%zmm5 , %%zmm8             \n\t" // F26 - dup - 8 - 27
        "vmovups  -64(%7,%0,8), %%zmm18                     \n\t" // A36
        "vfmadd231pd   %%zmm16, %%zmm4 , %%zmm28            \n\t" // F17
        "vfmadd231pd   %%zmm16, %%zmm4 , %%zmm9             \n\t" // F17 - dup - 9 - 28
        "vmovups     (%6,%0,8), %%zmm17                     \n\t" // A27
        "vmovups       %%zmm10, %%zmm29                     \n\t" // BY8 - 10 - 29
        "vmovups       %%zmm10, %%zmm24                     \n\t" // BY8 - 10 - 24
        "vmovups   64(%5,%0,8), %%zmm16                     \n\t" // LA18
        "vpcmpeqd      %%zmm11, %%zmm30, %%k4               \n\t" // C4
        "vmovups  128(%4,%0,8), %%zmm11                     \n\t" // LY9

        "vmovups       %%zmm25, %%zmm3                      \n\t" // BY4 - 25 - 3
        "vmovups       %%zmm30, -192(%4,%0,8)               \n\t" // S4
        "vfmadd231pd   %%zmm19, %%zmm7 , %%zmm31            \n\t" // F45
        "vfmadd231pd   %%zmm19, %%zmm7 , %%zmm12            \n\t" // F45 - dup - 12 - 31
        "kandw         %%k3   , %%k4   , %%k4               \n\t" // red - 3 - 4 -> 4
        "vfmadd231pd   %%zmm18, %%zmm6 , %%zmm27            \n\t" // F36
        "vfmadd231pd   %%zmm18, %%zmm6 , %%zmm8             \n\t" // F36 - dup - 8 - 27
        "vmovups  -64(%8,%0,8), %%zmm19                     \n\t" // A46
        "vfmadd231pd   %%zmm17, %%zmm5 , %%zmm28            \n\t" // F27
        "vfmadd231pd   %%zmm17, %%zmm5 , %%zmm9             \n\t" // F27 - dup - 9 - 28
        "vmovups     (%7,%0,8), %%zmm18                     \n\t" // A37
        "vfmadd231pd   %%zmm16, %%zmm4 , %%zmm29            \n\t" // F18
        "vfmadd231pd   %%zmm16, %%zmm4 , %%zmm10            \n\t" // F18 - dup - 10 - 29
        "vmovups   64(%6,%0,8), %%zmm17                     \n\t" // A28
        "vmovups       %%zmm11, %%zmm30                     \n\t" // BY9 - 11 - 30
        "vmovups       %%zmm11, %%zmm25                     \n\t" // BY9 - 11 - 25
        "vmovups  128(%5,%0,8), %%zmm16                     \n\t" // LA19
        "vpcmpeqd      %%zmm12, %%zmm31, %%k5               \n\t" // C5
        "vmovups  192(%4,%0,8), %%zmm12                     \n\t" // LY10

        "vmovups       %%zmm26, %%zmm13                     \n\t" // BY5 - 26 - 13
        "vmovups       %%zmm31, -128(%4,%0,8)               \n\t" // S5
        "vfmadd231pd   %%zmm19, %%zmm7 , %%zmm27            \n\t" // F46
        "vfmadd231pd   %%zmm19, %%zmm7 , %%zmm8             \n\t" // F46 - dup - 8 - 27
        "kandw         %%k4   , %%k5   , %%k5               \n\t" // red - 4 - 5 -> 5
        "vfmadd231pd   %%zmm18, %%zmm6 , %%zmm28            \n\t" // F37
        "vfmadd231pd   %%zmm18, %%zmm6 , %%zmm9             \n\t" // F37 - dup - 9 - 28
        "vmovups     (%8,%0,8), %%zmm19                     \n\t" // A47
        "vfmadd231pd   %%zmm17, %%zmm5 , %%zmm29            \n\t" // F28
        "vfmadd231pd   %%zmm17, %%zmm5 , %%zmm10            \n\t" // F28 - dup - 10 - 29
        "vmovups   64(%7,%0,8), %%zmm18                     \n\t" // A38
        "vfmadd231pd   %%zmm16, %%zmm4 , %%zmm30            \n\t" // F19
        "vfmadd231pd   %%zmm16, %%zmm4 , %%zmm11            \n\t" // F19 - dup - 11 - 30
        "kxorw         %%k0   , %%k5   , %%k5               \n\t"
        "vmovups  128(%6,%0,8), %%zmm17                     \n\t" // A29
        "vmovups       %%zmm12, %%zmm31                     \n\t" // BY10 - 12 - 31
        "vmovups       %%zmm12, %%zmm26                     \n\t" // BY10 - 12 - 26
        "vmovups  192(%5,%0,8), %%zmm16                     \n\t" // LA10
        "vpcmpeqd      %%zmm8 , %%zmm27, %%k1               \n\t" // C1
        "vmovups  256(%4,%0,8), %%zmm8                      \n\t" // LY11

        "addq          $40    , %0                          \n\t"
        "ktestw        %%k0   , %%k5                        \n\t"
        "jnz           3f                                   \n\t"
        "subq          $40    , %1                          \n\t"
        "jnz           1b                                   \n\t"


        "2:                                                 \n\t"
        "vmovups       %%zmm22, %%zmm0                      \n\t" // BY1 - 22 - 0
        "vmovups       %%zmm27, -384(%4,%0,8)               \n\t" // S1
        "vfmadd231pd   %%zmm19, %%zmm7 , %%zmm28            \n\t" // F42
        "vfmadd231pd   %%zmm19, %%zmm7 , %%zmm9             \n\t" // F42 - dup - 9 - 28
        "vfmadd231pd   %%zmm18, %%zmm6 , %%zmm29            \n\t" // F33
        "vfmadd231pd   %%zmm18, %%zmm6 , %%zmm10            \n\t" // F33 - dup - 10 - 29
        "vmovups -256(%8,%0,8), %%zmm19                     \n\t" // A43
        "vfmadd231pd   %%zmm17, %%zmm5 , %%zmm30            \n\t" // F24
        "vfmadd231pd   %%zmm17, %%zmm5 , %%zmm11            \n\t" // F24 - dup - 11 - 30
        "vmovups -192(%7,%0,8), %%zmm18                     \n\t" // A34
        "vfmadd231pd   %%zmm16, %%zmm4 , %%zmm31            \n\t" // F15
        "vfmadd231pd   %%zmm16, %%zmm4 , %%zmm12            \n\t" // F15 - dup - 12 - 31
        "vmovups -128(%6,%0,8), %%zmm17                     \n\t" // A25
        "vmovups       %%zmm8 , %%zmm27                     \n\t" // BY6 - 8 - 27
        "vmovups       %%zmm8 , %%zmm22                     \n\t" // BY6 - 8 - 22
        "vmovups  -64(%5,%0,8), %%zmm16                     \n\t" // LA16

        "vmovups       %%zmm23, %%zmm1                      \n\t" // BY2 - 23 - 1
        "vpcmpeqd      %%zmm9 , %%zmm28, %%k2               \n\t" // C2
        "vmovups       %%zmm28, -320(%4,%0,8)               \n\t" // S2
        "vfmadd231pd   %%zmm19, %%zmm7 , %%zmm29            \n\t" // F43
        "vfmadd231pd   %%zmm19, %%zmm7 , %%zmm10            \n\t" // F43 - dup - 10 - 29
        "vfmadd231pd   %%zmm18, %%zmm6 , %%zmm30            \n\t" // F34
        "vfmadd231pd   %%zmm18, %%zmm6 , %%zmm11            \n\t" // F34 - dup - 11 - 30
        "kandw         %%k1   , %%k2   , %%k2               \n\t" // red - 1 - 2 -> 2
        "vmovups -192(%8,%0,8), %%zmm19                     \n\t" // A44
        "vfmadd231pd   %%zmm17, %%zmm5 , %%zmm31            \n\t" // F25
        "vfmadd231pd   %%zmm17, %%zmm5 , %%zmm12            \n\t" // F25 - dup - 12 - 31
        "vmovups -128(%7,%0,8), %%zmm18                     \n\t" // A35
        "vfmadd231pd   %%zmm16, %%zmm4 , %%zmm27            \n\t" // F16
        "vfmadd231pd   %%zmm16, %%zmm4 , %%zmm8             \n\t" // F16 - dup - 8 - 27
        "vmovups  -64(%6,%0,8), %%zmm17                     \n\t" // A26

        "vmovups       %%zmm24, %%zmm2                      \n\t" // BY3 - 24 - 2
        "vpcmpeqd      %%zmm10, %%zmm29, %%k3               \n\t" // C3
        "vmovups       %%zmm29, -256(%4,%0,8)               \n\t" // S3
        "vfmadd231pd   %%zmm19, %%zmm7 , %%zmm30            \n\t" // F44
        "vfmadd231pd   %%zmm19, %%zmm7 , %%zmm11            \n\t" // F44 - dup - 11 - 30
        "vfmadd231pd   %%zmm18, %%zmm6 , %%zmm31            \n\t" // F35
        "vfmadd231pd   %%zmm18, %%zmm6 , %%zmm12            \n\t" // F35 - dup - 12 - 31
        "kandw         %%k2   , %%k3   , %%k3               \n\t" // red - 2 - 3 -> 3
        "vmovups -128(%8,%0,8), %%zmm19                     \n\t" // A45
        "vfmadd231pd   %%zmm17, %%zmm5 , %%zmm27            \n\t" // F26
        "vfmadd231pd   %%zmm17, %%zmm5 , %%zmm8             \n\t" // F26 - dup - 8 - 27
        "vmovups  -64(%7,%0,8), %%zmm18                     \n\t" // A36

        "vmovups       %%zmm25, %%zmm3                      \n\t" // BY4 - 25 - 3
        "vpcmpeqd      %%zmm11, %%zmm30, %%k4               \n\t" // C4
        "vmovups       %%zmm30, -192(%4,%0,8)               \n\t" // S4
        "vfmadd231pd   %%zmm19, %%zmm7 , %%zmm31            \n\t" // F45
        "vfmadd231pd   %%zmm19, %%zmm7 , %%zmm12            \n\t" // F45 - dup - 12 - 31
        "vfmadd231pd   %%zmm18, %%zmm6 , %%zmm27            \n\t" // F36
        "vfmadd231pd   %%zmm18, %%zmm6 , %%zmm8             \n\t" // F36 - dup - 8 - 27
        "kandw         %%k3   , %%k4   , %%k4               \n\t" // red - 3 - 4 -> 4
        "vmovups  -64(%8,%0,8), %%zmm19                     \n\t" // A46

        "vmovups       %%zmm26, %%zmm13                     \n\t" // BY5 - 26 - 13
        "vpcmpeqd      %%zmm12, %%zmm31, %%k5               \n\t" // C5
        "vmovups       %%zmm31, -128(%4,%0,8)               \n\t" // S5
        "vfmadd231pd   %%zmm19, %%zmm7 , %%zmm27            \n\t" // F46
        "vfmadd231pd   %%zmm19, %%zmm7 , %%zmm8             \n\t" // F46 - dup - 8 - 27
        "kandw         %%k4   , %%k5   , %%k5               \n\t" // red - 4 - 5 -> 5

        "vmovups       %%zmm22, %%zmm15                     \n\t" // BY1 - 22 - 15
        "vpcmpeqd      %%zmm8 , %%zmm27, %%k1               \n\t" // C1
        "vmovups       %%zmm27, -64(%4,%0,8)                \n\t" // S1
        
        "kandw         %%k1   , %%k5   , %%k5               \n\t" // red - 3 - 4 -> 4
        "kxorw         %%k0   , %%k5   , %%k5               \n\t"
        "ktestw        %%k0   , %%k5                        \n\t"
        "jnz           4f                                   \n\t"

        "jmp           6f                                   \n\t"


        "3:                                                 \n\t"
        "addq          $1     , %2                          \n\t"
        // recover from prologue
        "subq          $40    , %0                          \n\t"
        "vmovups       %%zmm0 , %%zmm8                      \n\t" // restore from chpt - 0 - 8

        "vmovups      %%zmm8 , %%zmm27                      \n\t" // BY1 - 8 - 27
        "vmovups      %%zmm8 , %%zmm22                      \n\t" // BY1 - 8 - 22
        "vmovups -384(%5,%0,8), %%zmm16                     \n\t" // LA11
        "vmovups       %%zmm1 , %%zmm9                      \n\t" // restore from chpt - 1 - 9

        "vfmadd231pd  %%zmm16, %%zmm4 , %%zmm27             \n\t" // F11
        "vfmadd231pd  %%zmm16, %%zmm4 , %%zmm8              \n\t" // F11 - dup - 8 - 27
        "vmovups -384(%6,%0,8), %%zmm17                     \n\t" // LA21
        "vmovups      %%zmm9 , %%zmm28                      \n\t" // BY2 - 9 - 28
        "vmovups      %%zmm9 , %%zmm23                      \n\t" // BY2 - 9 - 23
        "vmovups -320(%5,%0,8), %%zmm16                     \n\t" // LA12
        "vmovups       %%zmm2 , %%zmm10                     \n\t" // restore from chpt - 2 - 10

        "vfmadd231pd  %%zmm17, %%zmm5 , %%zmm27             \n\t" // F21
        "vfmadd231pd  %%zmm17, %%zmm5 , %%zmm8              \n\t" // F21 - dup - 8 - 27
        "vmovups -384(%7,%0,8), %%zmm18                     \n\t" // LA31
        "vfmadd231pd  %%zmm16, %%zmm4 , %%zmm28             \n\t" // F12
        "vfmadd231pd  %%zmm16, %%zmm4 , %%zmm9              \n\t" // F12 - dup - 9 - 28
        "vmovups -320(%6,%0,8), %%zmm17                     \n\t" // LA22
        "vmovups      %%zmm10, %%zmm29                      \n\t" // BY3 - 10 - 29
        "vmovups      %%zmm10, %%zmm24                      \n\t" // BY3 - 10 - 24
        "vmovups -256(%5,%0,8), %%zmm16                     \n\t" // LA13
        "vmovups       %%zmm3 , %%zmm11                     \n\t" // restore from chpt - 3 - 11

        "vfmadd231pd  %%zmm18, %%zmm6 , %%zmm27             \n\t" // F31
        "vfmadd231pd  %%zmm18, %%zmm6 , %%zmm8              \n\t" // F31 - dup - 8 - 27
        "vmovups -384(%8,%0,8), %%zmm19                     \n\t" // LA41
        "vfmadd231pd  %%zmm17, %%zmm5 , %%zmm28             \n\t" // F22
        "vfmadd231pd  %%zmm17, %%zmm5 , %%zmm9              \n\t" // F22 - dup - 9 - 28
        "vmovups -320(%7,%0,8), %%zmm18                     \n\t" // LA32
        "vfmadd231pd  %%zmm16, %%zmm4 , %%zmm29             \n\t" // F13
        "vfmadd231pd  %%zmm16, %%zmm4 , %%zmm10             \n\t" // F13 - dup - 10 - 29
        "vmovups -256(%6,%0,8), %%zmm17                     \n\t" // LA23
        "vmovups      %%zmm11, %%zmm30                      \n\t" // BY4 - 11 - 30
        "vmovups      %%zmm11, %%zmm25                      \n\t" // BY4 - 11 - 25
        "vmovups -192(%5,%0,8), %%zmm16                     \n\t" // LA14
        "vmovups       %%zmm13, %%zmm12                     \n\t" // restore from chpt - 13 - 12

        "vfmadd231pd  %%zmm19, %%zmm7 , %%zmm27             \n\t" // F41
        "vfmadd231pd  %%zmm19, %%zmm7 , %%zmm8              \n\t" // F41 - dup - 8 - 27
        "vfmadd231pd  %%zmm18, %%zmm6 , %%zmm28             \n\t" // F32
        "vfmadd231pd  %%zmm18, %%zmm6 , %%zmm9              \n\t" // F32 - dup - 9 - 28
        "vmovups -320(%8,%0,8), %%zmm19                     \n\t" // LA42
        "vfmadd231pd  %%zmm17, %%zmm5 , %%zmm29             \n\t" // F23
        "vfmadd231pd  %%zmm17, %%zmm5 , %%zmm10             \n\t" // F23 - dup - 10 - 29
        "vmovups -256(%7,%0,8), %%zmm18                     \n\t" // LA33
        "vfmadd231pd  %%zmm16, %%zmm4 , %%zmm30             \n\t" // F14
        "vfmadd231pd  %%zmm16, %%zmm4 , %%zmm11             \n\t" // F14 - dup - 11 - 30
        "vmovups -192(%6,%0,8), %%zmm17                     \n\t" // LA24
        "vmovups      %%zmm12, %%zmm31                      \n\t" // BY5 - 12 - 31
        "vmovups      %%zmm12, %%zmm26                      \n\t" // BY5 - 12 - 26
        "vmovups -128(%5,%0,8), %%zmm16                     \n\t" // LA15
        "vpcmpeqd     %%zmm8 , %%zmm27, %%k1                \n\t" // C1
        "vmovups -64(%4,%0,8), %%zmm8                       \n\t" // LY6

        // original loop body

        "vmovups       %%zmm22, %%zmm0                      \n\t" // BY1 - 22 - 0
        "vmovups       %%zmm27, -384(%4,%0,8)               \n\t" // S1
        "vfmadd231pd   %%zmm19, %%zmm7 , %%zmm28            \n\t" // F42
        "vfmadd231pd   %%zmm19, %%zmm7 , %%zmm9             \n\t" // F42 - dup - 9 - 28
        "vfmadd231pd   %%zmm18, %%zmm6 , %%zmm29            \n\t" // F33
        "vfmadd231pd   %%zmm18, %%zmm6 , %%zmm10            \n\t" // F33 - dup - 10 - 29
        "vmovups -256(%8,%0,8), %%zmm19                     \n\t" // A43
        "vfmadd231pd   %%zmm17, %%zmm5 , %%zmm30            \n\t" // F24
        "vfmadd231pd   %%zmm17, %%zmm5 , %%zmm11            \n\t" // F24 - dup - 11 - 30
        "vmovups -192(%7,%0,8), %%zmm18                     \n\t" // A34
        "vfmadd231pd   %%zmm16, %%zmm4 , %%zmm31            \n\t" // F15
        "vfmadd231pd   %%zmm16, %%zmm4 , %%zmm12            \n\t" // F15 - dup - 12 - 31
        "vmovups -128(%6,%0,8), %%zmm17                     \n\t" // A25
        "vmovups       %%zmm8 , %%zmm27                     \n\t" // BY6 - 8 - 27
        "vmovups       %%zmm8 , %%zmm22                     \n\t" // BY6 - 8 - 22
        "vmovups  -64(%5,%0,8), %%zmm16                     \n\t" // LA16
        "vpcmpeqd     %%zmm9 , %%zmm28, %%k2                \n\t" // C2
        "vmovups     (%4,%0,8), %%zmm9                      \n\t" // LY7

        "vmovups       %%zmm23, %%zmm1                      \n\t" // BY2 - 23 - 1
        "vmovups       %%zmm28, -320(%4,%0,8)               \n\t" // S2
        "vfmadd231pd   %%zmm19, %%zmm7 , %%zmm29            \n\t" // F43
        "vfmadd231pd   %%zmm19, %%zmm7 , %%zmm10            \n\t" // F43 - dup - 10 - 29
        "kandw         %%k1   , %%k2   , %%k2               \n\t" // red - 1 - 2 -> 2
        "vfmadd231pd   %%zmm18, %%zmm6 , %%zmm30            \n\t" // F34
        "vfmadd231pd   %%zmm18, %%zmm6 , %%zmm11            \n\t" // F34 - dup - 11 - 30
        "vmovups -192(%8,%0,8), %%zmm19                     \n\t" // A44
        "vfmadd231pd   %%zmm17, %%zmm5 , %%zmm31            \n\t" // F25
        "vfmadd231pd   %%zmm17, %%zmm5 , %%zmm12            \n\t" // F25 - dup - 12 - 31
        "vmovups -128(%7,%0,8), %%zmm18                     \n\t" // A35
        "vfmadd231pd   %%zmm16, %%zmm4 , %%zmm27            \n\t" // F16
        "vfmadd231pd   %%zmm16, %%zmm4 , %%zmm8             \n\t" // F16 - dup - 8 - 27
        "vmovups  -64(%6,%0,8), %%zmm17                     \n\t" // A26
        "vmovups       %%zmm9 , %%zmm28                     \n\t" // BY7 - 9 - 28
        "vmovups       %%zmm9 , %%zmm23                     \n\t" // BY7 - 9 - 23
        "vmovups     (%5,%0,8), %%zmm16                     \n\t" // LA17
        "vpcmpeqd     %%zmm10, %%zmm29, %%k3                \n\t" // C3
        "vmovups   64(%4,%0,8), %%zmm10                     \n\t" // LY8

        "vmovups       %%zmm24, %%zmm2                      \n\t" // BY3 - 24 - 2
        "vmovups       %%zmm29, -256(%4,%0,8)               \n\t" // S3
        "vfmadd231pd   %%zmm19, %%zmm7 , %%zmm30            \n\t" // F44
        "vfmadd231pd   %%zmm19, %%zmm7 , %%zmm11            \n\t" // F44 - dup - 11 - 30
        "kandw         %%k2   , %%k3   , %%k3               \n\t" // red - 2 - 3 -> 3
        "vfmadd231pd   %%zmm18, %%zmm6 , %%zmm31            \n\t" // F35
        "vfmadd231pd   %%zmm18, %%zmm6 , %%zmm12            \n\t" // F35 - dup - 12 - 31
        "vmovups -128(%8,%0,8), %%zmm19                     \n\t" // A45
        "vfmadd231pd   %%zmm17, %%zmm5 , %%zmm27            \n\t" // F26
        "vfmadd231pd   %%zmm17, %%zmm5 , %%zmm8             \n\t" // F26 - dup - 8 - 27
        "vmovups  -64(%7,%0,8), %%zmm18                     \n\t" // A36
        "vfmadd231pd   %%zmm16, %%zmm4 , %%zmm28            \n\t" // F17
        "vfmadd231pd   %%zmm16, %%zmm4 , %%zmm9             \n\t" // F17 - dup - 9 - 28
        "vmovups     (%6,%0,8), %%zmm17                     \n\t" // A27
        "vmovups       %%zmm10, %%zmm29                     \n\t" // BY8 - 10 - 29
        "vmovups       %%zmm10, %%zmm24                     \n\t" // BY8 - 10 - 24
        "vmovups   64(%5,%0,8), %%zmm16                     \n\t" // LA18
        "vpcmpeqd     %%zmm11, %%zmm30, %%k4                \n\t" // C4
        "vmovups  128(%4,%0,8), %%zmm11                     \n\t" // LY9

        "vmovups       %%zmm25, %%zmm3                      \n\t" // BY4 - 25 - 3
        "vmovups       %%zmm30, -192(%4,%0,8)               \n\t" // S4
        "vfmadd231pd   %%zmm19, %%zmm7 , %%zmm31            \n\t" // F45
        "vfmadd231pd   %%zmm19, %%zmm7 , %%zmm12            \n\t" // F45 - dup - 12 - 31
        "kandw         %%k3   , %%k4   , %%k4               \n\t" // red - 3 - 4 -> 4
        "vfmadd231pd   %%zmm18, %%zmm6 , %%zmm27            \n\t" // F36
        "vfmadd231pd   %%zmm18, %%zmm6 , %%zmm8             \n\t" // F36 - dup - 8 - 27
        "vmovups  -64(%8,%0,8), %%zmm19                     \n\t" // A46
        "vfmadd231pd   %%zmm17, %%zmm5 , %%zmm28            \n\t" // F27
        "vfmadd231pd   %%zmm17, %%zmm5 , %%zmm9             \n\t" // F27 - dup - 9 - 28
        "vmovups     (%7,%0,8), %%zmm18                     \n\t" // A37
        "vfmadd231pd   %%zmm16, %%zmm4 , %%zmm29            \n\t" // F18
        "vfmadd231pd   %%zmm16, %%zmm4 , %%zmm10            \n\t" // F18 - dup - 10 - 29
        "vmovups   64(%6,%0,8), %%zmm17                     \n\t" // A28
        "vmovups       %%zmm11, %%zmm30                     \n\t" // BY9 - 11 - 30
        "vmovups       %%zmm11, %%zmm25                     \n\t" // BY9 - 11 - 25
        "vmovups  128(%5,%0,8), %%zmm16                     \n\t" // LA19
        "vpcmpeqd     %%zmm12, %%zmm31, %%k5                \n\t" // C4
        "vmovups  192(%4,%0,8), %%zmm12                     \n\t" // LY10

        "vmovups       %%zmm26, %%zmm13                     \n\t" // BY5 - 26 - 13
        "vmovups       %%zmm31, -128(%4,%0,8)               \n\t" // S5
        "vfmadd231pd   %%zmm19, %%zmm7 , %%zmm27            \n\t" // F46
        "vfmadd231pd   %%zmm19, %%zmm7 , %%zmm8             \n\t" // F46 - dup - 8 - 27
        "kandw         %%k4   , %%k5   , %%k5               \n\t" // red - 4 - 5 -> 5
        "vfmadd231pd   %%zmm18, %%zmm6 , %%zmm28            \n\t" // F37
        "vfmadd231pd   %%zmm18, %%zmm6 , %%zmm9             \n\t" // F37 - dup - 9 - 28
        "vmovups     (%8,%0,8), %%zmm19                     \n\t" // A47
        "vfmadd231pd   %%zmm17, %%zmm5 , %%zmm29            \n\t" // F28
        "vfmadd231pd   %%zmm17, %%zmm5 , %%zmm10            \n\t" // F28 - dup - 10 - 29
        "vmovups   64(%7,%0,8), %%zmm18                     \n\t" // A38
        "vfmadd231pd   %%zmm16, %%zmm4 , %%zmm30            \n\t" // F19
        "vfmadd231pd   %%zmm16, %%zmm4 , %%zmm11            \n\t" // F19 - dup - 11 - 30
        "kxorw         %%k0   , %%k5   , %%k5               \n\t"
        "vmovups  128(%6,%0,8), %%zmm17                     \n\t" // A29
        "vmovups       %%zmm12, %%zmm31                     \n\t" // BY10 - 12 - 31
        "vmovups       %%zmm12, %%zmm26                     \n\t" // BY10 - 12 - 26
        "vmovups  192(%5,%0,8), %%zmm16                     \n\t" // LA10
        "vpcmpeqd     %%zmm8 , %%zmm27, %%k1                \n\t" // C1
        "vmovups  256(%4,%0,8), %%zmm8                      \n\t" // LY11

        "addq          $40    , %0                          \n\t"
        "ktestw        %%k0   , %%k5                        \n\t"
        "jnz           5f                                   \n\t"
        "subq          $40    , %1                          \n\t"
        "jnz           1b                                   \n\t"
        "jmp           2b                                   \n\t"

        // error handler to epilogue
        "4:                                                 \n\t"
        "addq          $1     , %2                          \n\t"
        "vmovups       %%zmm0 , %%zmm8                      \n\t" // restore from chpt - 0 - 8

        "vmovups      %%zmm8 , %%zmm27                      \n\t" // BY1 - 8 - 27
        "vmovups      %%zmm8 , %%zmm22                      \n\t" // BY1 - 8 - 22
        "vmovups -384(%5,%0,8), %%zmm16                     \n\t" // LA11
        "vmovups       %%zmm1 , %%zmm9                      \n\t" // restore from chpt - 1 - 9

        "vfmadd231pd  %%zmm16, %%zmm4 , %%zmm27             \n\t" // F11
        "vfmadd231pd  %%zmm16, %%zmm4 , %%zmm8              \n\t" // F11 - dup - 8 - 27
        "vmovups -384(%6,%0,8), %%zmm17                     \n\t" // LA21
        "vmovups      %%zmm9 , %%zmm28                      \n\t" // BY2 - 9 - 28
        "vmovups      %%zmm9 , %%zmm23                      \n\t" // BY2 - 9 - 23
        "vmovups -320(%5,%0,8), %%zmm16                     \n\t" // LA12
        "vmovups       %%zmm2 , %%zmm10                     \n\t" // restore from chpt - 2 - 10

        "vfmadd231pd  %%zmm17, %%zmm5 , %%zmm27             \n\t" // F21
        "vfmadd231pd  %%zmm17, %%zmm5 , %%zmm8              \n\t" // F21 - dup - 8 - 27
        "vmovups -384(%7,%0,8), %%zmm18                     \n\t" // LA31
        "vfmadd231pd  %%zmm16, %%zmm4 , %%zmm28             \n\t" // F12
        "vfmadd231pd  %%zmm16, %%zmm4 , %%zmm9              \n\t" // F12 - dup - 9 - 28
        "vmovups -320(%6,%0,8), %%zmm17                     \n\t" // LA22
        "vmovups      %%zmm10, %%zmm29                      \n\t" // BY3 - 10 - 29
        "vmovups      %%zmm10, %%zmm24                      \n\t" // BY3 - 10 - 24
        "vmovups -256(%5,%0,8), %%zmm16                     \n\t" // LA13
        "vmovups       %%zmm3 , %%zmm11                     \n\t" // restore from chpt - 3 - 11

        "vfmadd231pd  %%zmm18, %%zmm6 , %%zmm27             \n\t" // F31
        "vfmadd231pd  %%zmm18, %%zmm6 , %%zmm8              \n\t" // F31 - dup - 8 - 27
        "vmovups -384(%8,%0,8), %%zmm19                     \n\t" // LA41
        "vfmadd231pd  %%zmm17, %%zmm5 , %%zmm28             \n\t" // F22
        "vfmadd231pd  %%zmm17, %%zmm5 , %%zmm9              \n\t" // F22 - dup - 9 - 28
        "vmovups -320(%7,%0,8), %%zmm18                     \n\t" // LA32
        "vfmadd231pd  %%zmm16, %%zmm4 , %%zmm29             \n\t" // F13
        "vfmadd231pd  %%zmm16, %%zmm4 , %%zmm10             \n\t" // F13 - dup - 10 - 29
        "vmovups -256(%6,%0,8), %%zmm17                     \n\t" // LA23
        "vmovups      %%zmm11, %%zmm30                      \n\t" // BY4 - 11 - 30
        "vmovups      %%zmm11, %%zmm25                      \n\t" // BY4 - 11 - 25
        "vmovups -192(%5,%0,8), %%zmm16                     \n\t" // LA14
        "vmovups       %%zmm13, %%zmm12                     \n\t" // restore from chpt - 13 - 12

        "vfmadd231pd  %%zmm19, %%zmm7 , %%zmm27             \n\t" // F41
        "vfmadd231pd  %%zmm19, %%zmm7 , %%zmm8              \n\t" // F41 - dup - 8 - 27
        "vfmadd231pd  %%zmm18, %%zmm6 , %%zmm28             \n\t" // F32
        "vfmadd231pd  %%zmm18, %%zmm6 , %%zmm9              \n\t" // F32 - dup - 9 - 28
        "vmovups -320(%8,%0,8), %%zmm19                     \n\t" // LA42
        "vfmadd231pd  %%zmm17, %%zmm5 , %%zmm29             \n\t" // F23
        "vfmadd231pd  %%zmm17, %%zmm5 , %%zmm10             \n\t" // F23 - dup - 10 - 29
        "vmovups -256(%7,%0,8), %%zmm18                     \n\t" // LA33
        "vfmadd231pd  %%zmm16, %%zmm4 , %%zmm30             \n\t" // F14
        "vfmadd231pd  %%zmm16, %%zmm4 , %%zmm11             \n\t" // F14 - dup - 11 - 30
        "vmovups -192(%6,%0,8), %%zmm17                     \n\t" // LA24
        "vmovups      %%zmm12, %%zmm31                      \n\t" // BY5 - 12 - 31
        "vmovups      %%zmm12, %%zmm26                      \n\t" // BY5 - 12 - 26
        "vmovups -128(%5,%0,8), %%zmm16                     \n\t" // LA15
        "vpcmpeqd     %%zmm8 , %%zmm27, %%k1                \n\t" // C1
        "vmovups      %%zmm15, %%zmm8                       \n\t" // restore from chpt - 15 - 8

        // original epilogue
        "vmovups       %%zmm27, -384(%4,%0,8)               \n\t" // S1
        "vfmadd231pd   %%zmm19, %%zmm7 , %%zmm28            \n\t" // F42
        "vfmadd231pd   %%zmm19, %%zmm7 , %%zmm9             \n\t" // F42 - dup - 9 - 28
        "vfmadd231pd   %%zmm18, %%zmm6 , %%zmm29            \n\t" // F33
        "vfmadd231pd   %%zmm18, %%zmm6 , %%zmm10            \n\t" // F33 - dup - 10 - 29
        "vmovups -256(%8,%0,8), %%zmm19                     \n\t" // A43
        "vfmadd231pd   %%zmm17, %%zmm5 , %%zmm30            \n\t" // F24
        "vfmadd231pd   %%zmm17, %%zmm5 , %%zmm11            \n\t" // F24 - dup - 11 - 30
        "vmovups -192(%7,%0,8), %%zmm18                     \n\t" // A34
        "vfmadd231pd   %%zmm16, %%zmm4 , %%zmm31            \n\t" // F15
        "vfmadd231pd   %%zmm16, %%zmm4 , %%zmm12            \n\t" // F15 - dup - 12 - 31
        "vmovups -128(%6,%0,8), %%zmm17                     \n\t" // A25
        "vmovups  -64(%5,%0,8), %%zmm16                     \n\t" // LA16

        "vpcmpeqd      %%zmm9 , %%zmm28, %%k2               \n\t" // C2
        "vmovups       %%zmm28, -320(%4,%0,8)               \n\t" // S2
        "vfmadd231pd   %%zmm19, %%zmm7 , %%zmm29            \n\t" // F43
        "vfmadd231pd   %%zmm19, %%zmm7 , %%zmm10            \n\t" // F43 - dup - 10 - 29
        "vfmadd231pd   %%zmm18, %%zmm6 , %%zmm30            \n\t" // F34
        "vfmadd231pd   %%zmm18, %%zmm6 , %%zmm11            \n\t" // F34 - dup - 11 - 30
        "kandw         %%k1   , %%k2   , %%k2               \n\t" // red - 1 - 2 -> 2
        "vmovups -192(%8,%0,8), %%zmm19                     \n\t" // A44
        "vfmadd231pd   %%zmm17, %%zmm5 , %%zmm31            \n\t" // F25
        "vfmadd231pd   %%zmm17, %%zmm5 , %%zmm12            \n\t" // F25 - dup - 12 - 31
        "vmovups -128(%7,%0,8), %%zmm18                     \n\t" // A35
        "vfmadd231pd   %%zmm16, %%zmm4 , %%zmm27            \n\t" // F16
        "vfmadd231pd   %%zmm16, %%zmm4 , %%zmm8             \n\t" // F16 - dup - 8 - 27
        "vmovups  -64(%6,%0,8), %%zmm17                     \n\t" // A26

        "vpcmpeqd      %%zmm10, %%zmm29, %%k3               \n\t" // C3
        "vmovups       %%zmm29, -256(%4,%0,8)               \n\t" // S3
        "vfmadd231pd   %%zmm19, %%zmm7 , %%zmm30            \n\t" // F44
        "vfmadd231pd   %%zmm19, %%zmm7 , %%zmm11            \n\t" // F44 - dup - 11 - 30
        "vfmadd231pd   %%zmm18, %%zmm6 , %%zmm31            \n\t" // F35
        "vfmadd231pd   %%zmm18, %%zmm6 , %%zmm12            \n\t" // F35 - dup - 12 - 31
        "kandw         %%k2   , %%k3   , %%k3               \n\t" // red - 2 - 3 -> 3
        "vmovups -128(%8,%0,8), %%zmm19                     \n\t" // A45
        "vfmadd231pd   %%zmm17, %%zmm5 , %%zmm27            \n\t" // F26
        "vfmadd231pd   %%zmm17, %%zmm5 , %%zmm8             \n\t" // F26 - dup - 8 - 27
        "vmovups  -64(%7,%0,8), %%zmm18                     \n\t" // A36

        "vpcmpeqd      %%zmm11, %%zmm30, %%k4               \n\t" // C4
        "vmovups       %%zmm30, -192(%4,%0,8)               \n\t" // S4
        "vfmadd231pd   %%zmm19, %%zmm7 , %%zmm31            \n\t" // F45
        "vfmadd231pd   %%zmm19, %%zmm7 , %%zmm12            \n\t" // F45 - dup - 12 - 31
        "vfmadd231pd   %%zmm18, %%zmm6 , %%zmm27            \n\t" // F36
        "vfmadd231pd   %%zmm18, %%zmm6 , %%zmm8             \n\t" // F36 - dup - 8 - 27
        "kandw         %%k3   , %%k4   , %%k4               \n\t" // red - 3 - 4 -> 4
        "vmovups  -64(%8,%0,8), %%zmm19                     \n\t" // A46

        "vpcmpeqd      %%zmm12, %%zmm31, %%k5               \n\t" // C5
        "vmovups       %%zmm31, -128(%4,%0,8)               \n\t" // S5
        "vfmadd231pd   %%zmm19, %%zmm7 , %%zmm27            \n\t" // F46
        "vfmadd231pd   %%zmm19, %%zmm7 , %%zmm8             \n\t" // F46 - dup - 8 - 27
        "kandw         %%k4   , %%k5   , %%k5               \n\t" // red - 4 - 5 -> 5

        "vmovups       %%zmm22, %%zmm15                     \n\t" // BY1 - 22 - 15
        "vpcmpeqd      %%zmm8 , %%zmm27, %%k1               \n\t" // C1
        "vmovups       %%zmm27, -64(%4,%0,8)                \n\t" // S1
        
        "kandw         %%k1   , %%k5   , %%k5               \n\t" // red - 3 - 4 -> 4
        "kxorw         %%k0   , %%k5   , %%k5               \n\t"
        "ktestw        %%k0   , %%k5                        \n\t"
        "jnz           5f                                   \n\t" // 5f

        "jmp           6f                                   \n\t"

        "5:                                                 \n\t"

        "6:                                                 \n\t"
        "vzeroupper                      \n\t"

        :
            "+r" (i),   // 0    
            "+r" (n),   // 1
            "+r" (err_num) // 2
        :
            "r" (x),    // 3
            "r" (y),    // 4
            "r" (a1),   // 5
            "r" (a2),   // 6
            "r" (a3),   // 7
            "r" (a4)    // 8
        : "cc", 
        "%xmm4", "%xmm5", "%xmm6", "%xmm7",
        "%xmm12", "%xmm13", "%xmm14", "%xmm15",
        "memory"
    );
    return err_num;
}

static long dgemv_kernel_ft_trsv(long int n, double *a1, double *a2, double *a3, double *a4, double *x, double *y)
{
    register long int i = 0;
    register long int err_num = 0;
    __asm__ __volatile__(
        "vxorps       %%zmm4 , %%zmm4 , %%zmm4              \n\t"
        "vxorps       %%zmm5 , %%zmm5 , %%zmm5              \n\t"
        "vxorps       %%zmm6 , %%zmm6 , %%zmm6              \n\t"
        "vxorps       %%zmm7 , %%zmm7 , %%zmm7              \n\t"

        "vxorps       %%zmm8 , %%zmm8 , %%zmm8              \n\t"
        "vxorps       %%zmm9 , %%zmm9 , %%zmm9              \n\t"
        "vxorps       %%zmm10, %%zmm10, %%zmm10             \n\t"
        "vxorps       %%zmm11, %%zmm11, %%zmm11             \n\t"

        "kxnorw       %%k4   , %%k4   , %%k4                \n\t"

        // Prologue
        "vmovups    (%3,%0,8), %%zmm12                      \n\t" // LX

        "vmovups      %%zmm4 , %%zmm28                      \n\t" // B1
        "vmovups    (%5,%0,8), %%zmm16                      \n\t" // LA1
        
        "vfmadd231pd  %%zmm16, %%zmm12, %%zmm4              \n\t" // F1
        "vmovups      %%zmm5 , %%zmm29                      \n\t" // B2
        "vmovups    (%6,%0,8), %%zmm17                      \n\t" // LA2
        "vmovups      %%zmm12, %%zmm27                      \n\t" // BX
        
        "vfmadd231pd  %%zmm16, %%zmm12, %%zmm8              \n\t" // F1 - dup - 4 - 8
        "vfmadd231pd  %%zmm17, %%zmm12, %%zmm5              \n\t" // F2
        "vmovups      %%zmm6 , %%zmm30                      \n\t" // B3
        "vmovups    (%7,%0,8), %%zmm18                      \n\t" // LA3

        "addq       $8 , %0                                 \n\t"
        "subq       $8 , %1                                 \n\t"
        "jz         2f                                      \n\t"

        ".p2align 4                                         \n\t"
        "1:                                                 \n\t"
        "vmovups      %%zmm28, %%zmm0                       \n\t" // B11 - 28 - 0
        "vpcmpeqd     %%zmm4 , %%zmm8 , %%k0                \n\t" // C1
        "vfmadd231pd  %%zmm17, %%zmm12, %%zmm9              \n\t" // F2 - dup - 5 - 9
        "vfmadd231pd  %%zmm18, %%zmm12, %%zmm6              \n\t" // F3
        "vmovups      %%zmm7 , %%zmm31                      \n\t" // B4
        "vmovups -64(%8,%0,8), %%zmm19                      \n\t" // LA4   
        "vmovups    (%3,%0,8), %%zmm12                      \n\t" // LX5

        "vmovups      %%zmm29, %%zmm1                       \n\t" // B22 - 29 - 1
        "vpcmpeqd     %%zmm5 , %%zmm9 , %%k1                \n\t" // C2
        "vfmadd231pd  %%zmm18, %%zmm27, %%zmm10             \n\t" // F3 - dup - 6 - 10
        "vfmadd231pd  %%zmm19, %%zmm27, %%zmm7              \n\t" // F4
        "vmovups      %%zmm4 , %%zmm28                      \n\t" // B5
        "vmovups    (%5,%0,8), %%zmm16                      \n\t" // LA5  

        "kandw        %%k0   , %%k1   , %%k0                \n\t" // check - red - 0 - 1
        "vmovups      %%zmm30, %%zmm2                       \n\t" // B22 - 29 - 1
        "vpcmpeqd     %%zmm6 , %%zmm10, %%k2                \n\t" // C3
        "vfmadd231pd  %%zmm19, %%zmm27, %%zmm11             \n\t" // F4 - dup - 7 - 11
        "vfmadd231pd  %%zmm16, %%zmm12, %%zmm4              \n\t" // F5
        "vmovups      %%zmm5 , %%zmm29                      \n\t" // B6
        "vmovups    (%6,%0,8), %%zmm17                      \n\t" // LA6  
        "kandw        %%k0   , %%k2   , %%k0                \n\t" // check - red - 0 - 2
        "vmovups      %%zmm12, %%zmm27                      \n\t" // BX

        "vmovups      %%zmm31, %%zmm3                       \n\t" // B22 - 29 - 3
        "vpcmpeqd     %%zmm7 , %%zmm11, %%k3                \n\t" // C4
        "vfmadd231pd  %%zmm16, %%zmm12, %%zmm8              \n\t" // F5 - dup - 4 - 8
        "vfmadd231pd  %%zmm17, %%zmm12, %%zmm5              \n\t" // F6
        "vmovups      %%zmm6 , %%zmm30                      \n\t" // B7
        "kandw        %%k0   , %%k3   , %%k0                \n\t" // check - red - 0 - 3
        "vmovups    (%7,%0,8), %%zmm18                      \n\t" // LA7

        "kxorw        %%k0   , %%k4   , %%k0                \n\t"

        "addq             $8     , %0                           \n\t"

        "ktestw       %%k0   , %%k4                         \n\t"
        "jnz           4f                                    \n\t"

        "subq           $8 , %1                          \n\t"
        "jnz            1b                               \n\t"

        // Epilogue
        "2:                                                 \n\t"
        "vmovups      %%zmm28, %%zmm0                       \n\t" // B11 - 28 - 0
        "vpcmpeqd     %%zmm4 , %%zmm8 , %%k0                \n\t" // C1
        "vfmadd231pd  %%zmm17, %%zmm12, %%zmm9              \n\t" // F2 - dup - 5 - 9
        "vfmadd231pd  %%zmm18, %%zmm12, %%zmm6              \n\t" // F3
        "vmovups      %%zmm7 , %%zmm31                      \n\t" // B4
        "vmovups -64(%8,%0,8), %%zmm19                      \n\t" // LA4

        "vmovups      %%zmm29, %%zmm1                       \n\t" // B22 - 29 - 1
        "vpcmpeqd     %%zmm5 , %%zmm9 , %%k1                \n\t" // C2
        "vfmadd231pd  %%zmm18, %%zmm12, %%zmm10             \n\t" // F3 - dup - 6 - 10
        "vfmadd231pd  %%zmm19, %%zmm12, %%zmm7              \n\t" // F4

        "kandw        %%k0   , %%k1   , %%k0                \n\t" // check - red - 0 - 1

        "vmovups      %%zmm30, %%zmm2                       \n\t" // B22 - 30 - 2
        "vpcmpeqd     %%zmm6 , %%zmm10, %%k2                \n\t" // C3
        "vfmadd231pd  %%zmm19, %%zmm12, %%zmm11             \n\t" // F4 - dup - 7 - 11
        "kandw        %%k0   , %%k2   , %%k0                \n\t" // check - red - 0 - 2

        "vmovups      %%zmm31, %%zmm3                       \n\t" // B22 - 31 - 3
        "vpcmpeqd     %%zmm7 , %%zmm11, %%k3                \n\t" // C4

        "kandw        %%k0   , %%k3   , %%k0                \n\t" // check - red - 0 - 3
        "kxorw        %%k0   , %%k4   , %%k0                \n\t"
        "ktestw       %%k0   , %%k4                         \n\t"
        "jnz           7f                                    \n\t"

        // reduction
        "3:                                                 \n\t"

        "vmovsd         (%4),    %%xmm0         \n\t"
        "vmovsd        8(%4),    %%xmm1         \n\t"
        "vmovsd       16(%4),    %%xmm2         \n\t"
        "vmovsd       24(%4),    %%xmm3         \n\t"

        "vextractf64x4     $0, %%zmm4 , %%ymm16             \n\t"
        "vextractf64x4     $0, %%zmm5 , %%ymm17             \n\t"
        "vextractf64x4     $0, %%zmm6 , %%ymm18             \n\t"
        "vextractf64x4     $0, %%zmm7 , %%ymm19             \n\t"
        
        "vextractf64x4     $1, %%zmm4 , %%ymm4              \n\t"
        "vextractf64x4     $1, %%zmm5 , %%ymm5              \n\t"
        "vextractf64x4     $1, %%zmm6 , %%ymm6              \n\t"
        "vextractf64x4     $1, %%zmm7 , %%ymm7              \n\t"

        "vaddpd       %%ymm4 , %%ymm16, %%ymm4              \n\t"
        "vaddpd       %%ymm5 , %%ymm17, %%ymm5              \n\t"
        "vaddpd       %%ymm6 , %%ymm18, %%ymm6              \n\t"
        "vaddpd       %%ymm7 , %%ymm19, %%ymm7              \n\t"

        "vextractf128     $1 , %%ymm4, %%xmm12              \n\t"
        "vextractf128     $1 , %%ymm5, %%xmm13              \n\t"
        "vextractf128     $1 , %%ymm6, %%xmm14              \n\t"
        "vextractf128     $1 , %%ymm7, %%xmm15              \n\t"

        "vaddpd       %%xmm4 , %%xmm12, %%xmm4              \n\t"
        "vaddpd       %%xmm5 , %%xmm13, %%xmm5              \n\t"
        "vaddpd       %%xmm6 , %%xmm14, %%xmm6              \n\t"
        "vaddpd       %%xmm7 , %%xmm15, %%xmm7              \n\t"

        "vhaddpd      %%xmm4 , %%xmm4 , %%xmm4              \n\t"
        "vhaddpd      %%xmm5 , %%xmm5 , %%xmm5              \n\t"
        "vhaddpd      %%xmm6 , %%xmm6 , %%xmm6              \n\t"
        "vhaddpd      %%xmm7 , %%xmm7 , %%xmm7              \n\t"

        "vsubsd       %%xmm4,    %%xmm0, %%xmm4 \n\t"
        "vsubsd       %%xmm5,    %%xmm1, %%xmm5 \n\t"
        "vsubsd       %%xmm6,    %%xmm2, %%xmm6 \n\t"
        "vsubsd       %%xmm7,    %%xmm3, %%xmm7 \n\t"

        "vmovsd       %%xmm4 ,    (%4)                      \n\t"
        "vmovsd       %%xmm5 ,   8(%4)                      \n\t"
        "vmovsd       %%xmm6 ,  16(%4)                      \n\t"
        "vmovsd       %%xmm7 ,  24(%4)                      \n\t"
        "jmp          5f                                    \n\t"

        // error handler to loop body
        "4:                                                 \n\t"
        "addq          $1     , %2                          \n\t"
        "subq       $8 , %0                                 \n\t"
        // recover from prelogue
        "vmovups -64(%3,%0,8), %%zmm12                      \n\t" // LX

        "vmovups      %%zmm0 , %%zmm4                       \n\t" // restore from chpt
        "vmovups      %%zmm0 , %%zmm8                       \n\t" // restore from chpt
        "vmovups -64(%5,%0,8), %%zmm16                      \n\t" // LA1
        
        "vfmadd231pd  %%zmm16, %%zmm12, %%zmm4              \n\t" // F1
        "vmovups      %%zmm1 , %%zmm5                       \n\t" // restore from chpt
        "vmovups      %%zmm1 , %%zmm9                       \n\t" // restore from chpt
        "vmovups -64(%6,%0,8), %%zmm17                      \n\t" // LA2
        "vmovups      %%zmm12, %%zmm27                      \n\t" // BX
        
        "vfmadd231pd  %%zmm16, %%zmm12, %%zmm8              \n\t" // F1 - dup - 4 - 8
        "vfmadd231pd  %%zmm17, %%zmm12, %%zmm5              \n\t" // F2
        "vmovups      %%zmm2 , %%zmm6                       \n\t" // restore from chpt
        "vmovups      %%zmm2 , %%zmm10                      \n\t" // restore from chpt
        "vmovups -64(%7,%0,8), %%zmm18                      \n\t" // LA3

        "vpcmpeqd     %%zmm4 , %%zmm8 , %%k0                \n\t" // C1
        "vfmadd231pd  %%zmm17, %%zmm12, %%zmm9              \n\t" // F2 - dup - 5 - 9
        "vfmadd231pd  %%zmm18, %%zmm12, %%zmm6              \n\t" // F3
        "vmovups      %%zmm3 , %%zmm7                       \n\t" // restore from chpt
        "vmovups      %%zmm3 , %%zmm11                      \n\t" // restore from chpt
        "vmovups -64(%8,%0,8), %%zmm19                      \n\t" // LA4   
        "vmovups    (%3,%0,8), %%zmm12                      \n\t" // LX5

        "vpcmpeqd     %%zmm5 , %%zmm9 , %%k1                \n\t" // C2
        "vfmadd231pd  %%zmm18, %%zmm27, %%zmm10             \n\t" // F3 - dup - 6 - 10
        "vfmadd231pd  %%zmm19, %%zmm27, %%zmm7              \n\t" // F4
        "vmovups      %%zmm4 , %%zmm28                      \n\t" // B5
        "vmovups    (%5,%0,8), %%zmm16                      \n\t" // LA5  

        "kandw        %%k0   , %%k1   , %%k0                \n\t" // check - red - 0 - 1
        "vpcmpeqd     %%zmm6 , %%zmm10, %%k2                \n\t" // C3
        "vfmadd231pd  %%zmm19, %%zmm27, %%zmm11             \n\t" // F4 - dup - 7 - 11
        "vfmadd231pd  %%zmm16, %%zmm12, %%zmm4              \n\t" // F5
        "vmovups      %%zmm5 , %%zmm29                      \n\t" // B6
        "vmovups    (%6,%0,8), %%zmm17                      \n\t" // LA6  
        "kandw        %%k0   , %%k2   , %%k0                \n\t" // check - red - 0 - 2
        "vmovups      %%zmm12, %%zmm27                      \n\t" // BX

        "vpcmpeqd     %%zmm7 , %%zmm11, %%k3                \n\t" // C4
        "vfmadd231pd  %%zmm16, %%zmm12, %%zmm8              \n\t" // F5 - dup - 4 - 8
        "vfmadd231pd  %%zmm17, %%zmm12, %%zmm5              \n\t" // F6
        "vmovups      %%zmm6 , %%zmm30                      \n\t" // B7
        "kandw        %%k0   , %%k3   , %%k0                \n\t" // check - red - 0 - 3
        "vmovups    (%7,%0,8), %%zmm18                      \n\t" // LA7

        "kxorw        %%k0   , %%k4   , %%k0                \n\t"

        "addq         $8     , %0                           \n\t"

        "ktestw       %%k0   , %%k4                         \n\t"
        "jnz          6f                                    \n\t"

        "subq       $8 , %1                                 \n\t"
        "jnz        1b                                      \n\t"
        "jmp        2b                                      \n\t"

        "7:                                                 \n\t"
        "addq          $1     , %2                          \n\t"
        "vmovups -64(%3,%0,8), %%zmm12                      \n\t" // LX

        "vmovups      %%zmm0 , %%zmm4                       \n\t" // restore from chpt
        "vmovups      %%zmm0 , %%zmm8                       \n\t" // restore from chpt
        "vmovups -64(%5,%0,8), %%zmm16                      \n\t" // LA1
        
        "vfmadd231pd  %%zmm16, %%zmm12, %%zmm4              \n\t" // F1
        "vmovups      %%zmm1 , %%zmm5                       \n\t" // restore from chpt
        "vmovups      %%zmm1 , %%zmm9                       \n\t" // restore from chpt
        "vmovups -64(%6,%0,8), %%zmm17                      \n\t" // LA2
        "vmovups      %%zmm12, %%zmm27                      \n\t" // BX
        
        "vfmadd231pd  %%zmm16, %%zmm12, %%zmm8              \n\t" // F1 - dup - 4 - 8
        "vfmadd231pd  %%zmm17, %%zmm12, %%zmm5              \n\t" // F2
        "vmovups      %%zmm2 , %%zmm6                       \n\t" // restore from chpt
        "vmovups      %%zmm2 , %%zmm10                      \n\t" // restore from chpt
        "vmovups -64(%7,%0,8), %%zmm18                      \n\t" // LA3

        "vpcmpeqd     %%zmm4 , %%zmm8 , %%k0                \n\t" // C1
        "vfmadd231pd  %%zmm17, %%zmm12, %%zmm9              \n\t" // F2 - dup - 5 - 9
        "vfmadd231pd  %%zmm18, %%zmm12, %%zmm6              \n\t" // F3
        "vmovups      %%zmm3 , %%zmm7                       \n\t" // restore from chpt
        "vmovups      %%zmm3 , %%zmm11                      \n\t" // restore from chpt
        "vmovups -64(%8,%0,8), %%zmm19                      \n\t" // LA4

        "vpcmpeqd     %%zmm5 , %%zmm9 , %%k1                \n\t" // C2
        "vfmadd231pd  %%zmm18, %%zmm12, %%zmm10             \n\t" // F3 - dup - 6 - 10
        "vfmadd231pd  %%zmm19, %%zmm12, %%zmm7              \n\t" // F4

        "kandw        %%k0   , %%k1   , %%k0                \n\t" // check - red - 0 - 1

        "vpcmpeqd     %%zmm6 , %%zmm10, %%k2                \n\t" // C3
        "vfmadd231pd  %%zmm19, %%zmm12, %%zmm11             \n\t" // F4 - dup - 7 - 11
        "kandw        %%k0   , %%k2   , %%k0                \n\t" // check - red - 0 - 2

        "vpcmpeqd     %%zmm7 , %%zmm11, %%k3                \n\t" // C4

        "kandw        %%k0   , %%k3   , %%k0                \n\t" // check - red - 0 - 3
        "ktestw       %%k0   , %%k4                         \n\t"
        "jz           6f                                    \n\t"
        "jmp          3b                                    \n\t"

        // still incorrect
        "6:                                                 \n\t"


        "5:                                                 \n\t"
        "vzeroupper                                         \n\t"

        :
            "+r" (i),   // 0    
            "+r" (n),   // 1
            "+r" (err_num)    // 2
        :
            "r" (x),    // 3
            "r" (y),    // 4
            "r" (a1),   // 5
            "r" (a2),   // 6
            "r" (a3),   // 7
            "r" (a4)    // 8
        : "cc",
        "%xmm0", "%xmm1", "%xmm2", "%xmm3",
        "%xmm4", "%xmm5", "%xmm6", "%xmm7",
        "%xmm12", "%xmm13", "%xmm14", "%xmm15",
        "memory"
    );
    return err_num;
}

void ftblas_dtrsv_low_col_ft(double *A, int LDA, double *b, int n){
    int i,j,t;
    double *b_ptr=b,*a_ptr;
    double *a1,*a2,*a3,*a4,*x,*y;
    double coef;
    double buffer_x[4];
    int i8_left;
    int upper;
    int n4=n&-4;
    for (i = 0; i < n4; i += 4){
        i8_left = ((n4-i-4)/40 >= 2) ? (n4-i-4) / 40 * 40 - 32 : 0;
        upper=n4-i-4-i8_left;
        a_ptr=A+i*LDA+i;
        b_ptr=b+i;
        b_ptr[0]/=a_ptr[0];
        b_ptr[1]-=b_ptr[0]*a_ptr[1];
        b_ptr[1]/=a_ptr[1+LDA];
        b_ptr[2]-=(b_ptr[0]*a_ptr[2]+b_ptr[1]*a_ptr[2+1*LDA]);
        b_ptr[2]/=a_ptr[2+2*LDA];
        b_ptr[3]-=(b_ptr[0]*a_ptr[3]+b_ptr[1]*a_ptr[3+1*LDA]+b_ptr[2]*a_ptr[3+2*LDA]);
        b_ptr[3]/=a_ptr[3+3*LDA];
        if (upper){
            a1=a_ptr+4;
            a2=a1+LDA;
            a3=a1+LDA*2;
            a4=a1+LDA*3;
            x=b+i;
            y=b+i+4;
            __m256d vx1=_mm256_broadcast_sd(x);
            __m256d vx2=_mm256_broadcast_sd(x+1);
            __m256d vx3=_mm256_broadcast_sd(x+2);
            __m256d vx4=_mm256_broadcast_sd(x+3);
            for (t=0;t<upper;t+=4){
                __m256d vy=_mm256_loadu_pd(y);
                __m256d res1 = _mm256_mul_pd(_mm256_loadu_pd(a1),vx1);
                __m256d res2 = _mm256_mul_pd(_mm256_loadu_pd(a2),vx2);
                __m256d res3 = _mm256_mul_pd(_mm256_loadu_pd(a3),vx3);
                __m256d res4 = _mm256_mul_pd(_mm256_loadu_pd(a4),vx4);
                res1 = _mm256_add_pd(res1,res2);
                res3 = _mm256_add_pd(res3,res4);
                res1 = _mm256_add_pd(res1,res3);
                vy = _mm256_sub_pd(vy,res1);
                _mm256_storeu_pd(y,vy);
                y+=4;a1+=4;a2+=4;a3+=4;a4+=4;
            }

        }
        if (i8_left) {
            a1=a_ptr+(n4-i8_left-i);
            a2=a1+LDA;
            a3=a1+LDA*2;
            a4=a1+LDA*3;
            x=b+i;
            buffer_x[0]=-1.0*x[0];buffer_x[1]=-1.0*x[1];
            buffer_x[2]=-1.0*x[2];buffer_x[3]=-1.0*x[3];
            y=b+(n4-i8_left);
            long err_num = ft_dgemv_n_kernel_trsv(i8_left,a1,a2,a3,a4,buffer_x,y);
            if (err_num) printf("detected and corrected err: %ld\n", err_num);
        }

    }
    if (n4==n) return;
    //deal with the boundary condition
    for (i=n4;i<n;i++){
        double tmp=0.;
        a_ptr=A+i;
        b_ptr=b;
        for (j=0;j<n4;j++){
            tmp+=*a_ptr*(*b_ptr);
            b_ptr++;a_ptr+=LDA;
        }
        b[i]-=tmp;
    }
    a_ptr=A+LDA*n4+n4;
    b_ptr=b+n4;
    int diff=n-n4;
    if (diff==3){
        b_ptr[0]/=a_ptr[0];
        b_ptr[1]-=b_ptr[0]*a_ptr[1];
        b_ptr[1]/=a_ptr[1+LDA];
        b_ptr[2]-=(b_ptr[0]*a_ptr[2]+b_ptr[1]*a_ptr[2+1*LDA]);
        b_ptr[2]/=a_ptr[2+2*LDA];
    }else if (diff==2){
        b_ptr[0]/=a_ptr[0];
        b_ptr[1]-=b_ptr[0]*a_ptr[1];
        b_ptr[1]/=a_ptr[1+LDA];
    }else{
        b_ptr[0]/=a_ptr[0];
    }
}

void ftblas_dtrsv_low_row_ft(double *A, int LDA, double *b, int n){
    int i,j;
    double *b_ptr=b,*a_ptr;
    double coef;
    double t_1=0,t_2=0,t_t=0,t0,t1;
    int n4=n&-4;
    for (i = 0; i < n4; i += 4){
        int i32 = i & -8;
        //printf("i=%d,n=%d,i32=%d\n",i,n,i32);
        a_ptr=A+i*LDA;
        if (i32) {
            double *a1=a_ptr;
            double *a2=a_ptr+LDA;
            double *a3=a_ptr+LDA*2;
            double *a4=a_ptr+LDA*3;
            double *x=b;
            double *y=b+i;
            long err_num = dgemv_kernel_ft_trsv(i32, a1, a2, a3, a4, x, y);
            if (err_num) printf("detected and corrected err: %ld\n", err_num);
        }

        if (i!=i32){
            int t;
            double *a1=a_ptr+i32;
            double *a2=a_ptr+LDA+i32;
            double *a3=a_ptr+LDA*2+i32;
            double *a4=a_ptr+LDA*3+i32;
            double *x=b+i32;
            double *y=b+i;

            __m256d vx=_mm256_loadu_pd(x);
            __m256d vy=_mm256_loadu_pd(y);
            __m256d res1 = _mm256_mul_pd(_mm256_loadu_pd(a1),vx);
            __m256d res2 = _mm256_mul_pd(_mm256_loadu_pd(a2),vx);
            __m256d res3 = _mm256_mul_pd(_mm256_loadu_pd(a3),vx);
            __m256d res4 = _mm256_mul_pd(_mm256_loadu_pd(a4),vx);
            __m128d halfv1 = _mm_add_pd(_mm256_extractf128_pd(res1, 0), _mm256_extractf128_pd(res1, 1));
            __m128d halfv2 = _mm_add_pd(_mm256_extractf128_pd(res2, 0), _mm256_extractf128_pd(res2, 1));
            __m128d halfv3 = _mm_add_pd(_mm256_extractf128_pd(res3, 0), _mm256_extractf128_pd(res3, 1));
            __m128d halfv4 = _mm_add_pd(_mm256_extractf128_pd(res4, 0), _mm256_extractf128_pd(res4, 1));
            halfv1 = _mm_hadd_pd(halfv1, halfv1);
            halfv2 = _mm_hadd_pd(halfv2, halfv2);
            halfv3 = _mm_hadd_pd(halfv3, halfv3);
            halfv4 = _mm_hadd_pd(halfv4, halfv4);
            __m256d res;
            res[0]=halfv1[0];res[1]=halfv2[0];res[2]=halfv3[0];res[3]=halfv4[0];


            vy = _mm256_sub_pd(vy,res);
            _mm256_storeu_pd(y,vy);
        }
        a_ptr+=i;
        b_ptr=b+i;

        b_ptr[0]/=a_ptr[0];
        b_ptr[1]-=b_ptr[0]*a_ptr[LDA];
        b_ptr[1]/=a_ptr[1+LDA];
        b_ptr[2]-=(b_ptr[0]*a_ptr[2*LDA]+b_ptr[1]*a_ptr[1+2*LDA]);
        b_ptr[2]/=a_ptr[2+2*LDA];
        b_ptr[3]-=(b_ptr[0]*a_ptr[3*LDA]+b_ptr[1]*a_ptr[1+3*LDA]+b_ptr[2]*a_ptr[2+3*LDA]);
        b_ptr[3]/=a_ptr[3+3*LDA];
    }
    if (n4==n) return;
    //deal with the boundary condition
    for (i=n4;i<n;i++){
        int k;
        double sum=0.;
        a_ptr=A+LDA*i;
        int i32=i&-32;
        __m512d res1=_mm512_setzero_pd();
        __m512d res2=_mm512_setzero_pd();
        __m512d res3=_mm512_setzero_pd();
        __m512d res4=_mm512_setzero_pd();
        for (j=0;j<i32;j+=32){
            res1=_mm512_fmadd_pd(_mm512_loadu_pd(a_ptr+j),_mm512_loadu_pd(b+j),res1);
            res2=_mm512_fmadd_pd(_mm512_loadu_pd(a_ptr+j+8),_mm512_loadu_pd(b+j+8),res2);
            res3=_mm512_fmadd_pd(_mm512_loadu_pd(a_ptr+j+16),_mm512_loadu_pd(b+j+16),res3);
            res4=_mm512_fmadd_pd(_mm512_loadu_pd(a_ptr+j+24),_mm512_loadu_pd(b+j+24),res4);
        }
        res1=_mm512_add_pd(res1,res2);
        res3=_mm512_add_pd(res3,res4);
        res1=_mm512_add_pd(res1,res3);
        for (k=0;k<8;k++) sum+=res1[k];
        for (j=i32;j<i;j++){
            sum+=b[j]*a_ptr[j];
        }
        b[i]-=sum;
        b[i]/=a_ptr[i];
    }
}