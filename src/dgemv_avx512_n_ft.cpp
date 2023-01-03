#include "../include/ftblas.h"

static long dgemv_n_ft_kernel(long int n, double *a1, double *a2, double *a3, double *a4, double *x, double *y)
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

static long dgemv_n_kernel_unroll(long int n, double *a1, double *a2, double *a3, double *a4, double *x, double *y)
{
    register long int i = 0;
    
    __asm__ __volatile__(
        "vbroadcastsd		(%2) , %%zmm4    \n\t" // x
        "vbroadcastsd	   8(%2) , %%zmm5    \n\t" // x
        "vbroadcastsd	  16(%2) , %%zmm6    \n\t" // x
        "vbroadcastsd	  24(%2) , %%zmm7    \n\t" // x
        "kxorw   %%k5   , %%k5   , %%k5                 \n\t"
        ".p2align 4                          \n\t"
        "1:                     			 \n\t"
        "vmovups	(%3,%0,8), %%zmm12       \n\t" // y
        "vmovups	  %%zmm12, %%zmm13       \n\t"
        "vmovups	(%4,%0,8), %%zmm16       \n\t" // A1
        "vmovups	(%5,%0,8), %%zmm17       \n\t" // A2
        "vmovups	(%6,%0,8), %%zmm18       \n\t" // A3
        "vmovups	(%7,%0,8), %%zmm19       \n\t" // A4

        "vfmadd231pd  %%zmm16, %%zmm4 , %%zmm12             \n\t"
        "vfmadd231pd  %%zmm16, %%zmm4 , %%zmm13             \n\t"
        "vfmadd231pd  %%zmm17, %%zmm5 , %%zmm12             \n\t"
        "vfmadd231pd  %%zmm17, %%zmm5 , %%zmm13             \n\t"
        "vfmadd231pd  %%zmm18, %%zmm6 , %%zmm12             \n\t"
        "vfmadd231pd  %%zmm18, %%zmm6 , %%zmm13             \n\t"
        "vfmadd231pd  %%zmm19, %%zmm7 , %%zmm12             \n\t"
        "vfmadd231pd  %%zmm19, %%zmm7 , %%zmm13             \n\t"

        "vmovups  64(%3,%0,8), %%zmm14       \n\t" // y
        "vmovups	  %%zmm14, %%zmm15       \n\t"
        "vmovups  64(%4,%0,8), %%zmm16       \n\t" // A1
        "vmovups  64(%5,%0,8), %%zmm17       \n\t" // A2
        "vmovups  64(%6,%0,8), %%zmm18       \n\t" // A3
        "vmovups  64(%7,%0,8), %%zmm19       \n\t" // A4

        "vfmadd231pd  %%zmm16, %%zmm4 , %%zmm14             \n\t"
        "vfmadd231pd  %%zmm16, %%zmm4 , %%zmm15             \n\t"
        "vfmadd231pd  %%zmm17, %%zmm5 , %%zmm14             \n\t"
        "vfmadd231pd  %%zmm17, %%zmm5 , %%zmm15             \n\t"
        "vfmadd231pd  %%zmm18, %%zmm6 , %%zmm14             \n\t"
        "vfmadd231pd  %%zmm18, %%zmm6 , %%zmm15             \n\t"
        "vfmadd231pd  %%zmm19, %%zmm7 , %%zmm14             \n\t"
        "vfmadd231pd  %%zmm19, %%zmm7 , %%zmm15             \n\t"


        "vmovups 128(%3,%0,8), %%zmm20       \n\t" // y
        "vmovups	  %%zmm20, %%zmm21       \n\t"
        "vmovups 128(%4,%0,8), %%zmm16       \n\t" // A1
        "vmovups 128(%5,%0,8), %%zmm17       \n\t" // A2
        "vmovups 128(%6,%0,8), %%zmm18       \n\t" // A3
        "vmovups 128(%7,%0,8), %%zmm19       \n\t" // A4

        "vfmadd231pd  %%zmm16, %%zmm4 , %%zmm20             \n\t"
        "vfmadd231pd  %%zmm16, %%zmm4 , %%zmm21             \n\t"
        "vfmadd231pd  %%zmm17, %%zmm5 , %%zmm20             \n\t"
        "vfmadd231pd  %%zmm17, %%zmm5 , %%zmm21             \n\t"
        "vfmadd231pd  %%zmm18, %%zmm6 , %%zmm20             \n\t"
        "vfmadd231pd  %%zmm18, %%zmm6 , %%zmm21             \n\t"
        "vfmadd231pd  %%zmm19, %%zmm7 , %%zmm20             \n\t"
        "vfmadd231pd  %%zmm19, %%zmm7 , %%zmm21             \n\t"

        "vmovups 192(%3,%0,8), %%zmm22       \n\t" // y
        "vmovups	  %%zmm22, %%zmm23       \n\t"
        "vmovups 192(%4,%0,8), %%zmm16       \n\t" // A1
        "vmovups 192(%5,%0,8), %%zmm17       \n\t" // A2
        "vmovups 192(%6,%0,8), %%zmm18       \n\t" // A3
        "vmovups 192(%7,%0,8), %%zmm19       \n\t" // A4

        "vfmadd231pd  %%zmm16, %%zmm4 , %%zmm22             \n\t"
        "vfmadd231pd  %%zmm16, %%zmm4 , %%zmm23             \n\t"
        "vfmadd231pd  %%zmm17, %%zmm5 , %%zmm22             \n\t"
        "vfmadd231pd  %%zmm17, %%zmm5 , %%zmm23             \n\t"
        "vfmadd231pd  %%zmm18, %%zmm6 , %%zmm22             \n\t"
        "vfmadd231pd  %%zmm18, %%zmm6 , %%zmm23             \n\t"
        "vfmadd231pd  %%zmm19, %%zmm7 , %%zmm22             \n\t"
        "vfmadd231pd  %%zmm19, %%zmm7 , %%zmm23             \n\t"

        "vmovups 256(%3,%0,8), %%zmm24       \n\t" // y
        "vmovups	  %%zmm24, %%zmm25       \n\t"
        "vmovups 256(%4,%0,8), %%zmm16       \n\t" // A1
        "vmovups 256(%5,%0,8), %%zmm17       \n\t" // A2
        "vmovups 256(%6,%0,8), %%zmm18       \n\t" // A3
        "vmovups 256(%7,%0,8), %%zmm19       \n\t" // A4

        "vfmadd231pd  %%zmm16, %%zmm4 , %%zmm24             \n\t"
        "vfmadd231pd  %%zmm16, %%zmm4 , %%zmm25             \n\t"
        "vfmadd231pd  %%zmm17, %%zmm5 , %%zmm24             \n\t"
        "vfmadd231pd  %%zmm17, %%zmm5 , %%zmm25             \n\t"
        "vfmadd231pd  %%zmm18, %%zmm6 , %%zmm24             \n\t"
        "vfmadd231pd  %%zmm18, %%zmm6 , %%zmm25             \n\t"
        "vfmadd231pd  %%zmm19, %%zmm7 , %%zmm24             \n\t"
        "vfmadd231pd  %%zmm19, %%zmm7 , %%zmm25             \n\t"

        "vpcmpeqd     %%zmm12, %%zmm13, %%k0                \n\t"
        "vpcmpeqd     %%zmm14, %%zmm15, %%k1                \n\t"
        "vpcmpeqd     %%zmm20, %%zmm21, %%k2                \n\t"
        "vpcmpeqd     %%zmm22, %%zmm23, %%k3                \n\t"
        "vpcmpeqd     %%zmm24, %%zmm25, %%k4                \n\t"

        "kandw         %%k0   , %%k2   , %%k0               \n\t" // red - 1 - 2 -> 2
        "kandw         %%k1   , %%k3   , %%k1               \n\t" // red - 1 - 2 -> 2
        "kandw         %%k0   , %%k4   , %%k0               \n\t" // red - 1 - 2 -> 2
        "kandw         %%k0   , %%k1   , %%k0               \n\t" // red - 1 - 2 -> 2
        "kortestw       %%k0   , %%k5                       \n\t"
        "jnc		2f		                 \n\t"

        "vmovups	  %%zmm12, (%3,%0,8)                    \n\t"
        "vmovups	  %%zmm14, 64(%3,%0,8)                    \n\t"
        "vmovups	  %%zmm20, 128(%3,%0,8)                    \n\t"
        "vmovups	  %%zmm22, 192(%3,%0,8)                    \n\t"
        "vmovups	  %%zmm24, 256(%3,%0,8)                    \n\t"

        "addq		$40, %0	  	 	         \n\t"
        "subq		$40, %1	  	 	         \n\t"
        "jnz		1b		                 \n\t"

        "2:                     			 \n\t"
        "vzeroupper			 \n\t"

        :
            "+r" (i),	// 0	
            "+r" (n)  	// 1
        :
            "r" (x),    // 2
            "r" (y),    // 3
            "r" (a1),   // 4
            "r" (a2),   // 5
            "r" (a3),   // 6
            "r" (a4)    // 7
        : "cc", 
        "%xmm4", "%xmm5", "%xmm6", "%xmm7",
        "%xmm12", "%xmm13", "%xmm14", "%xmm15",
        "memory"
    );
    return 0;
}

void ftblas_dgemv_n_ft(double *A, double *x, double *y, int m, int n, int lda)
{
    int m4 = m & -4;
    int n8 = n / 40 * 40;
    int i = 0, j, ii;
    int n8_left = (n/40 >= 2) ? n / 40 * 40 - 32 : 0;

    double *ptr_a1, *ptr_a2, *ptr_a3, *ptr_a4;
    double *ptr_ai;
    double *ptr_x, *ptr_y;
    register int n2 = lda + lda;

    ptr_y = y;
    ptr_ai = A;

    for (j = 0; j < m4; j += 4)
    {
        ptr_a1 = ptr_ai + j * lda;
        ptr_x = x + j;
        ptr_a2 = ptr_a1 + lda;
        ptr_a3 = ptr_a1 + n2;
        ptr_a4 = ptr_a2 + n2;

        if (n8_left)
        {
            long err_num = dgemv_n_ft_kernel(n8_left, ptr_a1, ptr_a2, ptr_a3, ptr_a4, ptr_x, ptr_y);
            if (err_num) printf("detected and corrected error: %ld\n", err_num);
        }

        if (n8_left == n) continue;

        i = n8_left;

        register double xj0 = x[j];
        register double xj1 = x[j+1];
        register double xj2 = x[j+2];
        register double xj3 = x[j+3];

        while (i < n)
        {
            register int in0 = j * lda + i;
            y[i] += A[in0] * xj0 + A[in0 + lda] * xj1 + A[in0 + n2] * xj2 + A[in0 + lda * 3] * xj3;
            i++;
        }
    }

    while (j < m)
    {
        for (i = 0; i < n; i++)
        {
            y[i] += A[j * lda + i] * x[j];
        }
        j++;
    }

}