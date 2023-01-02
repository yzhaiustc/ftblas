#include "../include/ftblas.h"

static void dgemv_n_kernel(long int n, double *a1, double *a2, double *a3, double *a4, double *x, double *y)
{
    register long int i = 0;
    
    __asm__ __volatile__(
        "vbroadcastsd		(%2) , %%zmm4                    \n\t" // x1
        "vbroadcastsd	   8(%2) , %%zmm5                    \n\t" // x2
        "vbroadcastsd	  16(%2) , %%zmm6                    \n\t" // x3
        "vbroadcastsd	  24(%2) , %%zmm7                    \n\t" // x4

        // Prologue
        "prefetchw        960(%3, %0, 8)                    \n\t" // PY
        "vmovups    (%3,%0,8), %%zmm8                       \n\t" // LY1

        "vmovups      %%zmm8 , %%zmm27                      \n\t" // BY1 - 8 - 27
        "vmovups    (%4,%0,8), %%zmm16                      \n\t" // LA11
        "vmovups  64(%3,%0,8), %%zmm9                       \n\t" // LY2

        "vfmadd231pd  %%zmm16, %%zmm4 , %%zmm27             \n\t" // F11
        "vmovups	(%5,%0,8), %%zmm17                      \n\t" // LA21
        "vmovups      %%zmm9 , %%zmm28                      \n\t" // BY2 - 9 - 28
        "vmovups  64(%4,%0,8), %%zmm16                      \n\t" // LA12
        "vmovups 128(%3,%0,8), %%zmm10                      \n\t" // LY3

        "vfmadd231pd  %%zmm17, %%zmm5 , %%zmm27             \n\t" // F21
        "vmovups	(%6,%0,8), %%zmm18                      \n\t" // LA31
        "vfmadd231pd  %%zmm16, %%zmm4 , %%zmm28             \n\t" // F12
        "vmovups  64(%5,%0,8), %%zmm17                      \n\t" // LA22
        "vmovups      %%zmm10, %%zmm29                      \n\t" // BY3 - 10 - 29
        "vmovups 128(%4,%0,8), %%zmm16                      \n\t" // LA13
        "vmovups 192(%3,%0,8), %%zmm11                      \n\t" // LY4

        "vfmadd231pd  %%zmm18, %%zmm6 , %%zmm27             \n\t" // F31
        "vmovups	(%7,%0,8), %%zmm19                      \n\t" // LA41
        "vfmadd231pd  %%zmm17, %%zmm5 , %%zmm28             \n\t" // F22
        "vmovups  64(%6,%0,8), %%zmm18                      \n\t" // LA32
        "vfmadd231pd  %%zmm16, %%zmm4 , %%zmm29             \n\t" // F13
        "vmovups 128(%5,%0,8), %%zmm17                      \n\t" // LA23
        "vmovups      %%zmm11, %%zmm30                      \n\t" // BY4 - 11 - 30
        "vmovups 192(%4,%0,8), %%zmm16                      \n\t" // LA14
        "vmovups 256(%3,%0,8), %%zmm12                      \n\t" // LY5

        "vfmadd231pd  %%zmm19, %%zmm7 , %%zmm27             \n\t" // F41
        "vfmadd231pd  %%zmm18, %%zmm6 , %%zmm28             \n\t" // F32
        "vmovups  64(%7,%0,8), %%zmm19                      \n\t" // LA42
        "vfmadd231pd  %%zmm17, %%zmm5 , %%zmm29             \n\t" // F23
        "vmovups 128(%6,%0,8), %%zmm18                      \n\t" // LA33
        "vfmadd231pd  %%zmm16, %%zmm4 , %%zmm30             \n\t" // F14
        "vmovups 192(%5,%0,8), %%zmm17                      \n\t" // LA24
        "vmovups      %%zmm12, %%zmm31                      \n\t" // BY5 - 12 - 31
        "vmovups 256(%4,%0,8), %%zmm16                      \n\t" // LA15
        "vpcmpeqd     %%zmm8 , %%zmm27, %%k1                \n\t" // C1
        "vmovups 320(%3,%0,8), %%zmm8                       \n\t" // LY6

        "addq		$48, %0	  	 	                        \n\t"
        "subq		$48, %1	  	 	                        \n\t"
        "jz 		2f		                                \n\t"

        // Loop Body
        ".p2align 4                                         \n\t"
        "1:                     			                \n\t"
        "vmovups	   %%zmm27, -384(%3,%0,8)               \n\t" // S1
        "vfmadd231pd   %%zmm19, %%zmm7 , %%zmm28            \n\t" // F42
        "vfmadd231pd   %%zmm18, %%zmm6 , %%zmm29            \n\t" // F33
        "vmovups -256(%7,%0,8), %%zmm19                     \n\t" // A43
        "vfmadd231pd   %%zmm17, %%zmm5 , %%zmm30            \n\t" // F24
        "vmovups -192(%6,%0,8), %%zmm18                     \n\t" // A34
        "vfmadd231pd   %%zmm16, %%zmm4 , %%zmm31            \n\t" // F15
        "vmovups -128(%5,%0,8), %%zmm17                     \n\t" // A25
        "vmovups       %%zmm8 , %%zmm27                     \n\t" // BY6 - 8 - 27
        "vmovups  -64(%4,%0,8), %%zmm16                     \n\t" // LA16
        "vmovups     (%3,%0,8), %%zmm9                      \n\t" // LY7

        "vmovups	   %%zmm28, -320(%3,%0,8)               \n\t" // S2
        "vfmadd231pd   %%zmm19, %%zmm7 , %%zmm29            \n\t" // F43
        "vfmadd231pd   %%zmm18, %%zmm6 , %%zmm30            \n\t" // F34
        "vmovups -192(%7,%0,8), %%zmm19                     \n\t" // A44
        "vfmadd231pd   %%zmm17, %%zmm5 , %%zmm31            \n\t" // F25
        "vmovups -128(%6,%0,8), %%zmm18                     \n\t" // A35
        "vfmadd231pd   %%zmm16, %%zmm4 , %%zmm27            \n\t" // F16
        "vmovups  -64(%5,%0,8), %%zmm17                     \n\t" // A26
        "vmovups       %%zmm9 , %%zmm28                     \n\t" // BY7 - 9 - 28
        "vmovups     (%4,%0,8), %%zmm16                     \n\t" // LA17
        "vmovups   64(%3,%0,8), %%zmm10                     \n\t" // LY8

        "vmovups	   %%zmm29, -256(%3,%0,8)               \n\t" // S3
        "vfmadd231pd   %%zmm19, %%zmm7 , %%zmm30            \n\t" // F44
        "vfmadd231pd   %%zmm18, %%zmm6 , %%zmm31            \n\t" // F35
        "vmovups -128(%7,%0,8), %%zmm19                     \n\t" // A45
        "vfmadd231pd   %%zmm17, %%zmm5 , %%zmm27            \n\t" // F26
        "vmovups  -64(%6,%0,8), %%zmm18                     \n\t" // A36
        "vfmadd231pd   %%zmm16, %%zmm4 , %%zmm28            \n\t" // F17
        "vmovups     (%5,%0,8), %%zmm17                     \n\t" // A27
        "vmovups       %%zmm10, %%zmm29                     \n\t" // BY8 - 10 - 29
        "vmovups   64(%4,%0,8), %%zmm16                     \n\t" // LA18
        "vmovups  128(%3,%0,8), %%zmm11                     \n\t" // LY9

        "vmovups	   %%zmm30, -192(%3,%0,8)               \n\t" // S4
        "vfmadd231pd   %%zmm19, %%zmm7 , %%zmm31            \n\t" // F45
        "vfmadd231pd   %%zmm18, %%zmm6 , %%zmm27            \n\t" // F36
        "vmovups  -64(%7,%0,8), %%zmm19                     \n\t" // A46
        "vfmadd231pd   %%zmm17, %%zmm5 , %%zmm28            \n\t" // F27
        "vmovups     (%6,%0,8), %%zmm18                     \n\t" // A37
        "vfmadd231pd   %%zmm16, %%zmm4 , %%zmm29            \n\t" // F18
        "vmovups   64(%5,%0,8), %%zmm17                     \n\t" // A28
        "vmovups       %%zmm11, %%zmm30                     \n\t" // BY9 - 11 - 30
        "vmovups  128(%4,%0,8), %%zmm16                     \n\t" // LA19
        "vmovups  192(%3,%0,8), %%zmm12                     \n\t" // LY10

        "vmovups	   %%zmm31, -128(%3,%0,8)               \n\t" // S5
        "vfmadd231pd   %%zmm19, %%zmm7 , %%zmm27            \n\t" // F46
        "vfmadd231pd   %%zmm18, %%zmm6 , %%zmm28            \n\t" // F37
        "vmovups     (%7,%0,8), %%zmm19                     \n\t" // A47
        "vfmadd231pd   %%zmm17, %%zmm5 , %%zmm29            \n\t" // F28
        "vmovups   64(%6,%0,8), %%zmm18                     \n\t" // A38
        "vfmadd231pd   %%zmm16, %%zmm4 , %%zmm30            \n\t" // F19
        "vmovups  128(%5,%0,8), %%zmm17                     \n\t" // A29
        "vmovups       %%zmm12, %%zmm31                     \n\t" // BY10 - 12 - 31
        "vmovups  192(%4,%0,8), %%zmm16                     \n\t" // LA10
        "vmovups  256(%3,%0,8), %%zmm8                      \n\t" // LY11

        "addq		$40, %0	  	 	                        \n\t"
        "subq		$40, %1	  	 	                        \n\t"
        "jnz		1b		                                \n\t"


        "2:                                                 \n\t"
        "vmovups	   %%zmm27, -384(%3,%0,8)               \n\t" // S1
        "vfmadd231pd   %%zmm19, %%zmm7 , %%zmm28            \n\t" // F42
        "vfmadd231pd   %%zmm18, %%zmm6 , %%zmm29            \n\t" // F33
        "vmovups -256(%7,%0,8), %%zmm19                     \n\t" // A43
        "vfmadd231pd   %%zmm17, %%zmm5 , %%zmm30            \n\t" // F24
        "vmovups -192(%6,%0,8), %%zmm18                     \n\t" // A34
        "vfmadd231pd   %%zmm16, %%zmm4 , %%zmm31            \n\t" // F15
        "vmovups -128(%5,%0,8), %%zmm17                     \n\t" // A25
        "vmovups       %%zmm8 , %%zmm27                     \n\t" // BY6 - 8 - 27
        "vmovups  -64(%4,%0,8), %%zmm16                     \n\t" // LA16

        "vmovups	   %%zmm28, -320(%3,%0,8)               \n\t" // S2
        "vfmadd231pd   %%zmm19, %%zmm7 , %%zmm29            \n\t" // F43
        "vfmadd231pd   %%zmm18, %%zmm6 , %%zmm30            \n\t" // F34
        "vmovups -192(%7,%0,8), %%zmm19                     \n\t" // A44
        "vfmadd231pd   %%zmm17, %%zmm5 , %%zmm31            \n\t" // F25
        "vmovups -128(%6,%0,8), %%zmm18                     \n\t" // A35
        "vfmadd231pd   %%zmm16, %%zmm4 , %%zmm27            \n\t" // F16
        "vmovups  -64(%5,%0,8), %%zmm17                     \n\t" // A26

        "vmovups	   %%zmm29, -256(%3,%0,8)               \n\t" // S3
        "vfmadd231pd   %%zmm19, %%zmm7 , %%zmm30            \n\t" // F44
        "vfmadd231pd   %%zmm18, %%zmm6 , %%zmm31            \n\t" // F35
        "vmovups  -128(%7,%0,8), %%zmm19                     \n\t" // A45
        "vfmadd231pd   %%zmm17, %%zmm5 , %%zmm27            \n\t" // F26
        "vmovups  -64(%6,%0,8), %%zmm18                     \n\t" // A36

        "vmovups	   %%zmm30, -192(%3,%0,8)               \n\t" // S4
        "vfmadd231pd   %%zmm19, %%zmm7 , %%zmm31            \n\t" // F45
        "vfmadd231pd   %%zmm18, %%zmm6 , %%zmm27            \n\t" // F36
        "vmovups  -64(%7,%0,8), %%zmm19                     \n\t" // A46

        "vmovups	   %%zmm31, -128(%3,%0,8)               \n\t" // S5
        "vfmadd231pd   %%zmm19, %%zmm7 , %%zmm27            \n\t" // F46

        "vmovups       %%zmm27, -64(%3,%0,8)                \n\t" // S1

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
}

void ftblas_dgemv_n_ori(double *A, double *x, double *y, int m, int n, int lda)
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
            dgemv_n_kernel(n8_left, ptr_a1, ptr_a2, ptr_a3, ptr_a4, ptr_x, ptr_y);
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