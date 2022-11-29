#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INIT_n1(no)\
	"vxorpd %%ymm"#no",%%ymm"#no",%%ymm"#no";"\

#define LOAD_X(no,ox)\
	"vmovups "#ox"(%1),%%ymm"#no";"\

#define FMA(a,b)\
	"vfmadd231pd %%ymm"#a",%%ymm"#a",%%ymm"#b";"\

#define POST_PROCESS\
	"vaddpd %%ymm4,%%ymm6,%%ymm4;"\
	"vaddpd %%ymm5,%%ymm7,%%ymm5;"\
	"vaddpd %%ymm4,%%ymm5,%%ymm0;"\
	"vextractf128 $1,%%ymm0,%%xmm1;"\
	"vaddpd %%xmm0,%%xmm1,%%xmm0;"\
	"vhaddpd %%xmm0,%%xmm0,%%xmm0;"\
	"vmovsd %%xmm0,(%2);"\

#define INIT_n4_1\
	INIT_n1(4)\
	INIT_n1(5)\
	INIT_n1(6)\
	INIT_n1(7)\

#define PROLOG\
	LOAD_X(0,0)\
  FMA(0,4)\
  LOAD_X(1,32)\
  FMA(1,5)\
  LOAD_X(2,64)\
  FMA(2,6)\
  LOAD_X(3,96)\

#define EPILOG\
  FMA(3,7)\

#define UNROLL_1(f1,f2,l1,l2)\
  FMA(f1,f2)\
  LOAD_X(l1,l2)\

#define ORI_KERNEL {\
	__asm__ __volatile__(\
		INIT_n4_1\
		PROLOG\
		"addq $128,%1;"\
		"subq $16,%0;"\
		"jz 2f;"\
		"1:\n\t"\
    UNROLL_1(3,7,0,0)\
    UNROLL_1(0,4,1,32)\
    UNROLL_1(1,5,2,64)\
    UNROLL_1(2,6,3,96)\
		"addq $128,%1;"\
		"subq $16,%0;"\
		"jnz 1b;"\
		"2:\n\t"\
		EPILOG\
		"3:\n\t"\
		POST_PROCESS\
		"vzeroupper;"\
        :"+r"(n),"+r"(ptr_x),"+r"(res)\
        ::"cc","memory","ymm0","ymm1","ymm2","ymm3","ymm4","ymm5","ymm6","ymm7","ymm8","ymm9","ymm10","ymm11","ymm12","ymm13","ymm14","ymm15"\
    );\
}

void dnrm2_kernel(long n, double *x, double *res)
{
  double *ptr_x=x;
  ORI_KERNEL
}

double ori_dnrm2(long int n, double *x, long int inc_x)
{
    if (inc_x != 1)
    {
        long int cnt_i;
        double *ptr_x = x;
        long int inc_x2 = 2 * inc_x, inc_x3 = 3 * inc_x, inc_x4 = 4 * inc_x;
        long int n4 = n & -4;
        double res = 0., sum1 = 0., sum2 = 0., sum3 = 0., sum4 = 0.;
        for (cnt_i = 0; cnt_i < n4; cnt_i+=4)
        {
            sum1 += *ptr_x * *ptr_x;
            sum2 += *(ptr_x+inc_x) * *(ptr_x+inc_x);
            sum3 += *(ptr_x+inc_x2) * *(ptr_x+inc_x2);
            sum4 += *(ptr_x+inc_x3) * *(ptr_x+inc_x3);
            ptr_x+=inc_x4;
        }
        res += (sum1 + sum2 + sum3 + sum4);
        if (n4 == n) return sqrt(res);
        ptr_x = x + n4;
        for (cnt_i = n4; cnt_i < n; cnt_i++)
        {
            res += *(ptr_x) * *(ptr_x);
            ptr_x+=inc_x;
        }
        return sqrt(res);
    }

    // inc_x==1
    long int cnt_i;
    double *ptr_x;
    long int n16 = n & -16;
    double res = 0.;
    if (n16) 
    {
        dnrm2_kernel(n16, x, &res);
    }
    if (n16 == n) return sqrt(res);
    ptr_x = x + n16;
    for (cnt_i = n16; cnt_i < n; cnt_i++)
    {
        res += *(ptr_x) * *(ptr_x);
        ptr_x++;
    }
    return sqrt(res);
}