#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INIT_n1(no)\
	"vxorpd %%ymm"#no",%%ymm"#no",%%ymm"#no";"\

#define LOAD_X(no,ox)\
	"vmovups "#ox"(%2),%%ymm"#no";"\

#define FMA(a,b)\
	"vfmadd231pd %%ymm"#a",%%ymm"#a",%%ymm"#b";"\

#define FMA_2(a,b,c)\
	"vfmadd231pd %%ymm"#a",%%ymm"#a",%%ymm"#b";"\
	"vfmadd231pd %%ymm"#a",%%ymm"#a",%%ymm"#c";"\

#define COMPARE(a,b,c)\
	"vpcmpeqd %%ymm"#a",%%ymm"#b",%%ymm"#c";"\

#define EXTRACT(ymm,reg)\
	"vmovmskpd %%ymm"#ymm","#reg";"\

#define POST_PROCESS\
	"vaddpd %%ymm4,%%ymm6,%%ymm4;"\
	"vaddpd %%ymm5,%%ymm7,%%ymm5;"\
	"vaddpd %%ymm4,%%ymm5,%%ymm0;"\
	"vextractf128 $1,%%ymm0,%%xmm1;"\
	"vaddpd %%xmm0,%%xmm1,%%xmm0;"\
	"vhaddpd %%xmm0,%%xmm0,%%xmm0;"\
	"vmovsd %%xmm0,(%3);"\

#define INIT_n4_1\
	INIT_n1(4)\
	INIT_n1(5)\
	INIT_n1(6)\
	INIT_n1(7)\

#define INIT_n4_2\
	INIT_n1(8)\
	INIT_n1(9)\
	INIT_n1(10)\
	INIT_n1(11)\

#define PROLOG\
	LOAD_X(0,0)\
  FMA(0,4)\
  LOAD_X(1,32)\
  FMA(0,8)\
  FMA(1,5)\
  LOAD_X(2,64)\
  COMPARE(4,8,12)\
  FMA(1,9)\
  FMA(2,6)\
  LOAD_X(3,96)\

#define EPILOG\
	EXTRACT(12,%%r10d)\
	COMPARE(5,9,13)\
  FMA(2,10)\
  FMA(3,7)\
  EXTRACT(13,%%r11d)\
  COMPARE(6,10,14)\
  FMA(3,11)\
  EXTRACT(14,%%r12d)\
  COMPARE(7,11,15)\
  EXTRACT(14,%%r13d)\

#define UNROLL_1(e1,e2,c1,c2,c3,f1,f2,f3,f4,l1,l2)\
	EXTRACT(e1,e2)\
	COMPARE(c1,c2,c3)\
  FMA(f1,f2)\
  FMA(f3,f4)\
  LOAD_X(l1,l2)\

#define FT_KERNEL {\
	__asm__ __volatile__(\
		INIT_n4_1\
		INIT_n4_2\
		"mov $15,%%r14d;"\
		PROLOG\
		"addq $128,%2;"\
		"subq $16,%0;"\
		"jz 2f;"\
		"1:\n\t"\
    UNROLL_1(12,%%r10d,5,9,13,2,10,3,7,0,0)\
    UNROLL_1(13,%%r11d,6,10,14,3,11,0,4,1,32)\
		"and %%r11d,%%r10d;"\
    UNROLL_1(14,%%r12d,7,11,15,0,8,1,5,2,64)\
		"and %%r12d,%%r10d;"\
    UNROLL_1(15,%%r13d,4,8,12,1,9,2,6,3,96)\
		"and %%r13d,%%r10d;"\
		"addq $128,%2;"\
		"and %%r13d,%%r10d;"\
		"cmp %%r14d,%%r10d;"\
		"jne 4f;"\
		"subq $16,%0;"\
		"jnz 1b;"\
		"2:\n\t"\
		EPILOG\
		"3:\n\t"\
		POST_PROCESS\
		"jmp 5f;"\
		"4:\n\t"\
		"addq $1,%1;"\
		"5:\n\t"\
		"vzeroupper;"\
        :"+r"(n),"+r"(err_num),"+r"(ptr_x),"+r"(res)\
        ::"cc","memory","ymm0","ymm1","ymm2","ymm3","ymm4","ymm5","ymm6","ymm7","ymm8","ymm9","ymm10","ymm11","ymm12","ymm13","ymm14","ymm15","r10","r11","r12","r13","r14"\
    );\
}

static long int dnrm2_kernel(long n, double *x, double *res)
{
  long register err_num = 0;
  double *ptr_x=x;
  FT_KERNEL
  return err_num;
}

double ft_dnrm2(long int n, double *x, long int inc_x)
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
        long int err_num = dnrm2_kernel(n16, x, &res);
        if (err_num) printf("detected error number : %ld\n", err_num);
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