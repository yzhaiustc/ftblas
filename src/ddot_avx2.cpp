#include <stdio.h>
#include <stdlib.h>
#include "../include/ftblas.h"

#define PREF_X(pdist)\
    "prefetcht0 "#pdist"(%1);"\

#define PREF_Y(pdist)\
    "prefetcht0 "#pdist"(%2);"\

#define INIT_n1(no)\
	"vxorpd %%ymm"#no",%%ymm"#no",%%ymm"#no";"\

#define LOAD_X(no,ox)\
	"vmovups "#ox"(%1),%%ymm"#no";"\

#define LOAD_Y(no,oy)\
	"vmovups "#oy"(%2),%%ymm"#no";"\

#define FMA(a,b,c)\
	"vfmadd231pd %%ymm"#a",%%ymm"#b",%%ymm"#c";"\

#define INIT_n4_1\
	INIT_n1(0)\
	INIT_n1(1)\
	INIT_n1(2)\
	INIT_n1(3)\

#define PROLOG\
	LOAD_X(8,0)\
	LOAD_Y(9,0)\
	LOAD_X(10,32)\
	FMA(8,9,0)\
	LOAD_Y(11,32)\
	LOAD_X(8,64)\
	FMA(10,11,1)\
	LOAD_Y(9,64)\
	LOAD_X(10,96)\


#define EPILOG\
	FMA(8,9,2)\
	LOAD_Y(11,-32)\
	FMA(10,11,3)\

#define UNROLL_1(fx,fy,f1f,ly,oy,lx,ox)\
	FMA(fx,fy,f1f)\
	LOAD_Y(ly,oy)\
	LOAD_X(lx,ox)\

#define POST_PROCESS\
	"vaddpd %%ymm0,%%ymm1,%%ymm0;"\
	"vaddpd %%ymm2,%%ymm3,%%ymm2;"\
	"vaddpd %%ymm0,%%ymm2,%%ymm0;"\
	"vextractf128 $1,%%ymm0,%%xmm1;"\
	"vaddpd %%xmm0,%%xmm1,%%xmm0;"\
	"vhaddpd %%xmm0,%%xmm0,%%xmm0;"\
	"vmovsd %%xmm0,(%3);"\


#define ORI_KERNEL {\
	__asm__ __volatile__(\
		INIT_n4_1\
		PROLOG\
		"addq $128,%1;"\
		"addq $128,%2;"\
		"subq $16,%0;"\
		"jz 2f;"\
		"1:\n\t"\
		UNROLL_1(8,9,2,11,-32,8,0)\
		UNROLL_1(10,11,3,9,0,10,32)\
		UNROLL_1(8,9,0,11,32,8,64)\
		UNROLL_1(10,11,1,9,64,10,96)\
		"addq $128,%1;"\
		"addq $128,%2;"\
		"subq $16,%0;"\
		"jnz 1b;"\
		"2:\n\t"\
		EPILOG\
		"3:\n\t"\
		POST_PROCESS\
		"vzeroupper;"\
        :"+r"(n),"+r"(ptr_x),"+r"(ptr_y),"+r"(dot)\
        ::"cc","memory","ymm0","ymm1","ymm2","ymm3","ymm4","ymm5","ymm6","ymm7","ymm8","ymm9","ymm10","ymm11","ymm12","ymm13","ymm14","ymm15"\
    );\
}

void ori_ddot_kernel(long n, double *x, double *y, double *dot)
{
  double *ptr_x=x,*ptr_y=y;
  ORI_KERNEL
}

double ftblas_ddot_ori(long int n, double *x, long int inc_x, double *y, long int inc_y)
{
	// it is of users' responsibility to make sure
	// length(x) / inc_x >= n && length(y) / inc_y >= n
	double res = 0.;
	if (inc_x != 1 || inc_y != 1)
	{
		long int cnt_i = 0, n4 = n & -4;
		long int inc_x2 = 2 * inc_x, inc_x3 = 3 * inc_x, inc_x4 = 4 * inc_x;
		long int inc_y2 = 2 * inc_y, inc_y3 = 3 * inc_y, inc_y4 = 4 * inc_y;
		double res_tmp1 = 0., res_tmp2 = 0., res_tmp3 = 0., res_tmp4 = 0.;
		double *ptr_x = x, *ptr_y = y;
		for (cnt_i = 0; cnt_i < n4; cnt_i += 4)
		{
			res_tmp1 += *(ptr_x) * *(ptr_y);
			res_tmp2 += *(ptr_x+inc_x) * *(ptr_y+inc_y);
			res_tmp3 += *(ptr_x+inc_x2) * *(ptr_y+inc_y2);
			res_tmp4 += *(ptr_x+inc_x3) * *(ptr_y+inc_y3);
			ptr_x += inc_x4; ptr_y += inc_y4;
		}
		res += res_tmp1 + res_tmp2 + res_tmp3 + res_tmp4;

		if (n4 == n) return res;
		for (cnt_i = n4; cnt_i < n; cnt_i++) 
		{
			res += *(ptr_x) * *(ptr_y);
			ptr_x+=inc_x; ptr_y+=inc_y;
		}
		return res;
	}

	long int n16 = n & -16;
	double res_dot = 0.;
	if (n16) 
  	{
    	ori_ddot_kernel(n16, x, y , &res_dot);
  	}
	long int cnt_i = n16;
	double *ptr_x = x + n16, *ptr_y = y + n16;
	res = res_dot;
	if (n16 == n) return res;
	for (; cnt_i < n; cnt_i++)
	{
		res += *(ptr_x) * *(ptr_y);
		ptr_x++; ptr_y++;
	}
	return res;
}