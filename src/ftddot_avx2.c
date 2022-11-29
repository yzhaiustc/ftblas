#include <stdio.h>
#include <stdlib.h>

#define INIT_n1(no)\
	"vxorpd %%ymm"#no",%%ymm"#no",%%ymm"#no";"\

#define LOAD_X(no,ox)\
	"vmovups "#ox"(%2),%%ymm"#no";"\

#define LOAD_Y(no,oy)\
	"vmovups "#oy"(%3),%%ymm"#no";"\

#define FMA(a,b,c)\
	"vfmadd231pd %%ymm"#a",%%ymm"#b",%%ymm"#c";"\

#define FMA_2(a,b,c,d)\
	"vfmadd231pd %%ymm"#a",%%ymm"#b",%%ymm"#c";"\
	"vfmadd231pd %%ymm"#a",%%ymm"#b",%%ymm"#d";"\

#define BACK(a,b)\
	"vmovups %%ymm"#a",%%ymm"#b";"

#define COMPARE(a,b,c)\
	"vpcmpeqd %%ymm"#a",%%ymm"#b",%%ymm"#c";"\

#define EXTRACT(ymm,reg)\
	"vmovmskpd %%ymm"#ymm","#reg";"\

#define INIT_n4_1\
	INIT_n1(0)\
	INIT_n1(1)\
	INIT_n1(2)\
	INIT_n1(3)\

#define PROLOG\
	LOAD_X(8,0)\
	LOAD_Y(9,0)\
	LOAD_X(10,32)\
	FMA_2(8,9,0,4)\
	LOAD_Y(11,32)\
	LOAD_X(8,64)\
	COMPARE(0,4,12)\
	FMA_2(10,11,1,5)\
	LOAD_Y(9,64)\
	BACK(2,14)\
	LOAD_X(10,96)\


#define EPILOG\
	EXTRACT(12,%%r10d)\
	COMPARE(1,5,13)\
	FMA_2(8,9,2,6)\
	LOAD_Y(11,-32)\
	EXTRACT(13,%%r11d)\
	COMPARE(2,6,14)\
	FMA_2(10,11,3,7)\
	EXTRACT(14,%%r12d)\
	COMPARE(3,7,15)\
	EXTRACT(15,%%r13d)\

#define UNROLL_1(cmp_e,ex,f1c,f2c,cmp_f,fx,fy,f1f,f2f,ly,oy,lx,ox)\
	EXTRACT(cmp_e,ex)\
	COMPARE(f1c,f2c,cmp_f)\
	FMA_2(fx,fy,f1f,f2f)\
	LOAD_Y(ly,oy)\
	LOAD_X(lx,ox)\

#define POST_PROCESS\
	"vaddpd %%ymm0,%%ymm1,%%ymm0;"\
	"vaddpd %%ymm2,%%ymm3,%%ymm2;"\
	"vaddpd %%ymm0,%%ymm2,%%ymm0;"\
	"vextractf128 $1,%%ymm0,%%xmm1;"\
	"vaddpd %%xmm0,%%xmm1,%%xmm0;"\
	"vhaddpd %%xmm0,%%xmm0,%%xmm0;"\
	"vmovsd %%xmm0,(%4);"\

#define INIT_n4_2\
	INIT_n1(4)\
	INIT_n1(5)\
	INIT_n1(6)\
	INIT_n1(7)\

#define FT_KERNEL {\
	__asm__ __volatile__(\
		INIT_n4_1\
		INIT_n4_2\
		"mov $15,%%r14d;"\
		PROLOG\
		"addq $128,%2;"\
		"addq $128,%3;"\
		"subq $16,%0;"\
		"jz 2f;"\
		"1:\n\t"\
		UNROLL_1(12,%%r10d,1,5,13,8,9,2,6,11,-32,8,0)\
		UNROLL_1(13,%%r11d,2,6,14,10,11,3,7,9,0,10,32)\
		"and %%r11d,%%r10d;"\
		UNROLL_1(14,%%r12d,3,7,15,8,9,0,4,11,32,8,64)\
		"and %%r12d,%%r10d;"\
		UNROLL_1(15,%%r13d,0,4,12,10,11,1,5,9,64,10,96)\
		"and %%r13d,%%r10d;"\
		"addq $128,%2;"\
		"and %%r13d,%%r10d;"\
		"addq $128,%3;"\
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
        :"+r"(n),"+r"(err_num),"+r"(ptr_x),"+r"(ptr_y),"+r"(dot)\
        ::"cc","memory","ymm0","ymm1","ymm2","ymm3","ymm4","ymm5","ymm6","ymm7","ymm8","ymm9","ymm10","ymm11","ymm12","ymm13","ymm14","ymm15","r10","r11","r12","r13","r14"\
    );\
}

static long ft_ddot_kernel(long n, double *x, double *y, double *dot)
{
  long register err_num = 0;
  double *ptr_x=x,*ptr_y=y;
  FT_KERNEL
  return err_num;
}

double ft_ddot_compute(long int n, double *x, long int inc_x, double *y, long int inc_y)
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
	long int err_num;
	if (n16) 
  	{
    	err_num = ft_ddot_kernel(n16, x, y , &res_dot);
		if (err_num) printf("detected error number: %ld\n", err_num);
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

// a driver layer useful for threaded version
double ft_ddot(long int n, double *x, long int inc_x, double *y, long int inc_y)
{
	return ft_ddot_compute(n, x, inc_x, y, inc_y);
}