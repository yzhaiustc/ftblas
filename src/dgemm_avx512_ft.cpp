//#define TIMER 1
#include "immintrin.h"
#include <stdint.h>

#include "../include/ftblas.h"

#define A(i,j) A[(i)+(j)*LDA]
#define B(i,j) B[(i)+(j)*LDB]
#define C(i,j) C[(i)+(j)*LDC]
#define OUT_M_BLOCKING 1152
#define OUT_N_BLOCKING 9216
#define M_BLOCKING 192
#define N_BLOCKING 96
#define K_BLOCKING 384
#define THRESHOLD 1e-2
// #define OUT_M_BLOCKING 24
// #define OUT_N_BLOCKING 9216
// #define M_BLOCKING 24
// #define N_BLOCKING 8
// #define K_BLOCKING 24


void scale_c(double *C,int M, int N, int LDC, double scalar, double *chk_c_col, double *chk_c_row){
    int m_count,n_count;
    int M8=M&-8,N4=N&-4,LDC2=LDC<<1,LDC3=LDC2+LDC,LDC4=LDC<<2;
    __m512d vscalar = _mm512_set1_pd(scalar), vec1, vec2, vec3,vec4;
    __m512d vtmp1, vtmp2, vtmp3, vtmp4;
    double *c_ptr_base1 = C,*c_ptr_base2 = C+LDC,*c_ptr_base3 = C+LDC2,*c_ptr_base4 = C+LDC3;
    double *c_ptr_dyn1,*c_ptr_dyn2,*c_ptr_dyn3,*c_ptr_dyn4;
    double *c_ptr_col, *c_ptr_row;
    double stmp1, stmp2, stmp3, stmp4;
    for (n_count=0;n_count<N4;n_count+=4){
        c_ptr_dyn1 = c_ptr_base1;c_ptr_dyn2 = c_ptr_base2;c_ptr_dyn3 = c_ptr_base3;c_ptr_dyn4 = c_ptr_base4;
        c_ptr_col = chk_c_col; c_ptr_row = chk_c_row + n_count;
        vtmp1=_mm512_setzero_pd();
        vtmp2=_mm512_setzero_pd();
        vtmp3=_mm512_setzero_pd();
        vtmp4=_mm512_setzero_pd();
        stmp1=0.;stmp2=0.;stmp3=0.;stmp4=0.;
        for (m_count=0;m_count<M8;m_count+=8){
            vec1=_mm512_mul_pd(_mm512_loadu_pd(c_ptr_dyn1),vscalar);
            vec2=_mm512_mul_pd(_mm512_loadu_pd(c_ptr_dyn2),vscalar);
            vec3=_mm512_mul_pd(_mm512_loadu_pd(c_ptr_dyn3),vscalar);
            vec4=_mm512_mul_pd(_mm512_loadu_pd(c_ptr_dyn4),vscalar);
            _mm512_storeu_pd(c_ptr_dyn1,vec1);
            _mm512_storeu_pd(c_ptr_dyn2,vec2);
            _mm512_storeu_pd(c_ptr_dyn3,vec3);
            _mm512_storeu_pd(c_ptr_dyn4,vec4);
            vtmp1=_mm512_add_pd(vtmp1,vec1);
            vtmp2=_mm512_add_pd(vtmp2,vec2);
            vtmp3=_mm512_add_pd(vtmp3,vec3);
            vtmp4=_mm512_add_pd(vtmp4,vec4);
            _mm512_storeu_pd(c_ptr_col,_mm512_add_pd(_mm512_loadu_pd(c_ptr_col),vec1+vec2+vec3+vec4));
            c_ptr_col+=8;
            c_ptr_dyn1+=8;c_ptr_dyn2+=8;c_ptr_dyn3+=8;c_ptr_dyn4+=8;
        }
        stmp1=_mm512_reduce_add_pd(vtmp1);
        stmp2=_mm512_reduce_add_pd(vtmp2);
        stmp3=_mm512_reduce_add_pd(vtmp3);
        stmp4=_mm512_reduce_add_pd(vtmp4);
        for (;m_count<M;m_count++){
            double ld1=*c_ptr_dyn1 * scalar;
            double ld2=*c_ptr_dyn2 * scalar;
            double ld3=*c_ptr_dyn3 * scalar;
            double ld4=*c_ptr_dyn4 * scalar;
            *c_ptr_dyn1 =ld1; c_ptr_dyn1++;
            *c_ptr_dyn2 =ld2; c_ptr_dyn2++;
            *c_ptr_dyn3 =ld3; c_ptr_dyn3++;
            *c_ptr_dyn4 =ld4; c_ptr_dyn4++;
            *c_ptr_col += (ld1+ld2+ld3+ld4);c_ptr_col++;
            stmp1+=ld1;stmp2+=ld2;stmp3+=ld3;stmp4+=ld4;
        }
        *c_ptr_row=stmp1;*(c_ptr_row+1)=stmp2;*(c_ptr_row+2)=stmp3;*(c_ptr_row+3)=stmp4;
        c_ptr_base1 += LDC4;c_ptr_base2 += LDC4;c_ptr_base3 += LDC4;c_ptr_base4 += LDC4;
    }
    for (;n_count<N;n_count++){
        c_ptr_dyn1 = c_ptr_base1;
        c_ptr_col = chk_c_col; c_ptr_row = chk_c_row + n_count;
        vtmp1=_mm512_setzero_pd();
        stmp1=0.;
        for (m_count=0;m_count<M8;m_count+=8){
            vec1=_mm512_mul_pd(_mm512_loadu_pd(c_ptr_dyn1),vscalar);
            _mm512_storeu_pd(c_ptr_dyn1,vec1);
            vtmp1=_mm512_add_pd(vtmp1,vec1);
            _mm512_storeu_pd(c_ptr_col,_mm512_add_pd(_mm512_loadu_pd(c_ptr_col),vec1));
            c_ptr_col+=8;
            c_ptr_dyn1+=8;
        }
        stmp1=_mm512_reduce_add_pd(vtmp1);
        for (;m_count<M;m_count++){
            double ld1=*c_ptr_dyn1 * scalar;
            *c_ptr_dyn1 =ld1; c_ptr_dyn1++;
            *c_ptr_col += ld1;c_ptr_col++;
            stmp1+=ld1;
        }
        *c_ptr_row=stmp1;
        c_ptr_base1 += LDC;
    }
}

void packing_b(double *src,double *dst,int leading_dim,int dim_first,int dim_second,double *chk_A,double *chk_B,double *chk_online_C_row,double alpha){
    //dim_first:K,dim_second:N
    double *tosrc1,*tosrc2,*todst;
    double *ptr_chk_b = chk_B, *ptr_chk_a = chk_A, *ptr_chk_c = chk_online_C_row;
    double ld1,ld2,ld_a,ld_c1,ld_c2;
    todst=dst;
    int count_first,count_second,count_sub=dim_second;
    for (count_second=0;count_sub>1;count_second+=2,count_sub-=2){
        tosrc1=src+count_second*leading_dim;tosrc2=tosrc1+leading_dim;
        ptr_chk_b = chk_B;ptr_chk_a = chk_A;
        ld_c1=0.;ld_c2=0.;
        for (count_first=0;count_first<dim_first;count_first++){
            ld1=*tosrc1;ld2=*tosrc2;ld_a=*ptr_chk_a;
            *todst=ld1;tosrc1++;todst++;
            *todst=ld2;tosrc2++;todst++;
            *ptr_chk_b+=(ld1+ld2);
            ld_c1+=(ld1*ld_a);ld_c2+=(ld2*ld_a);
            ptr_chk_b++;ptr_chk_a++;
        }
        ptr_chk_c[0]+=ld_c1*alpha;ptr_chk_c[1]+=ld_c2*alpha;
        ptr_chk_c+=2;
    }
    for (;count_sub>0;count_second++,count_sub-=1){
        tosrc1=src+count_second*leading_dim;
        ptr_chk_b = chk_B;ptr_chk_a = chk_A;
        ld_c1 = 0.;
        for (count_first=0;count_first<dim_first;count_first++){
            ld1=*tosrc1;ld_a = *ptr_chk_a;
            ld_c1 += (ld1 * ld_a);
            *todst=ld1;tosrc1++;todst++;
            *ptr_chk_b += ld1;
            ptr_chk_b++;ptr_chk_a++;
        }
        ptr_chk_c[0]+=ld_c1*alpha;
    }
}

void packing_a(double alpha,double *src, double *dst, int leading_dim, int dim_first, int dim_second,double *chk_B,double *chk_online_C_col){
    //dim_first: M, dim_second: K
    double *tosrc,*todst,*ptr_chk_b,*ptr_chk_c;
    todst=dst;
    int count_first,count_second,count_sub=dim_first;
    __m512d valpha=_mm512_set1_pd(alpha);
    __m128d valpha_128=_mm_set1_pd(alpha);
    for (count_first=0;count_sub>23;count_first+=24,count_sub-=24){
        tosrc=src+count_first;
        ptr_chk_c=chk_online_C_col+count_first;
        __m512d vchkc_1 = _mm512_loadu_pd(ptr_chk_c);
        __m512d vchkc_2 = _mm512_loadu_pd(ptr_chk_c+8);
        __m512d vchkc_3 = _mm512_loadu_pd(ptr_chk_c+16);
        for(count_second=0;count_second<dim_second;count_second++){
            ptr_chk_b=chk_B+count_second;
            __m512d cb = _mm512_set1_pd(*ptr_chk_b);
            __m512d ld1 = _mm512_mul_pd(_mm512_loadu_pd(tosrc),valpha);
            __m512d ld2 = _mm512_mul_pd(_mm512_loadu_pd(tosrc+8),valpha);
            __m512d ld3 = _mm512_mul_pd(_mm512_loadu_pd(tosrc+16),valpha);
            _mm512_store_pd(todst,ld1);
            _mm512_store_pd(todst+8,ld2);
            _mm512_store_pd(todst+16,ld3);
            vchkc_1=_mm512_fmadd_pd(cb,ld1,vchkc_1);
            vchkc_2=_mm512_fmadd_pd(cb,ld2,vchkc_2);
            vchkc_3=_mm512_fmadd_pd(cb,ld3,vchkc_3);
            tosrc+=leading_dim;
            todst+=24;
        }
        _mm512_storeu_pd(ptr_chk_c,vchkc_1);
        _mm512_storeu_pd(ptr_chk_c+8,vchkc_2);
        _mm512_storeu_pd(ptr_chk_c+16,vchkc_3);
    }
    // edge case
    for (;count_sub>7;count_first+=8,count_sub-=8){
        tosrc=src+count_first;
        ptr_chk_c=chk_online_C_col+count_first;
        __m512d vchkc_1 = _mm512_loadu_pd(ptr_chk_c);
        for(count_second=0;count_second<dim_second;count_second++){
            ptr_chk_b=chk_B+count_second;
            __m512d cb = _mm512_set1_pd(*ptr_chk_b);
            __m512d ld1 = _mm512_mul_pd(_mm512_loadu_pd(tosrc),valpha);
            _mm512_store_pd(todst,ld1);
            vchkc_1=_mm512_fmadd_pd(cb,ld1,vchkc_1);
            tosrc+=leading_dim;
            todst+=8;
        }
        _mm512_storeu_pd(ptr_chk_c,vchkc_1);
    }
    for (;count_sub>1;count_first+=2,count_sub-=2){
        tosrc=src+count_first;
        ptr_chk_c=chk_online_C_col+count_first;
        __m128d vchkc_1 = _mm_loadu_pd(ptr_chk_c);
        for(count_second=0;count_second<dim_second;count_second++){
            ptr_chk_b=chk_B+count_second;
            __m128d cb = _mm_set1_pd(*ptr_chk_b);
            __m128d ld1=_mm_mul_pd(_mm_loadu_pd(tosrc),valpha_128);
            _mm_store_pd(todst,ld1);
            vchkc_1=_mm_fmadd_pd(cb,ld1,vchkc_1);
            tosrc+=leading_dim;
            todst+=2;
        }
        _mm_storeu_pd(ptr_chk_c,vchkc_1);
    }
    //TODO: add chksum support for m divisible by 1
    for (;count_sub>0;count_first+=1,count_sub-=1){
        tosrc=src+count_first;
        for(count_second=0;count_second<dim_second;count_second++){
            *todst=(*tosrc)*alpha;
            tosrc+=leading_dim;
            todst++;
        }
    }
}

#define init_m24n1 \
  "vpxorq %%zmm8,%%zmm8,%%zmm8;vpxorq %%zmm9,%%zmm9,%%zmm9;vpxorq %%zmm10,%%zmm10,%%zmm10;"
#define init_m8n1 \
  "vpxorq %%zmm8,%%zmm8,%%zmm8;"
#define init_m2n1 \
  "vpxorq %%xmm8,%%xmm8,%%xmm8;"
#define init_m24n2 \
  init_m24n1 \
  "vpxorq %%zmm11,%%zmm11,%%zmm11;vpxorq %%zmm12,%%zmm12,%%zmm12;vpxorq %%zmm13,%%zmm13,%%zmm13;"
#define init_m8n2 \
  init_m8n1 \
  "vpxorq %%zmm11,%%zmm11,%%zmm11;"
#define init_m2n2 \
  init_m2n1 \
  "vpxorq %%xmm9,%%xmm9,%%xmm9;"
#define init_m8n1_chks(c1,c2) \
  "vpxorq %%zmm"#c1",%%zmm"#c1",%%zmm"#c1";"\
  "vpxorq %%zmm"#c2",%%zmm"#c2",%%zmm"#c2";"
#define init_m2n1_chks(c1,c2) \
  "vpxorq %%xmm"#c1",%%xmm"#c1",%%xmm"#c1";"\
  "vpxorq %%xmm"#c2",%%xmm"#c2",%%xmm"#c2";"
#define init_m24n4 \
  init_m24n2 \
  "vpxorq %%zmm14,%%zmm14,%%zmm14;vpxorq %%zmm15,%%zmm15,%%zmm15;vpxorq %%zmm16,%%zmm16,%%zmm16;"\
  "vpxorq %%zmm17,%%zmm17,%%zmm17;vpxorq %%zmm18,%%zmm18,%%zmm18;vpxorq %%zmm19,%%zmm19,%%zmm19;"
#define init_m8n4 \
  init_m8n2 \
  "vpxorq %%zmm14,%%zmm14,%%zmm14;"\
  "vpxorq %%zmm17,%%zmm17,%%zmm17;"
#define init_m2n4 \
  init_m2n2 \
  "vpxorq %%xmm10,%%xmm10,%%xmm10;"\
  "vpxorq %%xmm11,%%xmm11,%%xmm11;"
#define init_m24n6 \
  init_m24n4 \
  "vpxorq %%zmm20,%%zmm20,%%zmm20;vpxorq %%zmm21,%%zmm21,%%zmm21;vpxorq %%zmm22,%%zmm22,%%zmm22;"\
  "vpxorq %%zmm23,%%zmm23,%%zmm23;vpxorq %%zmm24,%%zmm24,%%zmm24;vpxorq %%zmm25,%%zmm25,%%zmm25;"
#define init_m8n6 \
  init_m8n4 \
  "vpxorq %%zmm20,%%zmm20,%%zmm20;"\
  "vpxorq %%zmm23,%%zmm23,%%zmm23;"
#define init_m2n6 \
  init_m2n4 \
  "vpxorq %%xmm12,%%xmm12,%%xmm12;"\
  "vpxorq %%xmm13,%%xmm13,%%xmm13;"
#define init_m24n8 \
  init_m24n6 \
  "vpxorq %%zmm26,%%zmm26,%%zmm26;vpxorq %%zmm27,%%zmm27,%%zmm27;vpxorq %%zmm28,%%zmm28,%%zmm28;"\
  "vpxorq %%zmm29,%%zmm29,%%zmm29;vpxorq %%zmm30,%%zmm30,%%zmm30;vpxorq %%zmm31,%%zmm31,%%zmm31;"
#define init_m8n8 \
  init_m8n6 \
  "vpxorq %%zmm26,%%zmm26,%%zmm26;"\
  "vpxorq %%zmm29,%%zmm29,%%zmm29;"
#define init_m2n8 \
  init_m2n6 \
  "vpxorq %%xmm14,%%xmm14,%%xmm14;"\
  "vpxorq %%xmm15,%%xmm15,%%xmm15;"
#define save_init_m24 \
  "movq %2,%3; addq $192,%2;"
#define save_init_m8 \
  "movq %2,%3; addq $64,%2;"
#define save_init_m2 \
  "movq %2,%3; addq $16,%2;"
#define save_init_m1 \
  "movq %2,%3; addq $8,%2;"
#define kernel_m24n2_1 \
  "vbroadcastsd (%1),%%zmm4;vfmadd231pd %%zmm1,%%zmm4,%%zmm8; vfmadd231pd %%zmm2,%%zmm4,%%zmm9; vfmadd231pd %%zmm3,%%zmm4,%%zmm10;"\
  "vbroadcastsd 8(%1),%%zmm5;vfmadd231pd %%zmm1,%%zmm5,%%zmm11; vfmadd231pd %%zmm2,%%zmm5,%%zmm12; vfmadd231pd %%zmm3,%%zmm5,%%zmm13;prefetcht0 384(%0);"
#define kernel_m8n2_1 \
  "vbroadcastsd (%1),%%zmm4;vfmadd231pd %%zmm1,%%zmm4,%%zmm8;"\
  "vbroadcastsd 8(%1),%%zmm5;vfmadd231pd %%zmm1,%%zmm5,%%zmm11;"
#define kernel_m2n2_1 \
  "vmovddup (%1),%%xmm4;vfmadd231pd %%xmm1,%%xmm4,%%xmm8;"\
  "vmovddup 8(%1),%%xmm5;vfmadd231pd %%xmm1,%%xmm5,%%xmm9;"
#define kernel_m24n2_2 \
  "vbroadcastsd (%1,%%r11,1),%%zmm4;vfmadd231pd %%zmm1,%%zmm4,%%zmm14; vfmadd231pd %%zmm2,%%zmm4,%%zmm15; vfmadd231pd %%zmm3,%%zmm4,%%zmm16;"\
  "vbroadcastsd 8(%1,%%r11,1),%%zmm5;vfmadd231pd %%zmm1,%%zmm5,%%zmm17; vfmadd231pd %%zmm2,%%zmm5,%%zmm18; vfmadd231pd %%zmm3,%%zmm5,%%zmm19;"
#define kernel_m8n2_2 \
  "vbroadcastsd (%1,%%r11,1),%%zmm4;vfmadd231pd %%zmm1,%%zmm4,%%zmm14;"\
  "vbroadcastsd 8(%1,%%r11,1),%%zmm5;vfmadd231pd %%zmm1,%%zmm5,%%zmm17;"
#define kernel_m2n2_2 \
  "vmovddup (%1,%%r11,1),%%xmm4;vfmadd231pd %%xmm1,%%xmm4,%%xmm10;"\
  "vmovddup 8(%1,%%r11,1),%%xmm5;vfmadd231pd %%xmm1,%%xmm5,%%xmm11;"
#define kernel_m24n2_3 \
  "vbroadcastsd (%1,%%r11,2),%%zmm4;vfmadd231pd %%zmm1,%%zmm4,%%zmm20; vfmadd231pd %%zmm2,%%zmm4,%%zmm21; vfmadd231pd %%zmm3,%%zmm4,%%zmm22;"\
  "vbroadcastsd 8(%1,%%r11,2),%%zmm5;vfmadd231pd %%zmm1,%%zmm5,%%zmm23; vfmadd231pd %%zmm2,%%zmm5,%%zmm24; vfmadd231pd %%zmm3,%%zmm5,%%zmm25;prefetcht0 448(%0);"
#define kernel_m8n2_3 \
  "vbroadcastsd (%1,%%r11,2),%%zmm4;vfmadd231pd %%zmm1,%%zmm4,%%zmm20;"\
  "vbroadcastsd 8(%1,%%r11,2),%%zmm5;vfmadd231pd %%zmm1,%%zmm5,%%zmm23;"
#define kernel_m2n2_3 \
  "vmovddup (%1,%%r11,2),%%xmm4;vfmadd231pd %%xmm1,%%xmm4,%%xmm12;"\
  "vmovddup 8(%1,%%r11,2),%%xmm5;vfmadd231pd %%xmm1,%%xmm5,%%xmm13;"
#define kernel_m24n2_4 \
  "vbroadcastsd (%%r12),%%zmm4;vfmadd231pd %%zmm1,%%zmm4,%%zmm26; vfmadd231pd %%zmm2,%%zmm4,%%zmm27; vfmadd231pd %%zmm3,%%zmm4,%%zmm28;"\
  "vbroadcastsd 8(%%r12),%%zmm5;vfmadd231pd %%zmm1,%%zmm5,%%zmm29; vfmadd231pd %%zmm2,%%zmm5,%%zmm30; vfmadd231pd %%zmm3,%%zmm5,%%zmm31;"
#define kernel_m8n2_4 \
  "vbroadcastsd (%%r12),%%zmm4;vfmadd231pd %%zmm1,%%zmm4,%%zmm26;"\
  "vbroadcastsd 8(%%r12),%%zmm5;vfmadd231pd %%zmm1,%%zmm5,%%zmm29;"
#define kernel_m2n2_4 \
  "vmovddup (%%r12),%%xmm4;vfmadd231pd %%xmm1,%%xmm4,%%xmm14;"\
  "vmovddup 8(%%r12),%%xmm5;vfmadd231pd %%xmm1,%%xmm5,%%xmm15;"
#define LOAD_A_COL_m24 \
  "vmovaps (%0),%%zmm1;vmovaps 64(%0),%%zmm2;vmovaps 128(%0),%%zmm3;addq $192,%0;"
#define LOAD_A_COL_m8 \
  "vmovaps (%0),%%zmm1;addq $64,%0;"
#define LOAD_A_COL_m2 \
  "vmovaps (%0),%%xmm1;addq $16,%0;"
#define KERNEL_m24n8 \
  LOAD_A_COL_m24 \
  kernel_m24n2_1 \
  kernel_m24n2_2 \
  kernel_m24n2_3 \
  kernel_m24n2_4 \
  "addq $16,%1;addq $16,%%r12;"
#define KERNEL_m24n2 \
  LOAD_A_COL_m24 \
  kernel_m24n2_1 \
  "addq $16,%1;"
#define KERNEL_m8n8 \
  LOAD_A_COL_m8 \
  kernel_m8n2_1 \
  kernel_m8n2_2 \
  kernel_m8n2_3 \
  kernel_m8n2_4 \
  "addq $16,%1;addq $16,%%r12;"
#define KERNEL_m8n2 \
  LOAD_A_COL_m8 \
  kernel_m8n2_1 \
  "addq $16,%1;"
#define KERNEL_m2n8 \
  LOAD_A_COL_m2 \
  kernel_m2n2_1 \
  kernel_m2n2_2 \
  kernel_m2n2_3 \
  kernel_m2n2_4 \
  "addq $16,%1;addq $16,%%r12;"
#define KERNEL_m24n4 \
  LOAD_A_COL_m24 \
  kernel_m24n2_1 \
  kernel_m24n2_2 \
  "addq $16,%1;"
#define KERNEL_m8n4 \
  LOAD_A_COL_m8 \
  kernel_m8n2_1 \
  kernel_m8n2_2 \
  "addq $16,%1;"
#define KERNEL_m2n2 \
  LOAD_A_COL_m2 \
  kernel_m2n2_1 \
  "addq $16,%1;"
#define KERNEL_m2n4 \
  LOAD_A_COL_m2 \
  kernel_m2n2_1 \
  kernel_m2n2_2 \
  "addq $16,%1;"
#define save_m24n2_para(c1,c2,c3,c4,c5,c6,c7,c8) \
  "vaddpd (%3),%%zmm"#c1",%%zmm"#c1";vaddpd 64(%3),%%zmm"#c2",%%zmm"#c2";vaddpd 128(%3),%%zmm"#c3",%%zmm"#c3";"\
  "vaddpd %%zmm"#c1",%%zmm"#c7", %%zmm"#c7";vmovups %%zmm"#c1",(%3);vmovups %%zmm"#c2",64(%3);vmovups %%zmm"#c3",128(%3);vaddpd %%zmm"#c2",%%zmm"#c7", %%zmm"#c7";"\
  "vaddpd (%3,%4,1),%%zmm"#c4",%%zmm"#c4";vaddpd 64(%3,%4,1),%%zmm"#c5",%%zmm"#c5";vaddpd %%zmm"#c3",%%zmm"#c7", %%zmm"#c7";vaddpd 128(%3,%4,1),%%zmm"#c6",%%zmm"#c6";"\
  "vmovups %%zmm"#c4",(%3,%4,1);vaddpd %%zmm"#c4",%%zmm"#c8", %%zmm"#c8";vmovups %%zmm"#c5",64(%3,%4,1);vaddpd %%zmm"#c5",%%zmm"#c8", %%zmm"#c8";vmovups %%zmm"#c6",128(%3,%4,1);leaq (%3,%4,2),%3;"\
  "vaddpd %%zmm"#c6",%%zmm"#c8", %%zmm"#c8";"
#define save_m8n2_para(c1,c2,c3,c4) \
  "vaddpd (%3),%%zmm"#c1",%%zmm"#c1";"\
  "vaddpd %%zmm"#c1",%%zmm"#c3",%%zmm"#c3";vmovups %%zmm"#c1",(%3);"\
  "vaddpd (%3,%4,1),%%zmm"#c2",%%zmm"#c2";"\
  "vaddpd %%zmm"#c2",%%zmm"#c4",%%zmm"#c4";vmovups %%zmm"#c2",(%3,%4,1);leaq (%3,%4,2),%3;"
#define save_m24n2_1 \
  "vaddpd (%3),%%zmm8,%%zmm8;vaddpd 64(%3),%%zmm9,%%zmm9;vaddpd 128(%3),%%zmm10,%%zmm10;"\
  "vaddpd %%zmm8,%%zmm0, %%zmm0;vmovups %%zmm8,(%3);vmovups %%zmm9,64(%3);vmovups %%zmm10,128(%3);vaddpd %%zmm9,%%zmm0, %%zmm0;"\
  "vaddpd (%3,%4,1),%%zmm11,%%zmm11;vaddpd 64(%3,%4,1),%%zmm12,%%zmm12;vaddpd %%zmm10,%%zmm0, %%zmm0;vaddpd 128(%3,%4,1),%%zmm13,%%zmm13;"\
  "vmovups %%zmm11,(%3,%4,1);vaddpd %%zmm11,%%zmm1, %%zmm1;vmovups %%zmm12,64(%3,%4,1);vaddpd %%zmm12,%%zmm1, %%zmm1;vmovups %%zmm13,128(%3,%4,1);leaq (%3,%4,2),%3;"\
  "vaddpd %%zmm13,%%zmm1, %%zmm1;"
#define save_m8n2_1 \
  "vaddpd (%3),%%zmm8,%%zmm8;"\
  "vmovups %%zmm8,(%3);"\
  "vaddpd (%3,%4,1),%%zmm11,%%zmm11;"\
  "vmovups %%zmm11,(%3,%4,1);leaq (%3,%4,2),%3;"
#define save_m2n2_para(c1,c2,c3,c4) \
  "vaddpd (%3),%%xmm"#c1",%%xmm"#c1";"\
  "vaddpd %%xmm"#c1",%%xmm"#c3",%%xmm"#c3";vmovups %%xmm"#c1",(%3);"\
  "vaddpd (%3,%4,1),%%xmm"#c2",%%xmm"#c2";"\
  "vaddpd %%xmm"#c2",%%xmm"#c4",%%xmm"#c4";vmovups %%xmm"#c2",(%3,%4,1);leaq (%3,%4,2),%3;"
#define save_m2n2_1 \
  "vaddpd (%3),%%xmm8,%%xmm8;"\
  "vmovups %%xmm8,(%3);"\
  "vaddpd (%3,%4,1),%%xmm9,%%xmm9;"\
  "vmovups %%xmm9,(%3,%4,1);leaq (%3,%4,2),%3;"
#define save_m24n2_2 \
  "vaddpd (%3),%%zmm14,%%zmm14;vaddpd 64(%3),%%zmm15,%%zmm15;vaddpd 128(%3),%%zmm16,%%zmm16;"\
  "vmovups %%zmm14,(%3);vmovups %%zmm15,64(%3);vmovups %%zmm16,128(%3);"\
  "vaddpd (%3,%4,1),%%zmm17,%%zmm17;vaddpd 64(%3,%4,1),%%zmm18,%%zmm18;vaddpd 128(%3,%4,1),%%zmm19,%%zmm19;"\
  "vmovups %%zmm17,(%3,%4,1);vmovups %%zmm18,64(%3,%4,1);vmovups %%zmm19,128(%3,%4,1);leaq (%3,%4,2),%3;"
#define save_m8n2_2 \
  "vaddpd (%3),%%zmm14,%%zmm14;"\
  "vmovups %%zmm14,(%3);"\
  "vaddpd (%3,%4,1),%%zmm17,%%zmm17;"\
  "vmovups %%zmm17,(%3,%4,1);leaq (%3,%4,2),%3;"
#define save_m2n2_2 \
  "vaddpd (%3),%%xmm10,%%xmm10;"\
  "vmovups %%xmm10,(%3);"\
  "vaddpd (%3,%4,1),%%xmm11,%%xmm11;"\
  "vmovups %%xmm11,(%3,%4,1);leaq (%3,%4,2),%3;"
#define save_m24n2_3 \
  "vaddpd (%3),%%zmm20,%%zmm20;vaddpd 64(%3),%%zmm21,%%zmm21;vaddpd 128(%3),%%zmm22,%%zmm22;"\
  "vmovups %%zmm20,(%3);vmovups %%zmm21,64(%3);vmovups %%zmm22,128(%3);"\
  "vaddpd (%3,%4,1),%%zmm23,%%zmm23;vaddpd 64(%3,%4,1),%%zmm24,%%zmm24;vaddpd 128(%3,%4,1),%%zmm25,%%zmm25;"\
  "vmovups %%zmm23,(%3,%4,1);vmovups %%zmm24,64(%3,%4,1);vmovups %%zmm25,128(%3,%4,1);leaq (%3,%4,2),%3;"
#define save_m8n2_3 \
  "vaddpd (%3),%%zmm20,%%zmm20;"\
  "vmovups %%zmm20,(%3);"\
  "vaddpd (%3,%4,1),%%zmm23,%%zmm23;"\
  "vmovups %%zmm23,(%3,%4,1);leaq (%3,%4,2),%3;"
#define save_m2n2_3 \
  "vaddpd (%3),%%xmm12,%%xmm12;"\
  "vmovups %%xmm12,(%3);"\
  "vaddpd (%3,%4,1),%%xmm13,%%xmm13;"\
  "vmovups %%xmm13,(%3,%4,1);leaq (%3,%4,2),%3;"
#define save_m24n2_4 \
  "vaddpd (%3),%%zmm26,%%zmm26;vaddpd 64(%3),%%zmm27,%%zmm27;vaddpd 128(%3),%%zmm28,%%zmm28;"\
  "vmovups %%zmm26,(%3);vmovups %%zmm27,64(%3);vmovups %%zmm28,128(%3);"\
  "vaddpd (%3,%4,1),%%zmm29,%%zmm29;vaddpd 64(%3,%4,1),%%zmm30,%%zmm30;vaddpd 128(%3,%4,1),%%zmm31,%%zmm31;"\
  "vmovups %%zmm29,(%3,%4,1);vmovups %%zmm30,64(%3,%4,1);vmovups %%zmm31,128(%3,%4,1);leaq (%3,%4,2),%3;"
#define save_m8n2_4 \
  "vaddpd (%3),%%zmm26,%%zmm26;"\
  "vmovups %%zmm26,(%3);"\
  "vaddpd (%3,%4,1),%%zmm29,%%zmm29;"\
  "vmovups %%zmm29,(%3,%4,1);leaq (%3,%4,2),%3;"
#define save_m2n2_4 \
  "vaddpd (%3),%%xmm14,%%xmm14;"\
  "vmovups %%xmm14,(%3);"\
  "vaddpd (%3,%4,1),%%xmm15,%%xmm15;"\
  "vmovups %%xmm15,(%3,%4,1);leaq (%3,%4,2),%3;"
#define add_col_m8n2(c1,c2,c3)\
  "vaddpd %%zmm"#c1",%%zmm"#c2",%%zmm"#c3";"
#define add_col_m2n2(c1,c2,c3)\
  "vaddpd %%xmm"#c1",%%xmm"#c2",%%xmm"#c3";"
#define trans_and_save_8X8\
  "vunpcklpd %%zmm1,%%zmm0,%%zmm8;vunpckhpd %%zmm1,%%zmm0,%%zmm9;"\
  "vunpcklpd %%zmm3,%%zmm2,%%zmm10;vunpckhpd %%zmm3,%%zmm2,%%zmm11;"\
  "vunpcklpd %%zmm5,%%zmm4,%%zmm12;vunpckhpd %%zmm5,%%zmm4,%%zmm13;"\
  "vunpcklpd %%zmm7,%%zmm6,%%zmm14;vunpckhpd %%zmm7,%%zmm6,%%zmm15;"\
  "vshuff64x2 $0x88,%%zmm12,%%zmm8,%%zmm0;vshuff64x2 $0x88,%%zmm13,%%zmm9,%%zmm1;"\
  "vshuff64x2 $0xdd,%%zmm12,%%zmm8,%%zmm2;vshuff64x2 $0xdd,%%zmm13,%%zmm9,%%zmm3;"\
  "vshuff64x2 $0x88,%%zmm14,%%zmm10,%%zmm4;vshuff64x2 $0x88,%%zmm15,%%zmm11,%%zmm5;"\
  "vshuff64x2 $0xdd,%%zmm14,%%zmm10,%%zmm6;vshuff64x2 $0xdd,%%zmm15,%%zmm11,%%zmm7;"\
  "vshuff64x2 $0x88,%%zmm4,%%zmm0,%%zmm8;vshuff64x2 $0x88,%%zmm5,%%zmm1,%%zmm9;"\
  "vshuff64x2 $0x88,%%zmm6,%%zmm2,%%zmm10;vshuff64x2 $0x88,%%zmm7,%%zmm3,%%zmm11;"\
  "vshuff64x2 $0xdd,%%zmm4,%%zmm0,%%zmm12;vshuff64x2 $0xdd,%%zmm5,%%zmm1,%%zmm13;"\
  "vshuff64x2 $0xdd,%%zmm6,%%zmm2,%%zmm14;vshuff64x2 $0xdd,%%zmm7,%%zmm3,%%zmm15;"\
  "vaddpd %%zmm8,%%zmm9,%%zmm8;vaddpd %%zmm10,%%zmm11,%%zmm10;vaddpd %%zmm8,%%zmm10,%%zmm8;"\
  "vaddpd %%zmm12,%%zmm13,%%zmm12;vaddpd %%zmm14,%%zmm15,%%zmm14;vaddpd %%zmm12,%%zmm14,%%zmm12;"\
  "vaddpd %%zmm8,%%zmm12,%%zmm8; vshuff64x2 $0xd8,%%zmm8,%%zmm8,%%zmm8;vaddpd (%7),%%zmm8,%%zmm8;vmovups %%zmm8,(%7);"
#define TRANS_2X2(c1,c2,c3,c4)\
  "vunpcklpd %%xmm"#c2",%%xmm"#c1",%%xmm"#c3";vunpckhpd %%xmm"#c2",%%xmm"#c1",%%xmm"#c4";"
#define TRANS_2X4(c1,c2,c3,c4)\
  "vunpcklpd %%xmm"#c2",%%xmm"#c1",%%xmm8;vunpckhpd %%xmm"#c2",%%xmm"#c1",%%xmm9;"\
  "vunpcklpd %%xmm"#c4",%%xmm"#c3",%%xmm10;vunpckhpd %%xmm"#c4",%%xmm"#c3",%%xmm11;"
#define TRANS_2X8(c1,c2,c3,c4,c5,c6,c7,c8)\
  "vunpcklpd %%xmm"#c2",%%xmm"#c1",%%xmm8;vunpckhpd %%xmm"#c2",%%xmm"#c1",%%xmm9;"\
  "vunpcklpd %%xmm"#c4",%%xmm"#c3",%%xmm10;vunpckhpd %%xmm"#c4",%%xmm"#c3",%%xmm11;"\
  "vunpcklpd %%xmm"#c6",%%xmm"#c5",%%xmm12;vunpckhpd %%xmm"#c6",%%xmm"#c5",%%xmm13;"\
  "vunpcklpd %%xmm"#c8",%%xmm"#c7",%%xmm14;vunpckhpd %%xmm"#c8",%%xmm"#c7",%%xmm15;"
#define TRANS_4X2(c1,c2,c3,c4)\
  "vunpcklpd %%ymm"#c2",%%ymm"#c1",%%ymm"#c3";vunpckhpd %%ymm"#c2",%%ymm"#c1",%%ymm"#c4";"
#define TRANS_4X4(c1,c2,c3,c4)\
  "vunpcklpd %%ymm"#c2",%%ymm"#c1",%%ymm8;vunpckhpd %%ymm"#c2",%%ymm"#c1",%%ymm9;"\
  "vunpcklpd %%ymm"#c4",%%ymm"#c3",%%ymm10;vunpckhpd %%ymm"#c4",%%ymm"#c3",%%ymm11;"\
  "vshuff64x2 $0,%%ymm10,%%ymm8,%%ymm"#c1";vshuff64x2 $0,%%ymm11,%%ymm9,%%ymm"#c2";"\
  "vshuff64x2 $3,%%ymm10,%%ymm8,%%ymm"#c3";vshuff64x2 $3,%%ymm11,%%ymm9,%%ymm"#c4";"
#define GRAB_UPPER_8X2(c1,c2,c3,c4)\
  "vextractf64x4 $1, %%zmm"#c1", %%ymm"#c3";vextractf64x4 $1, %%zmm"#c2", %%ymm"#c4";"
#define GRAB_UPPER_8X4(c1,c2,c3,c4,c5,c6,c7,c8)\
  "vextractf64x4 $1, %%zmm"#c1", %%ymm"#c5";vextractf64x4 $1, %%zmm"#c2", %%ymm"#c6";"\
  "vextractf64x4 $1, %%zmm"#c3", %%ymm"#c7";vextractf64x4 $1, %%zmm"#c4", %%ymm"#c8";"
#define SAVE_m24n8 \
  save_init_m24 \
  init_m8n1_chks(0,1)\
  init_m8n1_chks(2,3)\
  init_m8n1_chks(4,5)\
  init_m8n1_chks(6,7)\
  save_m24n2_para(8,9,10,11,12,13,0,1) \
  save_m24n2_para(14,15,16,17,18,19,2,3) \
  save_m24n2_para(20,21,22,23,24,25,4,5) \
  save_m24n2_para(26,27,28,29,30,31,6,7)\
  add_col_m8n2(8,11,8) add_col_m8n2(14,17,14) add_col_m8n2(20,23,20) add_col_m8n2(26,29,26) add_col_m8n2(8,14,8) add_col_m8n2(20,26,20) add_col_m8n2(8,20,8)\
  add_col_m8n2(9,12,9) add_col_m8n2(15,18,15) add_col_m8n2(21,24,21) add_col_m8n2(27,30,27) add_col_m8n2(9,15,9) add_col_m8n2(21,27,21) add_col_m8n2(9,21,9)\
  add_col_m8n2(10,13,10) add_col_m8n2(16,19,16) add_col_m8n2(22,25,22) add_col_m8n2(28,31,28) add_col_m8n2(10,16,10) add_col_m8n2(22,28,22) add_col_m8n2(10,22,10)\
  "vaddpd (%6),%%zmm8,%%zmm8;vaddpd 64(%6),%%zmm9,%%zmm9;vaddpd 128(%6),%%zmm10,%%zmm10;"\
  "vmovups %%zmm8,(%6);vmovups %%zmm9,64(%6);vmovups %%zmm10,128(%6);addq $192,%6;"\
  trans_and_save_8X8
#define SAVE_m24n2 \
  save_init_m24 \
  init_m8n1_chks(0,1)\
  save_m24n2_para(8,9,10,11,12,13,0,1) \
  add_col_m8n2(8,11,8)\
  add_col_m8n2(9,12,9)\
  add_col_m8n2(10,13,10)\
  "vaddpd (%6),%%zmm8,%%zmm8;vaddpd 64(%6),%%zmm9,%%zmm9;vaddpd 128(%6),%%zmm10,%%zmm10;"\
  "vmovups %%zmm8,(%6);vmovups %%zmm9,64(%6);vmovups %%zmm10,128(%6);addq $192,%6;"\
  GRAB_UPPER_8X2(0,1,2,3)\
  TRANS_4X2(0,1,4,5) TRANS_4X2(2,3,6,7)\
  "vaddpd %%ymm4,%%ymm5,%%ymm4;vaddpd %%ymm6,%%ymm7,%%ymm6;"\
  "vextractf128 $1,%%ymm4,%%xmm2;vextractf128 $1,%%ymm6,%%xmm3;"\
  "vaddpd %%xmm4,%%xmm2,%%xmm4;vaddpd %%xmm6,%%xmm3,%%xmm6;"\
  "vaddpd %%xmm4,%%xmm6,%%xmm4;"\
  "vaddpd (%7),%%xmm4,%%xmm4;vmovups %%xmm4,(%7);"
#define SAVE_m8n8 \
  save_init_m8 \
  init_m8n1_chks(0,1)\
  init_m8n1_chks(2,3)\
  init_m8n1_chks(4,5)\
  init_m8n1_chks(6,7)\
  save_m8n2_para(8,11,0,1)\
  save_m8n2_para(14,17,2,3)\
  save_m8n2_para(20,23,4,5)\
  save_m8n2_para(26,29,6,7)\
  add_col_m8n2(8,11,8) add_col_m8n2(14,17,14) add_col_m8n2(20,23,20) add_col_m8n2(26,29,26) add_col_m8n2(8,14,8) add_col_m8n2(20,26,20) add_col_m8n2(8,20,8)\
  "vaddpd (%6),%%zmm8,%%zmm8;vmovups %%zmm8,(%6);addq $64,%6;"\
  trans_and_save_8X8
#define SAVE_m8n2 \
  save_init_m8 \
  init_m8n1_chks(0,1)\
  save_m8n2_para(8,11,0,1)\
  add_col_m8n2(8,11,8)\
  "vaddpd (%6),%%zmm8,%%zmm8;"\
  "vmovups %%zmm8,(%6);addq $64,%6;"\
  GRAB_UPPER_8X2(0,1,2,3)\
  TRANS_4X2(0,1,4,5) TRANS_4X2(2,3,6,7)\
  "vaddpd %%ymm4,%%ymm5,%%ymm4;vaddpd %%ymm6,%%ymm7,%%ymm6;"\
  "vextractf128 $1,%%ymm4,%%xmm2;vextractf128 $1,%%ymm6,%%xmm3;"\
  "vaddpd %%xmm4,%%xmm2,%%xmm4;vaddpd %%xmm6,%%xmm3,%%xmm6;"\
  "vaddpd %%xmm4,%%xmm6,%%xmm4;"\
  "vaddpd (%7),%%xmm4,%%xmm4;vmovups %%xmm4,(%7);"
#define SAVE_m2n8 \
  save_init_m2 \
  init_m2n1_chks(0,1)\
  init_m2n1_chks(2,3)\
  init_m2n1_chks(4,5)\
  init_m2n1_chks(6,7)\
  save_m2n2_para(8,9,0,1) \
  save_m2n2_para(10,11,2,3) \
  save_m2n2_para(12,13,4,5) \
  save_m2n2_para(14,15,6,7)\
  add_col_m2n2(8,9,8) add_col_m2n2(10,11,10) add_col_m2n2(12,13,12) add_col_m2n2(14,15,14) add_col_m2n2(8,10,8) add_col_m2n2(12,14,12) add_col_m2n2(8,12,8)\
  "vaddpd (%6),%%xmm8,%%xmm8;vmovups %%xmm8,(%6);addq $16,%6;"\
  TRANS_2X8(0,1,2,3,4,5,6,7) "vaddpd %%xmm8,%%xmm9,%%xmm8;vaddpd %%xmm10,%%xmm11,%%xmm10;vaddpd %%xmm12,%%xmm13,%%xmm12;vaddpd %%xmm14,%%xmm15,%%xmm14;"\
  "vaddpd (%7),%%xmm8,%%xmm8;vaddpd 16(%7),%%xmm10,%%xmm10;vaddpd 32(%7),%%xmm12,%%xmm12;vaddpd 48(%7),%%xmm14,%%xmm14;"\
  "vmovups %%xmm8,(%7);vmovups %%xmm10,16(%7);vmovups %%xmm12,32(%7);vmovups %%xmm14,48(%7);"
#define SAVE_m24n4 \
  save_init_m24 \
  init_m8n1_chks(0,1)\
  init_m8n1_chks(2,3)\
  save_m24n2_para(8,9,10,11,12,13,0,1) \
  save_m24n2_para(14,15,16,17,18,19,2,3) \
  add_col_m8n2(8,11,8) add_col_m8n2(14,17,14) add_col_m8n2(8,14,8)\
  add_col_m8n2(9,12,9) add_col_m8n2(15,18,15) add_col_m8n2(9,15,9)\
  add_col_m8n2(10,13,10) add_col_m8n2(16,19,16) add_col_m8n2(10,16,10)\
  "vaddpd (%6),%%zmm8,%%zmm8;vaddpd 64(%6),%%zmm9,%%zmm9;vaddpd 128(%6),%%zmm10,%%zmm10;"\
  "vmovups %%zmm8,(%6);vmovups %%zmm9,64(%6);vmovups %%zmm10,128(%6);addq $192,%6;"\
  GRAB_UPPER_8X4(0,1,2,3,4,5,6,7)\
  TRANS_4X4(0,1,2,3) TRANS_4X4(4,5,6,7)\
  "vaddpd %%ymm0,%%ymm1,%%ymm0;vaddpd %%ymm2,%%ymm3,%%ymm2;vaddpd %%ymm4,%%ymm5,%%ymm4;vaddpd %%ymm6,%%ymm7,%%ymm6;"\
  "vaddpd %%ymm0,%%ymm2,%%ymm0;vaddpd %%ymm4,%%ymm6,%%ymm4;vaddpd %%ymm0,%%ymm4,%%ymm0;"\
  "vaddpd (%7),%%ymm0,%%ymm0;vmovups %%ymm0,(%7);"
#define SAVE_m8n4 \
  save_init_m8 \
  init_m8n1_chks(0,1)\
  init_m8n1_chks(2,3)\
  save_m8n2_para(8,11,0,1)\
  save_m8n2_para(14,17,2,3)\
  add_col_m8n2(8,11,8) add_col_m8n2(14,17,14) add_col_m8n2(8,14,8)\
  "vaddpd (%6),%%zmm8,%%zmm8;"\
  "vmovups %%zmm8,(%6);addq $64,%6;"\
  GRAB_UPPER_8X4(0,1,2,3,4,5,6,7)\
  TRANS_4X4(0,1,2,3) TRANS_4X4(4,5,6,7)\
  "vaddpd %%ymm0,%%ymm1,%%ymm0;vaddpd %%ymm2,%%ymm3,%%ymm2;vaddpd %%ymm4,%%ymm5,%%ymm4;vaddpd %%ymm6,%%ymm7,%%ymm6;"\
  "vaddpd %%ymm0,%%ymm2,%%ymm0;vaddpd %%ymm4,%%ymm6,%%ymm4;vaddpd %%ymm0,%%ymm4,%%ymm0;"\
  "vaddpd (%7),%%ymm0,%%ymm0;vmovups %%ymm0,(%7);"
#define SAVE_m2n4 \
  save_init_m2 \
  init_m2n1_chks(0,1)\
  init_m2n1_chks(2,3)\
  save_m2n2_para(8,9,0,1) \
  save_m2n2_para(10,11,2,3)\
  add_col_m2n2(8,9,8) add_col_m2n2(10,11,10) add_col_m2n2(8,10,8)\
  "vaddpd (%6),%%xmm8,%%xmm8;vmovups %%xmm8,(%6);addq $16,%6;"\
  TRANS_2X4(0,1,2,3) "vaddpd %%xmm8,%%xmm9,%%xmm8;vaddpd %%xmm10,%%xmm11,%%xmm10;"\
  "vaddpd (%7),%%xmm8,%%xmm8;vaddpd 16(%7),%%xmm10,%%xmm10;"\
  "vmovups %%xmm8,(%7);vmovups %%xmm10,16(%7);"
#define SAVE_m2n2 \
  save_init_m2 \
  init_m2n1_chks(0,1)\
  save_m2n2_para(8,9,0,1)\
  add_col_m2n2(8,9,8)\
  "vaddpd (%6),%%xmm8,%%xmm8;vmovups %%xmm8,(%6);addq $16,%6;"\
  TRANS_2X2(0,1,2,3) "vaddpd %%xmm2,%%xmm3,%%xmm2;vaddpd (%7),%%xmm2,%%xmm2;vmovups %%xmm2,(%7);"
#define COMPUTE_m24n8 \
    "movq %%r14,%1;leaq (%1,%%r11,2),%%r12;addq %%r11,%%r12;movq %8,%%r13;"\
    init_m24n8 \
    "cmpq $4,%%r13;jb 724782f;\n\t"\
    "724781:\n\t"\
    KERNEL_m24n8 "subq $4,%%r13;testq $12,%%r13;movq $172,%%r10;cmovz %4,%%r10;"\
    KERNEL_m24n8 "prefetcht1 (%3);subq $129,%3;addq %%r10,%3;"\
    KERNEL_m24n8 "prefetcht1 (%5);addq $32,%5;cmpq $192,%%r13;cmoveq %2,%3;"\
    KERNEL_m24n8 \
    "cmpq $16,%%r13;jnb 724781b;\n\t"\
    "movq %2,%3;"\
    "cmpq $0,%%r13;je 724783f;\n\t"\
    "724782:\n\t"\
    "prefetcht0 (%3); prefetcht0 64(%3); prefetcht0 128(%3);decq %%r13;"\
    KERNEL_m24n8 \
    "addq %4,%3;testq %%r13,%%r13;jnz 724782b;\n\t"\
    "724783:\n\t"\
    SAVE_m24n8 

#define COMPUTE_m24n4 \
    "movq %%r14,%1;leaq (%1,%%r11,2),%%r12;addq %%r11,%%r12;movq %8,%%r13;"\
    init_m24n4 \
    "cmpq $4,%%r13;jb 724742f;\n\t"\
    "724741:\n\t"\
    KERNEL_m24n4 "subq $4,%%r13;testq $12,%%r13;movq $172,%%r10;cmovz %4,%%r10;"\
    KERNEL_m24n4 "prefetcht1 (%3);subq $129,%3;addq %%r10,%3;"\
    KERNEL_m24n4 "prefetcht1 (%5);addq $32,%5;cmpq $192,%%r13;cmoveq %2,%3;"\
    KERNEL_m24n4 \
    "cmpq $16,%%r13;jnb 724741b;\n\t"\
    "movq %2,%3;"\
    "cmpq $0,%%r13;je 724743f;\n\t"\
    "724742:\n\t"\
    "prefetcht0 (%3); prefetcht0 64(%3); prefetcht0 128(%3);decq %%r13;"\
    KERNEL_m24n4 \
    "addq %4,%3;testq %%r13,%%r13;jnz 724742b;\n\t"\
    "724743:\n\t"\
    SAVE_m24n4 

#define COMPUTE_m8n8 \
    "movq %%r14,%1;leaq (%1,%%r11,2),%%r12;addq %%r11,%%r12;movq %8,%%r13;"\
    init_m8n8 \
    "cmpq $4,%%r13;jb 78782f;\n\t"\
    "78781:\n\t"\
    KERNEL_m8n8 \
    KERNEL_m8n8 \
    KERNEL_m8n8 "subq $4,%%r13;"\
    KERNEL_m8n8 \
    "cmpq $4,%%r13;jnb 78781b;\n\t"\
    "movq %2,%3;"\
    "cmpq $0,%%r13;je 78783f;\n\t"\
    "78782:\n\t"\
    "prefetcht0 (%3); prefetcht0 64(%3); prefetcht0 128(%3);decq %%r13;"\
    KERNEL_m8n8 \
    "addq %4,%3;testq %%r13,%%r13;jnz 78782b;\n\t"\
    "78783:\n\t"\
    SAVE_m8n8

#define COMPUTE_m24n2 \
    "movq %%r14,%1;leaq (%1,%%r11,2),%%r12;addq %%r11,%%r12;movq %8,%%r13;"\
    init_m24n2 \
    "cmpq $4,%%r13;jb 724722f;\n\t"\
    "724721:\n\t"\
    KERNEL_m24n2 \
    KERNEL_m24n2 \
    KERNEL_m24n2 "subq $4,%%r13;"\
    KERNEL_m24n2 \
    "cmpq $4,%%r13;jnb 724721b;\n\t"\
    "movq %2,%3;"\
    "cmpq $0,%%r13;je 724723f;\n\t"\
    "724722:\n\t"\
    "prefetcht0 (%3); prefetcht0 64(%3); prefetcht0 128(%3);decq %%r13;"\
    KERNEL_m24n2 \
    "addq %4,%3;testq %%r13,%%r13;jnz 724722b;\n\t"\
    "724723:\n\t"\
    SAVE_m24n2

#define COMPUTE_m8n2 \
    "movq %%r14,%1;leaq (%1,%%r11,2),%%r12;addq %%r11,%%r12;movq %8,%%r13;"\
    init_m8n2 \
    "cmpq $4,%%r13;jb 78722f;\n\t"\
    "78721:\n\t"\
    KERNEL_m8n2 \
    KERNEL_m8n2 \
    KERNEL_m8n2 "subq $4,%%r13;"\
    KERNEL_m8n2 \
    "cmpq $4,%%r13;jnb 78721b;\n\t"\
    "movq %2,%3;"\
    "cmpq $0,%%r13;je 78723f;\n\t"\
    "78722:\n\t"\
    "prefetcht0 (%3); prefetcht0 64(%3); prefetcht0 128(%3);decq %%r13;"\
    KERNEL_m8n2 \
    "addq %4,%3;testq %%r13,%%r13;jnz 78722b;\n\t"\
    "78723:\n\t"\
    SAVE_m8n2

#define COMPUTE_m2n8 \
    "movq %%r14,%1;leaq (%1,%%r11,2),%%r12;addq %%r11,%%r12;movq %8,%%r13;"\
    init_m2n8 \
    "cmpq $4,%%r13;jb 72782f;\n\t"\
    "72781:\n\t"\
    KERNEL_m2n8 \
    KERNEL_m2n8 \
    KERNEL_m2n8 "subq $4,%%r13;"\
    KERNEL_m2n8 \
    "cmpq $4,%%r13;jnb 72781b;\n\t"\
    "movq %2,%3;"\
    "cmpq $0,%%r13;je 72783f;\n\t"\
    "72782:\n\t"\
    "prefetcht0 (%3); prefetcht0 64(%3); prefetcht0 128(%3);decq %%r13;"\
    KERNEL_m2n8 \
    "addq %4,%3;testq %%r13,%%r13;jnz 72782b;\n\t"\
    "72783:\n\t"\
    SAVE_m2n8

#define COMPUTE_m8n4 \
    "movq %%r14,%1;leaq (%1,%%r11,2),%%r12;addq %%r11,%%r12;movq %8,%%r13;"\
    init_m8n4 \
    "cmpq $4,%%r13;jb 78742f;\n\t"\
    "78741:\n\t"\
    KERNEL_m8n4 \
    KERNEL_m8n4 \
    KERNEL_m8n4 "subq $4,%%r13;"\
    KERNEL_m8n4 \
    "cmpq $4,%%r13;jnb 78741b;\n\t"\
    "movq %2,%3;"\
    "cmpq $0,%%r13;je 78743f;\n\t"\
    "78742:\n\t"\
    "prefetcht0 (%3); prefetcht0 64(%3); prefetcht0 128(%3);decq %%r13;"\
    KERNEL_m8n4 \
    "addq %4,%3;testq %%r13,%%r13;jnz 78742b;\n\t"\
    "78743:\n\t"\
    SAVE_m8n4 

#define COMPUTE_m2n2 \
    "movq %%r14,%1;leaq (%1,%%r11,2),%%r12;addq %%r11,%%r12;movq %8,%%r13;"\
    init_m2n2 \
    "cmpq $4,%%r13;jb 72722f;\n\t"\
    "72721:\n\t"\
    KERNEL_m2n2 \
    KERNEL_m2n2 \
    KERNEL_m2n2 "subq $4,%%r13;"\
    KERNEL_m2n2 \
    "cmpq $4,%%r13;jnb 72721b;\n\t"\
    "movq %2,%3;"\
    "cmpq $0,%%r13;je 72723f;\n\t"\
    "72722:\n\t"\
    "prefetcht0 (%3); prefetcht0 64(%3); prefetcht0 128(%3);decq %%r13;"\
    KERNEL_m2n2 \
    "addq %4,%3;testq %%r13,%%r13;jnz 72722b;\n\t"\
    "72723:\n\t"\
    SAVE_m2n2 

#define COMPUTE_m2n4 \
    "movq %%r14,%1;leaq (%1,%%r11,2),%%r12;addq %%r11,%%r12;movq %8,%%r13;"\
    init_m2n4 \
    "cmpq $4,%%r13;jb 72742f;\n\t"\
    "72741:\n\t"\
    KERNEL_m2n4 \
    KERNEL_m2n4 \
    KERNEL_m2n4 "subq $4,%%r13;"\
    KERNEL_m2n4 \
    "cmpq $4,%%r13;jnb 72741b;\n\t"\
    "movq %2,%3;"\
    "cmpq $0,%%r13;je 72743f;\n\t"\
    "72742:\n\t"\
    "prefetcht0 (%3); prefetcht0 64(%3); prefetcht0 128(%3);decq %%r13;"\
    KERNEL_m2n4 \
    "addq %4,%3;testq %%r13,%%r13;jnz 72742b;\n\t"\
    "72743:\n\t"\
    SAVE_m2n4 

#define macro_n8 {\
  b_pref = b_ptr + 8 * K;\
  __asm__ __volatile__(\
    "movq %9,%%r15; movq %1,%%r14; movq %8,%%r11; salq $4,%%r11;"\
    "cmpq $24,%%r15; jb 3243831f;"\
    "3243830:\n\t"\
    COMPUTE_m24n8 "subq $24,%%r15; cmpq $24,%%r15; jnb 3243830b;"\
    "3243831:\n\t"\
    "cmpq $8,%%r15; jb 3243833f;"\
    "3243832:\n\t"\
    COMPUTE_m8n8 "subq $8,%%r15; cmpq $8,%%r15; jnb 3243832b;"\
    "3243833:\n\t"\
    "cmpq $2,%%r15; jb 3243835f;"\
    "3243834:\n\t"\
    COMPUTE_m2n8 "subq $2,%%r15; cmpq $2,%%r15; jnb 3243834b;"\
    "3243835:\n\t"\
    "movq %%r14,%1;"\
  :"+r"(a_ptr),"+r"(b_ptr),"+r"(c_ptr),"+r"(c_tmp),"+r"(ldc_in_bytes),"+r"(b_pref),"+r"(chk_c_ptr_pos),"+r"(chk_c_row_ptr_pos):"m"(K),"m"(M):"r10","r11","r12","r13","r14","r15","cc","memory",\
    "zmm0","zmm1","zmm2","zmm3","zmm4","zmm5","zmm6","zmm7","zmm8","zmm9","zmm10","zmm11","zmm12","zmm13","zmm14","zmm15",\
    "zmm16","zmm17","zmm18","zmm19","zmm20","zmm21","zmm22","zmm23","zmm24","zmm25","zmm26","zmm27","zmm28","zmm29","zmm30","zmm31");\
  a_ptr -= M * K; b_ptr += 8 * K; c_ptr += 8 * ldc - M;\
}

#define macro_n4 {\
  b_pref = b_ptr + 4 * K;\
  __asm__ __volatile__(\
    "movq %9,%%r15; movq %1,%%r14; movq %8,%%r11; salq $4,%%r11;"\
    "cmpq $24,%%r15; jb 3243431f;"\
    "3243430:\n\t"\
    COMPUTE_m24n4 "subq $24,%%r15; cmpq $24,%%r15; jnb 3243430b;"\
    "3243431:\n\t"\
    "cmpq $8,%%r15; jb 3243433f;"\
    "3243432:\n\t"\
    COMPUTE_m8n4 "subq $8,%%r15; cmpq $8,%%r15; jnb 3243432b;"\
    "3243433:\n\t"\
    "cmpq $2,%%r15; jb 3243435f;"\
    "3243434:\n\t"\
    COMPUTE_m2n4 "subq $2,%%r15; cmpq $2,%%r15; jnb 3243434b;"\
    "3243435:\n\t"\
    "movq %%r14,%1;"\
  :"+r"(a_ptr),"+r"(b_ptr),"+r"(c_ptr),"+r"(c_tmp),"+r"(ldc_in_bytes),"+r"(b_pref),"+r"(chk_c_ptr_pos),"+r"(chk_c_row_ptr_pos):"m"(K),"m"(M):"r10","r11","r12","r13","r14","r15","cc","memory",\
    "zmm0","zmm1","zmm2","zmm3","zmm4","zmm5","zmm6","zmm7","zmm8","zmm9","zmm10","zmm11","zmm12","zmm13","zmm14","zmm15",\
    "zmm16","zmm17","zmm18","zmm19","zmm20","zmm21","zmm22","zmm23","zmm24","zmm25","zmm26","zmm27","zmm28","zmm29","zmm30","zmm31");\
  a_ptr -= M * K; b_ptr += 4 * K; c_ptr += 4 * ldc - M;\
}

#define macro_n2 {\
  b_pref = b_ptr + 2 * K;\
  __asm__ __volatile__(\
    "movq %9,%%r15; movq %1,%%r14; movq %8,%%r11; salq $4,%%r11;"\
    "cmpq $24,%%r15; jb 3243231f;"\
    "3243230:\n\t"\
    COMPUTE_m24n2 "subq $24,%%r15; cmpq $24,%%r15; jnb 3243230b;"\
    "3243231:\n\t"\
    "cmpq $8,%%r15; jb 3243233f;"\
    "3243232:\n\t"\
    COMPUTE_m8n2 "subq $8,%%r15; cmpq $8,%%r15; jnb 3243232b;"\
    "3243233:\n\t"\
    "cmpq $2,%%r15; jb 3243235f;"\
    "3243234:\n\t"\
    COMPUTE_m2n2 "subq $2,%%r15; cmpq $2,%%r15; jnb 3243234b;"\
    "3243235:\n\t"\
    "movq %%r14,%1;"\
  :"+r"(a_ptr),"+r"(b_ptr),"+r"(c_ptr),"+r"(c_tmp),"+r"(ldc_in_bytes),"+r"(b_pref),"+r"(chk_c_ptr_pos),"+r"(chk_c_row_ptr_pos):"m"(K),"m"(M):"r10","r11","r12","r13","r14","r15","cc","memory",\
    "zmm0","zmm1","zmm2","zmm3","zmm4","zmm5","zmm6","zmm7","zmm8","zmm9","zmm10","zmm11","zmm12","zmm13","zmm14","zmm15",\
    "zmm16","zmm17","zmm18","zmm19","zmm20","zmm21","zmm22","zmm23","zmm24","zmm25","zmm26","zmm27","zmm28","zmm29","zmm30","zmm31");\
  a_ptr -= M * K; b_ptr += 2 * K; c_ptr += 2 * ldc - M;\
}


void macro_kernel(double *a_buffer,double *b_buffer,int m,int n,int k,double *C, int LDC,int k_inc,double * chk_c_col_ptr,double * chk_c_row_ptr){
    int m_count,n_count,m_count_sub,n_count_sub;
    if (m==0||n==0||k==0) return;
    int64_t M=(int64_t)m,K=(int64_t)k,ldc_in_bytes=(int64_t)LDC*sizeof(double),ldc=(int32_t)LDC;
    double *a_ptr=a_buffer,*b_ptr=b_buffer,*c_ptr=C,*b_pref=b_ptr,*c_tmp=C;
    double *chk_c_ptr_pos=chk_c_col_ptr,*chk_c_row_ptr_pos=chk_c_row_ptr;
    // printf("m= %d, n=%d, k = %d\n",m,n,k);
    for (n_count_sub=n,n_count=0;n_count_sub>7;n_count_sub-=8,n_count+=8){
        //call the m layer with n=8;
        chk_c_ptr_pos = chk_c_col_ptr;
        macro_n8
        chk_c_row_ptr_pos+=8;
        //TODO: case when m is divisible by 1
    }
    for (;n_count_sub>3;n_count_sub-=4,n_count+=4){
        //call the m layer with n=4
        chk_c_ptr_pos = chk_c_col_ptr;
        macro_n4
        chk_c_row_ptr_pos+=4;
        //TODO: case when m is divisible by 1
    }
    for (;n_count_sub>1;n_count_sub-=2,n_count+=2){
        //call the m layer with n=2
        chk_c_ptr_pos = chk_c_col_ptr;
        macro_n2
        chk_c_row_ptr_pos+=2;
    }
    for (;n_count_sub>0;n_count_sub-=1,n_count+=1){
        //TODO:call the m layer with n=1
    }
}

void checksum_encoding_A_col_major(double *A, int m, int n, int lda, double *chksm_A_col){
  int i,j;
  int m8=m&-8,n4=n&-4;
  double *ptr_a1, *ptr_a2,*ptr_a3,*ptr_a4;
  double tmp1,tmp2,tmp3,tmp4;
  for (j=0;j<n4;j+=4){
    ptr_a1 = A + lda * j;ptr_a2=ptr_a1+lda;ptr_a3=ptr_a2+lda;ptr_a4=ptr_a3+lda;
    __m512d v1 = _mm512_setzero_pd();__m512d v2 = _mm512_setzero_pd();
    __m512d v3 = _mm512_setzero_pd();__m512d v4 = _mm512_setzero_pd();
    for (i=0;i<m8;i+=8){
      v1 = _mm512_add_pd(_mm512_loadu_pd(ptr_a1+i),v1);v2 = _mm512_add_pd(_mm512_loadu_pd(ptr_a2+i),v2);
      v3 = _mm512_add_pd(_mm512_loadu_pd(ptr_a3+i),v3);v4 = _mm512_add_pd(_mm512_loadu_pd(ptr_a4+i),v4);
    }
    tmp1 = _mm512_reduce_add_pd(v1);tmp2 = _mm512_reduce_add_pd(v2);
    tmp3 = _mm512_reduce_add_pd(v3);tmp4 = _mm512_reduce_add_pd(v4);
    for (i=m8;i<m;i++){
      tmp1+=*(ptr_a1+i);tmp2+=*(ptr_a2+i);
      tmp3+=*(ptr_a3+i);tmp4+=*(ptr_a4+i);
    }
    chksm_A_col[j]=tmp1;chksm_A_col[j+1]=tmp2;chksm_A_col[j+2]=tmp3;chksm_A_col[j+3]=tmp4;
  }
  if (n4==n) return;
  for (j=n4;j<n;j++){
    double *ptr_a = A + j*lda;
    double tmp = 0;
    for (i=0;i<m;i++) tmp += ptr_a[i];
    chksm_A_col[j]=tmp;
  }
}

int verify_chks(double *online_chks,double *c_chks, int n){
  int cnt=0,res=0;
  for (int i=0;i<n;i++){
    if (fabs(online_chks[i]-c_chks[i])>THRESHOLD) {
      res=i;
      cnt++;
    }
    c_chks[i]=0.;
  }
  if (cnt>1) return -1;
  return res;
}

void ftblas_dgemm_ft(\
    int M, \
    int N, \
    int K, \
    double alpha, \
    double *A, \
    int LDA, \
    double *B, \
    int LDB, \
    double beta, \
    double *C, \
    int LDC)\
{
    int i,j,k;
#ifdef TIMER
    double t_tot=0.,t_copy_a=0.,t_copy_b=0.,t_scale_c=0., t0, t1;
    t0=get_sec();
#endif

#ifdef TIMER
    t1=get_sec();
    t_scale_c+=(t1-t0);
#endif
    if (alpha == 0.||K==0) return;
    int M4,N8=N&-8,K4;
    double *a_buffer = (double *)aligned_alloc(4096,K_BLOCKING*OUT_M_BLOCKING*sizeof(double));
    double *b_buffer = (double *)aligned_alloc(4096,K_BLOCKING*OUT_N_BLOCKING*sizeof(double));
    double *chk_A = (double *)calloc(K,sizeof(double));
    checksum_encoding_A_col_major(A,M,K,LDA,chk_A);
    double *chk_B = (double *)calloc(K,sizeof(double));
    double *chk_c_col = (double *)aligned_alloc(4096,M*sizeof(double));
    double *chk_c_row = (double *)aligned_alloc(4096,N*sizeof(double));
    double *chk_online_C_row = (double *)calloc(N,sizeof(double));
    double *chk_online_C_col = (double *)calloc(M,sizeof(double));
    scale_c(C,M,N,LDC,beta,chk_online_C_col,chk_online_C_row);
    
    int second_m_count,second_n_count,second_m_inc,second_n_inc;
    int m_count,n_count,k_count;
    int m_inc,n_inc,k_inc;
    for (k_count=0;k_count<K;k_count+=k_inc){
        k_inc=(K-k_count>K_BLOCKING)?K_BLOCKING:K-k_count;
        for (n_count=0;n_count<N;n_count+=n_inc){
            n_inc=(N-n_count>OUT_N_BLOCKING)?OUT_N_BLOCKING:N-n_count;
#ifdef TIMER
            t0=get_sec();
#endif
            packing_b(B+k_count+n_count*LDB,b_buffer,LDB,k_inc,n_inc,chk_A+k_count,chk_B+k_count,chk_online_C_row,alpha);
#ifdef TIMER
            t1=get_sec();
            t_copy_b+=(t1-t0);
#endif
            for (m_count=0;m_count<M;m_count+=m_inc){
                m_inc=(M-m_count>OUT_M_BLOCKING)?OUT_M_BLOCKING:M-m_count;
#ifdef TIMER
                t0=get_sec();
#endif
                packing_a(alpha,A+m_count+k_count*LDA,a_buffer,LDA,m_inc,k_inc,chk_B+k_count,chk_online_C_col+m_count);
#ifdef TIMER
                t1=get_sec();
                t_copy_a+=(t1-t0);
#endif
                for (second_m_count=m_count;second_m_count<m_count+m_inc;second_m_count+=second_m_inc){
                    second_m_inc=(m_count+m_inc-second_m_count>M_BLOCKING)?M_BLOCKING:m_count+m_inc-second_m_count;
                    for (second_n_count=n_count;second_n_count<n_count+n_inc;second_n_count+=second_n_inc){
                        second_n_inc=(n_count+n_inc-second_n_count>N_BLOCKING)?N_BLOCKING:n_count+n_inc-second_n_count;
#ifdef TIMER
                        t0=get_sec();
#endif
                        macro_kernel(a_buffer+(second_m_count-m_count)*k_inc,b_buffer+(second_n_count-n_count)*k_inc,second_m_inc,second_n_inc,k_inc,&C(second_m_count,second_n_count),LDC,k_inc,chk_c_col+second_m_count,chk_c_row+second_n_count);
#ifdef TIMER
                        t1=get_sec();
                        t_tot+=(t1-t0);
#endif
                    }
                }
            }
        }
        // print_vector(chk_online_C_row,N);
        // print_vector(chk_c_row,N);
        // int v_chks_col=verify_chks(chk_online_C_col,chk_c_col,M);
        // int v_chks_row=verify_chks(chk_online_C_row,chk_c_row,N);

        // printf("correcting (%d,%d):\n",v_chks_row,v_chks_col);
        // if (v_chks_col==-1||v_chks_row==-1) {
        //   printf("too many errors, unable to recover!\n");
        //   return;
        // }else if (v_chks_row==1&&v_chks_col){
        //   printf("correcting (%d,%d):\n",v_chks_row,v_chks_col);
        // }
    }
#ifdef TIMER
    printf("Time in major loop: %f, perf=%f GFLOPS\n", t_tot,2.*1e-9*M*N*K/t_tot);
    printf("Time in copy A: %f, perf = %f GFLOPS\n", t_copy_a, 2.*1e-9*M*K/t_copy_a);
    printf("Time in copy B: %f, perf = %f GFLOPS\n", t_copy_b, 2.*1e-9*N*K/t_copy_b);
    printf("Time in scaling C: %f, perf = %f GFLOPS\n", t_scale_c, 2.*1e-9*M*N/t_scale_c);
    printf("Total: %f, perf = %f GFLOPS\n", t_tot+t_copy_b+t_scale_c+t_copy_a, 2.*1e-9*M*N*K/(t_tot+t_copy_b+t_scale_c+t_copy_a));
#endif
    // print_vector(chk_online_C_row,N);
    // print_vector(chk_c_row,N);
    // verify_matrix(chk_online_C_col,chk_c_col,M);
    // verify_matrix(chk_online_C_row,chk_c_row,N);
    free(a_buffer);free(b_buffer);
    free(chk_A);free(chk_B);free(chk_c_col);free(chk_c_row);
    free(chk_online_C_col);free(chk_online_C_row);
}