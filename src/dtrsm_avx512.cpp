//#define TIMER 1
#include "immintrin.h"
#include <stdint.h>
#include "../include/ftblas.h"
#define A(i,j) A[(i)+(j)*LDA]
#define B(i,j) B[(i)+(j)*LDB]
#define C(i,j) C[(i)+(j)*LDC]
#define M_BLOCKING 192
#define N_BLOCKING 9216
#define K_BLOCKING 384




void scale_c(double *C,int M, int N, int LDC, double scalar){
    int m_count,n_count;
    int M8=M&-8,N4=N&-4,LDC2=LDC<<1,LDC3=LDC2+LDC,LDC4=LDC<<2;
    __m512d vscalar = _mm512_set1_pd(scalar);
    double *c_ptr_base1 = C,*c_ptr_base2 = C+LDC,*c_ptr_base3 = C+LDC2,*c_ptr_base4 = C+LDC3;
    double *c_ptr_dyn1,*c_ptr_dyn2,*c_ptr_dyn3,*c_ptr_dyn4;
    for (n_count=0;n_count<N4;n_count+=4){
        c_ptr_dyn1 = c_ptr_base1;c_ptr_dyn2 = c_ptr_base2;c_ptr_dyn3 = c_ptr_base3;c_ptr_dyn4 = c_ptr_base4;
        for (m_count=0;m_count<M8;m_count+=8){
            _mm512_storeu_pd(c_ptr_dyn1,_mm512_mul_pd(_mm512_loadu_pd(c_ptr_dyn1),vscalar));
            _mm512_storeu_pd(c_ptr_dyn2,_mm512_mul_pd(_mm512_loadu_pd(c_ptr_dyn2),vscalar));
            _mm512_storeu_pd(c_ptr_dyn3,_mm512_mul_pd(_mm512_loadu_pd(c_ptr_dyn3),vscalar));
            _mm512_storeu_pd(c_ptr_dyn4,_mm512_mul_pd(_mm512_loadu_pd(c_ptr_dyn4),vscalar));
            c_ptr_dyn1+=8;c_ptr_dyn2+=8;c_ptr_dyn3+=8;c_ptr_dyn4+=8;
        }
        for (;m_count<M;m_count++){
            *c_ptr_dyn1 *= scalar; c_ptr_dyn1++;
            *c_ptr_dyn2 *= scalar; c_ptr_dyn2++;
            *c_ptr_dyn3 *= scalar; c_ptr_dyn3++;
            *c_ptr_dyn4 *= scalar; c_ptr_dyn4++;
        }
        c_ptr_base1 += LDC4;c_ptr_base2 += LDC4;c_ptr_base3 += LDC4;c_ptr_base4 += LDC4;
    }
    for (;n_count<N;n_count++){
        c_ptr_dyn1 = c_ptr_base1;
        for (m_count=0;m_count<M8;m_count+=8){
            _mm512_storeu_pd(c_ptr_dyn1,_mm512_mul_pd(_mm512_loadu_pd(c_ptr_dyn1),vscalar));
            c_ptr_dyn1+=8;
        }
        for (;m_count<M;m_count++){
            *c_ptr_dyn1 *= scalar; c_ptr_dyn1++;
        }
        c_ptr_base1 += LDC4;
    }
}

void packing_b_24x8(double *src,double *dst,int leading_dim,int dim_first,int dim_second){
    //dim_first:K,dim_second:N
    double *tosrc1,*tosrc2,*todst;
    todst=dst;
    int count_first,count_second,count_sub=dim_second;
    for (count_second=0;count_sub>1;count_second+=2,count_sub-=2){
        tosrc1=src+count_second*leading_dim;tosrc2=tosrc1+leading_dim;
        for (count_first=0;count_first<dim_first;count_first++){
            *todst=*tosrc1;tosrc1++;todst++;
            *todst=*tosrc2;tosrc2++;todst++;
        }
    }
    for (;count_sub>0;count_second++,count_sub-=1){
        tosrc1=src+count_second*leading_dim;
        for (count_first=0;count_first<dim_first;count_first++){
            *todst=*tosrc1;tosrc1++;todst++;
        }
    }
}

void packing_a_lower(double *src, double *dst, int leading_dim, int dim_first, int dim_second){
    if(dim_first==0 || dim_second==0) return;
    int count_first,count_second;
    double *tosrc,*todst,*tosrc1,*tosrc2,*tosrc3,*tosrc4;
    int c24,c8,c2,c1;
    int remain = dim_first;
    c24=remain/24;remain -= 24*c24;
    c8 = remain/8;remain -= 8*c8;
    c2 = remain/2;
    double *ptr_24 = dst + 24*dim_second*c24;
    double *ptr_8 = ptr_24+8*dim_second*c8;
    double *ptr_2 = ptr_8+2*dim_second*c2;
    __m512d z1,z2,z3,z4,z5,z6,z7,z8,z9,z10,z11,z12;
    // #pragma omp parallel for
    for(count_second=0;count_second<dim_second;count_second+=4){
      tosrc = src + count_second * leading_dim;
      tosrc1=tosrc;tosrc2=tosrc1+leading_dim;tosrc3=tosrc2+leading_dim;tosrc4=tosrc3+leading_dim;
      todst = dst + count_second * 24;
      for(count_first=dim_first;count_first>23;count_first-=24){
        z1=_mm512_loadu_pd(tosrc1);
        z2=_mm512_loadu_pd(tosrc1+8);
        z3=_mm512_loadu_pd(tosrc1+16);tosrc1+=24;
        z4=_mm512_loadu_pd(tosrc2);
        z5=_mm512_loadu_pd(tosrc2+8);
        z6=_mm512_loadu_pd(tosrc2+16);tosrc2+=24;
        z7=_mm512_loadu_pd(tosrc3);
        z8=_mm512_loadu_pd(tosrc3+8);
        z9=_mm512_loadu_pd(tosrc3+16);tosrc3+=24;
        z10=_mm512_loadu_pd(tosrc4);
        z11=_mm512_loadu_pd(tosrc4+8);
        z12=_mm512_loadu_pd(tosrc4+16);tosrc4+=24;
        _mm512_store_pd(todst,z1);
        _mm512_store_pd(todst+8,z2);
        _mm512_store_pd(todst+16,z3);
        _mm512_store_pd(todst+24,z4);
        _mm512_store_pd(todst+32,z5);
        _mm512_store_pd(todst+40,z6);
        _mm512_store_pd(todst+48,z7);
        _mm512_store_pd(todst+56,z8);
        _mm512_store_pd(todst+64,z9);
        _mm512_store_pd(todst+72,z10);
        _mm512_store_pd(todst+80,z11);
        _mm512_store_pd(todst+88,z12);
        todst+=24*dim_second;
      }
      todst = ptr_24 + 8*count_second;
      for(;count_first>7;count_first-=8){
        z1=_mm512_loadu_pd(tosrc1);tosrc1+=8;
        z4=_mm512_loadu_pd(tosrc2);tosrc2+=8;
        z7=_mm512_loadu_pd(tosrc3);tosrc3+=8;
        z10=_mm512_loadu_pd(tosrc4);tosrc4+=8;
        _mm512_store_pd(todst,z1);
        _mm512_store_pd(todst+8,z4);
        _mm512_store_pd(todst+16,z7);
        _mm512_store_pd(todst+24,z10);
        todst+=8*dim_second;
      }
      todst = ptr_8 + 2*count_second;
      for(;count_first>1;count_first-=2){
        todst[0]=tosrc[0];todst[1]=tosrc[1];
        tosrc+=2;todst+=2*dim_second;
      }
      todst = ptr_2 + count_second;
      if(count_first>0) *todst=*tosrc;
    }
    for(;count_second<dim_second;count_second++){
      tosrc = src + count_second * leading_dim;
      todst = dst + count_second * 24;
      for(count_first=dim_first;count_first>23;count_first-=24){
        _mm512_store_pd(todst,_mm512_loadu_pd(tosrc));
        _mm512_store_pd(todst+8,_mm512_loadu_pd(tosrc+8));
        _mm512_store_pd(todst+16,_mm512_loadu_pd(tosrc+16));
        tosrc+=24;todst+=24*dim_second;
      }
      todst = ptr_24 + 8*count_second;
      for(;count_first>7;count_first-=8){
        _mm512_store_pd(todst,_mm512_loadu_pd(tosrc));
        tosrc+=8;todst+=8*dim_second;
      }
      todst = ptr_8 + 2*count_second;
      for(;count_first>1;count_first-=2){
        todst[0]=tosrc[0];todst[1]=tosrc[1];
        tosrc+=2;todst+=2*dim_second;
      }
      todst = ptr_2 + count_second;
      if(count_first>0) *todst=*tosrc;
    }
}

void packing_a_24x8_trsm(double *src, double *dst, int leading_dim, int dim_first, int dim_second,int offset){
    if (dim_first==0||dim_second==0) return;
    if (offset<0) offset=0;
    if (offset>dim_second) offset=dim_second;
    double *tosrc,*todst;
    todst=dst;
    int count_first,count_second,count_sub=dim_first,count_diag,start_pos;
    int next_offset;
    double *tmp_dst,*tmp_src;
    for (count_first=0;count_sub>23;count_first+=24,count_sub-=24){
        tosrc=src+count_first;
        for(count_second=0;count_second<offset;count_second++){
            _mm512_store_pd(todst,_mm512_loadu_pd(tosrc));
            _mm512_store_pd(todst+8,_mm512_loadu_pd(tosrc+8));
            _mm512_store_pd(todst+16,_mm512_loadu_pd(tosrc+16));
            tosrc+=leading_dim;
            todst+=24;
        }
        start_pos=0;tmp_dst=todst;tmp_src=tosrc;
        next_offset=(offset+24<=dim_second)?offset+24:dim_second;
        for(count_second=offset;count_second<next_offset;count_second++){
          todst+=start_pos;tosrc+=start_pos;
          *todst=1./ *tosrc;tosrc++;todst++;
          for(count_diag=start_pos+1;count_diag<24;count_diag++){
            *todst=*tosrc;
            tosrc++;todst++;
          }
          tmp_dst+=24;todst=tmp_dst;
          tmp_src+=leading_dim;tosrc=tmp_src;
          start_pos++;
        }
        todst+=(dim_second-next_offset)*24;
        offset=next_offset;        
    }    
    // edge case
    for (;count_sub>7;count_first+=8,count_sub-=8){
        tosrc=src+count_first;
        for(count_second=0;count_second<offset;count_second++){
            _mm512_store_pd(todst,_mm512_loadu_pd(tosrc));
            tosrc+=leading_dim;
            todst+=8;
        }
        start_pos=0;tmp_dst=todst;tmp_src=tosrc;
        next_offset=(offset+8<=dim_second)?offset+8:dim_second;
        for(count_second=offset;count_second<next_offset;count_second++){
          todst+=start_pos;tosrc+=start_pos;
          *todst=1./ *tosrc;tosrc++;todst++;
          for(count_diag=start_pos+1;count_diag<8;count_diag++){
            *todst=*tosrc;
            tosrc++;todst++;
          }
          tmp_dst+=8;todst=tmp_dst;
          tmp_src+=leading_dim;tosrc=tmp_src;
          start_pos++;
        }
        todst+=(dim_second-next_offset)*8;
        offset=next_offset;     
    }
    for (;count_sub>1;count_first+=2,count_sub-=2){
        tosrc=src+count_first;
        for(count_second=0;count_second<offset;count_second++){
            _mm_store_pd(todst,_mm_loadu_pd(tosrc));
            tosrc+=leading_dim;
            todst+=2;
        }
        start_pos=0;tmp_dst=todst;tmp_src=tosrc;
        next_offset=(offset+2<=dim_second)?offset+2:dim_second;
        for(count_second=offset;count_second<next_offset;count_second++){
          todst+=start_pos;tosrc+=start_pos;
          *todst=1./ *tosrc;tosrc++;todst++;
          for(count_diag=start_pos+1;count_diag<2;count_diag++){
            *todst=*tosrc;
            tosrc++;todst++;
          }
          tmp_dst+=2;todst=tmp_dst;
          tmp_src+=leading_dim;tosrc=tmp_src;
          start_pos++;
        }
        todst+=(dim_second-next_offset)*2;
        offset=next_offset;
    }
    for (;count_sub>0;count_first+=1,count_sub-=1){
        tosrc=src+count_first;
        for(count_second=0;count_second<offset;count_second++){
            *todst=(*tosrc);
            tosrc+=leading_dim;
            todst++;
        }
        start_pos=0;tmp_dst=todst;tmp_src=tosrc;
        next_offset=(offset+1<=dim_second)?offset+1:dim_second;
        for(count_second=offset;count_second<next_offset;count_second++){
          todst+=start_pos;tosrc+=start_pos;
          *todst=1./ *tosrc;tosrc++;todst++;
          for(count_diag=start_pos+1;count_diag<1;count_diag++){
            *todst=*tosrc;
            tosrc++;todst++;
          }
          tmp_dst+=1;todst=tmp_dst;
          tmp_src+=leading_dim;tosrc=tmp_src;
          start_pos++;
        }
        todst+=(dim_second-next_offset);
        offset=next_offset;
    }
}
#define init_m24n1_para(c1,c2,c3) \
  "vpxorq %%zmm"#c1",%%zmm"#c1",%%zmm"#c1";vpxorq %%zmm"#c2",%%zmm"#c2",%%zmm"#c2";vpxorq %%zmm"#c3",%%zmm"#c3",%%zmm"#c3";"
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
  "vbroadcastsd (%1),%%zmm4;vfnmadd231pd %%zmm1,%%zmm4,%%zmm8; vfnmadd231pd %%zmm2,%%zmm4,%%zmm9; vfnmadd231pd %%zmm3,%%zmm4,%%zmm10;"\
  "vbroadcastsd 8(%1),%%zmm5;vfnmadd231pd %%zmm1,%%zmm5,%%zmm11; vfnmadd231pd %%zmm2,%%zmm5,%%zmm12; vfnmadd231pd %%zmm3,%%zmm5,%%zmm13;prefetcht0 384(%0);"
#define kernel_m8n2_1 \
  "vbroadcastsd (%1),%%zmm4;vfnmadd231pd %%zmm1,%%zmm4,%%zmm8;"\
  "vbroadcastsd 8(%1),%%zmm5;vfnmadd231pd %%zmm1,%%zmm5,%%zmm11;"
#define kernel_m2n2_1 \
  "vmovddup (%1),%%xmm4;vfnmadd231pd %%xmm1,%%xmm4,%%xmm8;"\
  "vmovddup 8(%1),%%xmm5;vfnmadd231pd %%xmm1,%%xmm5,%%xmm9;"
#define kernel_m24n2_2 \
  "vbroadcastsd (%1,%%r11,1),%%zmm4;vfnmadd231pd %%zmm1,%%zmm4,%%zmm14; vfnmadd231pd %%zmm2,%%zmm4,%%zmm15; vfnmadd231pd %%zmm3,%%zmm4,%%zmm16;"\
  "vbroadcastsd 8(%1,%%r11,1),%%zmm5;vfnmadd231pd %%zmm1,%%zmm5,%%zmm17; vfnmadd231pd %%zmm2,%%zmm5,%%zmm18; vfnmadd231pd %%zmm3,%%zmm5,%%zmm19;prefetcht0 448(%0);"
#define kernel_m8n2_2 \
  "vbroadcastsd (%1,%%r11,1),%%zmm4;vfnmadd231pd %%zmm1,%%zmm4,%%zmm14;"\
  "vbroadcastsd 8(%1,%%r11,1),%%zmm5;vfnmadd231pd %%zmm1,%%zmm5,%%zmm17;"
#define kernel_m2n2_2 \
  "vmovddup (%1,%%r11,1),%%xmm4;vfnmadd231pd %%xmm1,%%xmm4,%%xmm10;"\
  "vmovddup 8(%1,%%r11,1),%%xmm5;vfnmadd231pd %%xmm1,%%xmm5,%%xmm11;"
#define kernel_m24n2_3 \
  "vbroadcastsd (%1,%%r11,2),%%zmm4;vfnmadd231pd %%zmm1,%%zmm4,%%zmm20; vfnmadd231pd %%zmm2,%%zmm4,%%zmm21; vfnmadd231pd %%zmm3,%%zmm4,%%zmm22;"\
  "vbroadcastsd 8(%1,%%r11,2),%%zmm5;vfnmadd231pd %%zmm1,%%zmm5,%%zmm23; vfnmadd231pd %%zmm2,%%zmm5,%%zmm24; vfnmadd231pd %%zmm3,%%zmm5,%%zmm25;prefetcht0 512(%0);"
#define kernel_m8n2_3 \
  "vbroadcastsd (%1,%%r11,2),%%zmm4;vfnmadd231pd %%zmm1,%%zmm4,%%zmm20;"\
  "vbroadcastsd 8(%1,%%r11,2),%%zmm5;vfnmadd231pd %%zmm1,%%zmm5,%%zmm23;"
#define kernel_m2n2_3 \
  "vmovddup (%1,%%r11,2),%%xmm4;vfnmadd231pd %%xmm1,%%xmm4,%%xmm12;"\
  "vmovddup 8(%1,%%r11,2),%%xmm5;vfnmadd231pd %%xmm1,%%xmm5,%%xmm13;"
#define kernel_m24n2_4 \
  "vbroadcastsd (%%r12),%%zmm4;vfnmadd231pd %%zmm1,%%zmm4,%%zmm26; vfnmadd231pd %%zmm2,%%zmm4,%%zmm27; vfnmadd231pd %%zmm3,%%zmm4,%%zmm28;"\
  "vbroadcastsd 8(%%r12),%%zmm5;vfnmadd231pd %%zmm1,%%zmm5,%%zmm29; vfnmadd231pd %%zmm2,%%zmm5,%%zmm30; vfnmadd231pd %%zmm3,%%zmm5,%%zmm31;"
#define kernel_m8n2_4 \
  "vbroadcastsd (%%r12),%%zmm4;vfnmadd231pd %%zmm1,%%zmm4,%%zmm26;"\
  "vbroadcastsd 8(%%r12),%%zmm5;vfnmadd231pd %%zmm1,%%zmm5,%%zmm29;"
#define kernel_m2n2_4 \
  "vmovddup (%%r12),%%xmm4;vfnmadd231pd %%xmm1,%%xmm4,%%xmm14;"\
  "vmovddup 8(%%r12),%%xmm5;vfnmadd231pd %%xmm1,%%xmm5,%%xmm15;"
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
#define KERNEL_m8n8 \
  LOAD_A_COL_m8 \
  kernel_m8n2_1 \
  kernel_m8n2_2 \
  kernel_m8n2_3 \
  kernel_m8n2_4 \
  "addq $16,%1;addq $16,%%r12;"
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
#define KERNEL_m2n4 \
  LOAD_A_COL_m2 \
  kernel_m2n2_1 \
  kernel_m2n2_2 \
  "addq $16,%1;"
#define save_m24n2_1 \
  "vaddpd (%3),%%zmm8,%%zmm8;vaddpd 64(%3),%%zmm9,%%zmm9;vaddpd 128(%3),%%zmm10,%%zmm10;"\
  "vmovups %%zmm8,(%3);vmovups %%zmm9,64(%3);vmovups %%zmm10,128(%3);"\
  "vaddpd (%3,%4,1),%%zmm11,%%zmm11;vaddpd 64(%3,%4,1),%%zmm12,%%zmm12;vaddpd 128(%3,%4,1),%%zmm13,%%zmm13;"\
  "vmovups %%zmm11,(%3,%4,1);vmovups %%zmm12,64(%3,%4,1);vmovups %%zmm13,128(%3,%4,1);leaq (%3,%4,2),%3;"
#define save_m8n2_1 \
  "vaddpd (%3),%%zmm8,%%zmm8;"\
  "vmovups %%zmm8,(%3);"\
  "vaddpd (%3,%4,1),%%zmm11,%%zmm11;"\
  "vmovups %%zmm11,(%3,%4,1);leaq (%3,%4,2),%3;"
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
#define SAVE_m24n8 \
  save_init_m24 \
  save_m24n2_1 \
  save_m24n2_2 \
  save_m24n2_3 \
  save_m24n2_4
#define SAVE_m8n8 \
  save_init_m8 \
  save_m8n2_1 \
  save_m8n2_2 \
  save_m8n2_3 \
  save_m8n2_4
#define SAVE_m2n8 \
  save_init_m2 \
  save_m2n2_1 \
  save_m2n2_2 \
  save_m2n2_3 \
  save_m2n2_4
#define SAVE_m24n4 \
  save_init_m24 \
  save_m24n2_1 \
  save_m24n2_2
#define SAVE_m8n4 \
  save_init_m8 \
  save_m8n2_1 \
  save_m8n2_2
#define SAVE_m2n4 \
  save_init_m2 \
  save_m2n2_1 \
  save_m2n2_2

#define COMPUTE_m24n8 \
    "movq %%r14,%1;leaq (%1,%%r11,2),%%r12;addq %%r11,%%r12;movq %6,%%r13;"\
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
    "movq %%r14,%1;leaq (%1,%%r11,2),%%r12;addq %%r11,%%r12;movq %6,%%r13;"\
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
    "movq %%r14,%1;leaq (%1,%%r11,2),%%r12;addq %%r11,%%r12;movq %6,%%r13;"\
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

#define COMPUTE_m2n8 \
    "movq %%r14,%1;leaq (%1,%%r11,2),%%r12;addq %%r11,%%r12;movq %6,%%r13;"\
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
    "movq %%r14,%1;leaq (%1,%%r11,2),%%r12;addq %%r11,%%r12;movq %6,%%r13;"\
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

#define COMPUTE_m2n4 \
    "movq %%r14,%1;leaq (%1,%%r11,2),%%r12;addq %%r11,%%r12;movq %6,%%r13;"\
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
    "movq %7,%%r15; movq %1,%%r14; movq %6,%%r11; salq $4,%%r11;"\
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
  :"+r"(a_ptr),"+r"(b_ptr),"+r"(c_ptr),"+r"(c_tmp),"+r"(ldc_in_bytes),"+r"(b_pref):"m"(K),"m"(M):"r10","r11","r12","r13","r14","r15","cc","memory",\
    "zmm0","zmm1","zmm2","zmm3","zmm4","zmm5","zmm6","zmm7","zmm8","zmm9","zmm10","zmm11","zmm12","zmm13","zmm14","zmm15",\
    "zmm16","zmm17","zmm18","zmm19","zmm20","zmm21","zmm22","zmm23","zmm24","zmm25","zmm26","zmm27","zmm28","zmm29","zmm30","zmm31");\
  a_ptr -= M * K; b_ptr += 8 * K; c_ptr += 8 * ldc - M;\
}

#define macro_n4 {\
  b_pref = b_ptr + 4 * K;\
  __asm__ __volatile__(\
    "movq %7,%%r15; movq %1,%%r14; movq %6,%%r11; salq $4,%%r11;"\
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
  :"+r"(a_ptr),"+r"(b_ptr),"+r"(c_ptr),"+r"(c_tmp),"+r"(ldc_in_bytes),"+r"(b_pref):"m"(K),"m"(M):"r10","r11","r12","r13","r14","r15","cc","memory",\
    "zmm0","zmm1","zmm2","zmm3","zmm4","zmm5","zmm6","zmm7","zmm8","zmm9","zmm10","zmm11","zmm12","zmm13","zmm14","zmm15",\
    "zmm16","zmm17","zmm18","zmm19","zmm20","zmm21","zmm22","zmm23","zmm24","zmm25","zmm26","zmm27","zmm28","zmm29","zmm30","zmm31");\
  a_ptr -= M * K; b_ptr += 8 * K; c_ptr += 8 * ldc - M;\
}


void macro_kernel_gemm(double *a_buffer,double *b_buffer,int m,int n,int k,double *C, int LDC){
    int m_count,n_count,m_count_sub,n_count_sub;
    // printf("INNER: m = %d, n = %d, k = %d\n",m,n,k);
    if (m==0||n==0||k==0) return;
    int64_t M=(int64_t)m,K=(int64_t)k,ldc_in_bytes=(int64_t)LDC*sizeof(double),ldc=(int32_t)LDC;
    double *a_ptr=a_buffer,*b_ptr=b_buffer,*c_ptr=C,*b_pref=b_ptr,*c_tmp=C;
    // printf("m= %d, n=%d, k = %d\n",m,n,k);
    for (n_count_sub=n,n_count=0;n_count_sub>7;n_count_sub-=8,n_count+=8){
        //call the m layer with n=8;
        macro_n8
    }
    for (;n_count_sub>3;n_count_sub-=4,n_count+=4){
        //call the m layer with n=4
        macro_n4
    }
    for (;n_count_sub>1;n_count_sub-=2,n_count+=2){
        //call the m layer with n=2
    }
    for (;n_count_sub>0;n_count_sub-=1,n_count+=1){
        //call the m layer with n=1
    }
}

static void solve(int m,int n,double *a,double *b,double *c,int ldc){
  double a0,b0;
  int i,j,k;
  for (i=0;i<m;i++){
    a0 = a[i*m+i];
    for (j = 0; j < n; j ++) {
      b0 = c[j*ldc+i] * a0;
      b[i*n+j] = c[j*ldc+i] = b0;
      for (k = i + 1; k < m; k ++) c[j*ldc+k] -= b0 * a[i*m+k];
    }
  }
}

#define SAVE_C_COL_m24n1(c1,c2,c3) \
  "vmovups %%zmm"#c1",(%3);vmovups %%zmm"#c2",64(%3);vmovups %%zmm"#c3",128(%3);leaq (%3,%4,1),%3;"

#define SAVE_C_COL_m24n2(c1,c2,c3,c4,c5,c6) \
  "vmovups %%zmm"#c1",(%3);vmovups %%zmm"#c2",64(%3);vmovups %%zmm"#c3",128(%3);"\
  "vmovups %%zmm"#c4",(%3,%4,1);vmovups %%zmm"#c5",64(%3,%4,1);vmovups %%zmm"#c6",128(%3,%4,1);leaq (%3,%4,2),%3;"
#define TRANS_2X8(c1,c2,c3,c4,c5,c6,c7,c8)\
  "vunpcklpd %%xmm"#c2",%%xmm"#c1",%%xmm0;vunpckhpd %%xmm"#c2",%%xmm"#c1",%%xmm1;"\
  "vunpcklpd %%xmm"#c4",%%xmm"#c3",%%xmm2;vunpckhpd %%xmm"#c4",%%xmm"#c3",%%xmm3;"\
  "vunpcklpd %%xmm"#c6",%%xmm"#c5",%%xmm4;vunpckhpd %%xmm"#c6",%%xmm"#c5",%%xmm5;"\
  "vunpcklpd %%xmm"#c8",%%xmm"#c7",%%xmm6;vunpckhpd %%xmm"#c8",%%xmm"#c7",%%xmm7;"\
  "vmovups %%xmm0,%%xmm"#c1";vmovups %%xmm1,%%xmm"#c2";"\
  "vmovups %%xmm2,%%xmm"#c3";vmovups %%xmm3,%%xmm"#c4";"\
  "vmovups %%xmm4,%%xmm"#c5";vmovups %%xmm5,%%xmm"#c6";"\
  "vmovups %%xmm6,%%xmm"#c7";vmovups %%xmm7,%%xmm"#c8";"

#define TRANS_8X8(c1,c2,c3,c4,c5,c6,c7,c8)\
  "vunpcklpd %%zmm"#c2",%%zmm"#c1",%%zmm0;vunpckhpd %%zmm"#c2",%%zmm"#c1",%%zmm1;"\
  "vunpcklpd %%zmm"#c4",%%zmm"#c3",%%zmm2;vunpckhpd %%zmm"#c4",%%zmm"#c3",%%zmm3;"\
  "vunpcklpd %%zmm"#c6",%%zmm"#c5",%%zmm4;vunpckhpd %%zmm"#c6",%%zmm"#c5",%%zmm5;"\
  "vunpcklpd %%zmm"#c8",%%zmm"#c7",%%zmm6;vunpckhpd %%zmm"#c8",%%zmm"#c7",%%zmm7;"\
  "vshuff64x2 $0x88,%%zmm4,%%zmm0,%%zmm"#c1";vshuff64x2 $0x88,%%zmm5,%%zmm1,%%zmm"#c2";"\
  "vshuff64x2 $0xdd,%%zmm4,%%zmm0,%%zmm"#c3";vshuff64x2 $0xdd,%%zmm5,%%zmm1,%%zmm"#c4";"\
  "vshuff64x2 $0x88,%%zmm6,%%zmm2,%%zmm"#c5";vshuff64x2 $0x88,%%zmm7,%%zmm3,%%zmm"#c6";"\
  "vshuff64x2 $0xdd,%%zmm6,%%zmm2,%%zmm"#c7";vshuff64x2 $0xdd,%%zmm7,%%zmm3,%%zmm"#c8";"\
  "vshuff64x2 $0x88,%%zmm"#c5",%%zmm"#c1",%%zmm0;vshuff64x2 $0x88,%%zmm"#c6",%%zmm"#c2",%%zmm1;"\
  "vshuff64x2 $0x88,%%zmm"#c7",%%zmm"#c3",%%zmm2;vshuff64x2 $0x88,%%zmm"#c8",%%zmm"#c4",%%zmm3;"\
  "vshuff64x2 $0xdd,%%zmm"#c5",%%zmm"#c1",%%zmm4;vshuff64x2 $0xdd,%%zmm"#c6",%%zmm"#c2",%%zmm5;"\
  "vshuff64x2 $0xdd,%%zmm"#c7",%%zmm"#c3",%%zmm6;vshuff64x2 $0xdd,%%zmm"#c8",%%zmm"#c4",%%zmm7;"\
  "vshuff64x2 $0xd8,%%zmm0,%%zmm0,%%zmm"#c1";vshuff64x2 $0xd8,%%zmm1,%%zmm1,%%zmm"#c2";"\
  "vshuff64x2 $0xd8,%%zmm2,%%zmm2,%%zmm"#c3";vshuff64x2 $0xd8,%%zmm3,%%zmm3,%%zmm"#c4";"\
  "vshuff64x2 $0xd8,%%zmm4,%%zmm4,%%zmm"#c5";vshuff64x2 $0xd8,%%zmm5,%%zmm5,%%zmm"#c6";"\
  "vshuff64x2 $0xd8,%%zmm6,%%zmm6,%%zmm"#c7";vshuff64x2 $0xd8,%%zmm7,%%zmm7,%%zmm"#c8";"

#define TRANS_4X8_B(c1,c2,c3,c4)\
  "vmovups %%zmm"#c1",%%zmm4;vmovups %%zmm"#c2",%%zmm5;vmovups %%zmm"#c3",%%zmm6;vmovups %%zmm"#c4",%%zmm7;"\
  "vshuff64x2 $0x88,%%zmm5,%%zmm4,%%zmm0;vshuff64x2 $0xdd,%%zmm5,%%zmm4,%%zmm1;"\
  "vshuff64x2 $0x88,%%zmm7,%%zmm6,%%zmm2;vshuff64x2 $0xdd,%%zmm7,%%zmm6,%%zmm3;"\
  "vshuff64x2 $0x88,%%zmm2,%%zmm0,%%zmm4;vshuff64x2 $0x88,%%zmm3,%%zmm1,%%zmm5;"\
  "vshuff64x2 $0xdd,%%zmm2,%%zmm0,%%zmm6;vshuff64x2 $0xdd,%%zmm3,%%zmm1,%%zmm7;"\
  "vmovups %%zmm4,(%1);vmovups %%zmm5,(%1,%%r12,2);vmovups %%zmm6,(%1,%%r12,4);vmovups %%zmm7,(%6);"\
  "addq $64,%1;addq $64,%6;"

#define TRANS_1X8_B(c1,c2,c3,c4)\
  "vmovups %%xmm"#c1",(%1);vmovups %%xmm"#c2",(%1,%%r12,2);vmovups %%xmm"#c3",(%1,%%r12,4);vmovups %%xmm"#c4",(%6);"\
  "addq $16,%1;addq $16,%6;"

#define LOAD_ADD_C_COL_m24(c1,c2,c3) \
  "vaddpd (%3),%%zmm"#c1",%%zmm"#c1";vaddpd 64(%3),%%zmm"#c2",%%zmm"#c2";vaddpd 128(%3),%%zmm"#c3",%%zmm"#c3";addq %4,%3;"

#define LOAD_C_COL_m24(c1,c2,c3) \
  "vmovups (%3),%%zmm"#c1";vmovups 64(%3),%%zmm"#c2";vmovups 128(%3),%%zmm"#c3";leaq (%3,%4,1),%3;"

#define kernel_trsm_m8n2_1 \
  "vbroadcastsd (%1),%%zmm4;vfnmadd231pd %%zmm1,%%zmm4,%%zmm8;"\
  "vbroadcastsd 8(%1),%%zmm5;vfnmadd231pd %%zmm1,%%zmm5,%%zmm11;"
#define kernel_trsm_m8n2_2 \
  "vbroadcastsd (%1,%%r12,2),%%zmm4;vfnmadd231pd %%zmm1,%%zmm4,%%zmm14;"\
  "vbroadcastsd 8(%1,%%r12,2),%%zmm5;vfnmadd231pd %%zmm1,%%zmm5,%%zmm17;"
#define kernel_trsm_m8n2_3 \
  "vbroadcastsd (%1,%%r12,4),%%zmm4;vfnmadd231pd %%zmm1,%%zmm4,%%zmm20;"\
  "vbroadcastsd 8(%1,%%r12,4),%%zmm5;vfnmadd231pd %%zmm1,%%zmm5,%%zmm23;"
#define kernel_trsm_m8n2_4 \
  "vbroadcastsd (%6),%%zmm4;vfnmadd231pd %%zmm1,%%zmm4,%%zmm26;"\
  "vbroadcastsd 8(%6),%%zmm5;vfnmadd231pd %%zmm1,%%zmm5,%%zmm29;"
#define kernel_trsm_m24n2_1 \
  "vbroadcastsd (%1),%%zmm4;vfnmadd231pd %%zmm1,%%zmm4,%%zmm8; vfnmadd231pd %%zmm2,%%zmm4,%%zmm9; vfnmadd231pd %%zmm3,%%zmm4,%%zmm10;"\
  "vbroadcastsd 8(%1),%%zmm5;vfnmadd231pd %%zmm1,%%zmm5,%%zmm11; vfnmadd231pd %%zmm2,%%zmm5,%%zmm12; vfnmadd231pd %%zmm3,%%zmm5,%%zmm13;prefetcht0 384(%0);"
#define kernel_trsm_m24n2_2 \
  "vbroadcastsd (%1,%%r12,2),%%zmm4;vfnmadd231pd %%zmm1,%%zmm4,%%zmm14; vfnmadd231pd %%zmm2,%%zmm4,%%zmm15; vfnmadd231pd %%zmm3,%%zmm4,%%zmm16;"\
  "vbroadcastsd 8(%1,%%r12,2),%%zmm5;vfnmadd231pd %%zmm1,%%zmm5,%%zmm17; vfnmadd231pd %%zmm2,%%zmm5,%%zmm18; vfnmadd231pd %%zmm3,%%zmm5,%%zmm19;"
#define kernel_trsm_m24n2_3 \
  "vbroadcastsd (%1,%%r12,4),%%zmm4;vfnmadd231pd %%zmm1,%%zmm4,%%zmm20; vfnmadd231pd %%zmm2,%%zmm4,%%zmm21; vfnmadd231pd %%zmm3,%%zmm4,%%zmm22;"\
  "vbroadcastsd 8(%1,%%r12,4),%%zmm5;vfnmadd231pd %%zmm1,%%zmm5,%%zmm23; vfnmadd231pd %%zmm2,%%zmm5,%%zmm24; vfnmadd231pd %%zmm3,%%zmm5,%%zmm25;prefetcht0 448(%0);"
#define kernel_trsm_m24n2_4 \
  "vbroadcastsd (%6),%%zmm4;vfnmadd231pd %%zmm1,%%zmm4,%%zmm26; vfnmadd231pd %%zmm2,%%zmm4,%%zmm27; vfnmadd231pd %%zmm3,%%zmm4,%%zmm28;"\
  "vbroadcastsd 8(%6),%%zmm5;vfnmadd231pd %%zmm1,%%zmm5,%%zmm29; vfnmadd231pd %%zmm2,%%zmm5,%%zmm30; vfnmadd231pd %%zmm3,%%zmm5,%%zmm31;prefetcht0 512(%0);"
#define kernel_trsm_m2n2_1 \
  "vmovddup (%1),%%xmm4;vfnmadd231pd %%xmm1,%%xmm4,%%xmm8;"\
  "vmovddup 8(%1),%%xmm5;vfnmadd231pd %%xmm1,%%xmm5,%%xmm9;"
#define kernel_trsm_m2n2_2 \
  "vmovddup (%1,%%r12,2),%%xmm4;vfnmadd231pd %%xmm1,%%xmm4,%%xmm10;"\
  "vmovddup 8(%1,%%r12,2),%%xmm5;vfnmadd231pd %%xmm1,%%xmm5,%%xmm11;"
#define kernel_trsm_m2n2_3 \
  "vmovddup (%1,%%r12,4),%%xmm4;vfnmadd231pd %%xmm1,%%xmm4,%%xmm12;"\
  "vmovddup 8(%1,%%r12,4),%%xmm5;vfnmadd231pd %%xmm1,%%xmm5,%%xmm13;"
#define kernel_trsm_m2n2_4 \
  "vmovddup (%6),%%xmm4;vfnmadd231pd %%xmm1,%%xmm4,%%xmm14;"\
  "vmovddup 8(%6),%%xmm5;vfnmadd231pd %%xmm1,%%xmm5,%%xmm15;"

#define KERNEL_trsm_m24n8 \
  LOAD_A_COL_m24 \
  kernel_trsm_m24n2_1 \
  kernel_trsm_m24n2_2 \
  kernel_trsm_m24n2_3 \
  kernel_trsm_m24n2_4 \
  "addq $16,%1;addq $16,%6;"
#define KERNEL_trsm_m24n4 \
  LOAD_A_COL_m24 \
  kernel_trsm_m24n2_1 \
  kernel_trsm_m24n2_2 \
  "addq $16,%1;addq $16,%6;"
#define KERNEL_trsm_m8n8 \
  LOAD_A_COL_m8 \
  kernel_trsm_m8n2_1 \
  kernel_trsm_m8n2_2 \
  kernel_trsm_m8n2_3 \
  kernel_trsm_m8n2_4 \
  "addq $16,%1;addq $16,%6;"
#define KERNEL_trsm_m8n4 \
  LOAD_A_COL_m8 \
  kernel_trsm_m8n2_1 \
  kernel_trsm_m8n2_2 \
  "addq $16,%1;addq $16,%6;"
#define KERNEL_trsm_m2n8 \
  LOAD_A_COL_m2 \
  kernel_trsm_m2n2_1 \
  kernel_trsm_m2n2_2 \
  kernel_trsm_m2n2_3 \
  kernel_trsm_m2n2_4 \
  "addq $16,%1;addq $16,%6;"
#define KERNEL_trsm_m2n4 \
  LOAD_A_COL_m2 \
  kernel_trsm_m2n2_1 \
  kernel_trsm_m2n2_2 \
  "addq $16,%1;addq $16,%6;"
#define init_m24n1_trsm \
  "prefetcht1 (%3);vpxorq %%zmm8,%%zmm8,%%zmm8;prefetcht1 64(%3);vpxorq %%zmm9,%%zmm9,%%zmm9;prefetcht1 128(%3);vpxorq %%zmm10,%%zmm10,%%zmm10;addq %4,%3;"
#define init_m24n2_trsm \
  init_m24n1_trsm \
  "prefetcht1 (%3);vpxorq %%zmm11,%%zmm11,%%zmm11;prefetcht1 64(%3);vpxorq %%zmm12,%%zmm12,%%zmm12;prefetcht1 128(%3);vpxorq %%zmm13,%%zmm13,%%zmm13;addq %4,%3;"
#define init_m24n4_trsm \
  init_m24n2_trsm \
  "prefetcht1 (%3);vpxorq %%zmm14,%%zmm14,%%zmm14;prefetcht1 64(%3);vpxorq %%zmm15,%%zmm15,%%zmm15;prefetcht1 128(%3);vpxorq %%zmm16,%%zmm16,%%zmm16;"\
  "prefetcht1 (%3,%4,1);vpxorq %%zmm17,%%zmm17,%%zmm17;prefetcht1 64(%3,%4,1);vpxorq %%zmm18,%%zmm18,%%zmm18;prefetcht1 128(%3,%4,1);vpxorq %%zmm19,%%zmm19,%%zmm19;leaq (%3,%4,2),%3;"
#define init_m24n6_trsm \
  init_m24n4_trsm \
  "prefetcht1 (%3);vpxorq %%zmm20,%%zmm20,%%zmm20;prefetcht1 64(%3);vpxorq %%zmm21,%%zmm21,%%zmm21;prefetcht1 128(%3);vpxorq %%zmm22,%%zmm22,%%zmm22;"\
  "prefetcht1 (%3,%4,1);vpxorq %%zmm23,%%zmm23,%%zmm23;prefetcht1 64(%3,%4,1);vpxorq %%zmm24,%%zmm24,%%zmm24;prefetcht1 128(%3,%4,1);vpxorq %%zmm25,%%zmm25,%%zmm25;leaq (%3,%4,2),%3;"
#define init_m24n8_trsm \
  init_m24n6_trsm \
  "prefetcht1 (%3);vpxorq %%zmm26,%%zmm26,%%zmm26;prefetcht1 64(%3);vpxorq %%zmm27,%%zmm27,%%zmm27;prefetcht1 128(%3);vpxorq %%zmm28,%%zmm28,%%zmm28;"\
  "prefetcht1 (%3,%4,1);vpxorq %%zmm29,%%zmm29,%%zmm29;prefetcht1 64(%3,%4,1);vpxorq %%zmm30,%%zmm30,%%zmm30;prefetcht1 128(%3,%4,1);vpxorq %%zmm31,%%zmm31,%%zmm31;"
#define GEMM_LT_m24n8 \
  "movq %2,%3;leaq (%%r12,%%r12,2),%6;movq %%r14,%1;addq %6,%6;"\
  "movq %%r15,%0; leaq (%%r15,%6,4),%%r15; movq %%r13,%5;"\
  "leaq (%1,%%r12,4),%6;leaq (%6,%%r12,2),%6;"\
  init_m24n8_trsm\
  "cmpq $4,%5; jb 124182f;"\
  "124181:\n\t"\
  KERNEL_trsm_m24n8 "subq $4,%5;"\
  KERNEL_trsm_m24n8 \
  KERNEL_trsm_m24n8 \
  KERNEL_trsm_m24n8 \
  "cmpq $24,%5; jnb 124181b;"\
  "movq %2,%3;"\
  "124182:\n\t"\
  "testq %5,%5; jz 124184f;"\
  "124183:\n\t"\
  KERNEL_trsm_m24n8  PREF_TRSM_ADD_C_COL_m24 "subq $2,%5;"\
  KERNEL_trsm_m24n8  PREF_TRSM_ADD_C_COL_m24\
  "testq %5,%5;jnz 124183b;"\
  "124184:\n\t"\
  "addq $24,%%r13;"

#define GEMM_LT_m24n4 \
  "movq %2,%3;leaq (%%r12,%%r12,2),%6;movq %%r14,%1;addq %6,%6;"\
  "movq %%r15,%0; leaq (%%r15,%6,4),%%r15; movq %%r13,%5;"\
  "leaq (%1,%%r12,4),%6;leaq (%6,%%r12,2),%6;"\
  init_m24n4_trsm\
  "cmpq $4,%5; jb 124142f;"\
  "124141:\n\t"\
  KERNEL_trsm_m24n4 "subq $4,%5;"\
  KERNEL_trsm_m24n4 \
  KERNEL_trsm_m24n4 \
  KERNEL_trsm_m24n4 \
  "cmpq $24,%5; jnb 124141b;"\
  "movq %2,%3;"\
  "124142:\n\t"\
  "testq %5,%5; jz 124144f;"\
  "124143:\n\t"\
  KERNEL_trsm_m24n4  PREF_TRSM_ADD_C_COL_m24 "subq $2,%5;"\
  KERNEL_trsm_m24n4  PREF_TRSM_ADD_C_COL_m24\
  "testq %5,%5;jnz 124143b;"\
  "124144:\n\t"\
  "addq $24,%%r13;"


#define GEMM_LT_m8n8 \
  "movq %2,%3;movq %%r14,%1;"\
  "movq %%r15,%0; leaq (%%r15,%%r12,8),%%r15; movq %%r13,%5;"\
  "leaq (%1,%%r12,4),%6;leaq (%6,%%r12,2),%6;"\
  init_m8n8\
  "cmpq $4,%5; jb 18182f;"\
  "18181:\n\t"\
  KERNEL_trsm_m8n8 "subq $4,%5;"\
  KERNEL_trsm_m8n8 \
  KERNEL_trsm_m8n8 \
  KERNEL_trsm_m8n8 \
  "cmpq $8,%5; jnb 18181b;"\
  "movq %2,%3;"\
  "18182:\n\t"\
  "testq %5,%5; jz 18184f;"\
  "18183:\n\t"\
  KERNEL_trsm_m8n8 "subq $2,%5;"\
  KERNEL_trsm_m8n8 \
  "testq %5,%5;jnz 18183b;"\
  "18184:\n\t"\
  "addq $8,%%r13;"

#define GEMM_LT_m8n4 \
  "movq %2,%3;movq %%r14,%1;"\
  "movq %%r15,%0; leaq (%%r15,%%r12,8),%%r15; movq %%r13,%5;"\
  "leaq (%1,%%r12,4),%6;leaq (%6,%%r12,2),%6;"\
  init_m8n4\
  "cmpq $4,%5; jb 18142f;"\
  "18141:\n\t"\
  KERNEL_trsm_m8n4 "subq $4,%5;"\
  KERNEL_trsm_m8n4 \
  KERNEL_trsm_m8n4 \
  KERNEL_trsm_m8n4 \
  "cmpq $8,%5; jnb 18141b;"\
  "movq %2,%3;"\
  "18142:\n\t"\
  "testq %5,%5; jz 18144f;"\
  "18143:\n\t"\
  KERNEL_trsm_m8n4 "subq $2,%5;"\
  KERNEL_trsm_m8n4 \
  "testq %5,%5;jnz 18143b;"\
  "18144:\n\t"\
  "addq $8,%%r13;"

#define GEMM_LT_m2n8 \
  "movq %2,%3;movq %%r14,%1;"\
  "movq %%r15,%0; leaq (%%r15,%%r12,2),%%r15; movq %%r13,%5;addq $2,%%r13;"\
  "leaq (%1,%%r12,4),%6;leaq (%6,%%r12,2),%6;"\
  init_m2n8\
  "cmpq $4,%5; jb 12182f;"\
  "12183:\n\t"\
  KERNEL_trsm_m2n8 "decq %5;"\
  "testq %5,%5;jnz 12183b;"\
  "12182:\n\t"\

#define GEMM_LT_m2n4 \
  "movq %2,%3;movq %%r14,%1;"\
  "movq %%r15,%0; leaq (%%r15,%%r12,2),%%r15; movq %%r13,%5;addq $2,%%r13;"\
  "leaq (%1,%%r12,4),%6;leaq (%6,%%r12,2),%6;"\
  init_m2n4\
  "cmpq $4,%5; jb 12142f;"\
  "12143:\n\t"\
  KERNEL_trsm_m2n4 "decq %5;"\
  "testq %5,%5;jnz 12143b;"\
  "12142:\n\t"\

#define solve_lt_m1n1(offset,c1)\
  "vbroadcastsd "#offset"(%0),%%zmm0; vmulpd %%zmm0, %%zmm"#c1", %%zmm"#c1";"
#define substract_m1n1(offset,c1,c2)\
  "vbroadcastsd "#offset"(%0),%%zmm0; vfnmadd231pd %%zmm0,%%zmm"#c1",%%zmm"#c2";"

#define solve_lt_m24n8 \
  solve_lt_m1n1(0,8) "prefetcht0 832(%0);"\
  substract_m1n1(8,8,11) substract_m1n1(16,8,14)\
  substract_m1n1(24,8,17) substract_m1n1(32,8,20)\
  substract_m1n1(40,8,23) substract_m1n1(48,8,26)\
  substract_m1n1(56,8,29) substract_m1n1(64,8,9) "prefetcht0 896(%0);"\
  substract_m1n1(72,8,12) substract_m1n1(80,8,15)\
  substract_m1n1(88,8,18) substract_m1n1(96,8,21)\
  substract_m1n1(104,8,24) substract_m1n1(112,8,27)\
  substract_m1n1(120,8,30) substract_m1n1(128,8,10) "prefetcht0 960(%0);"\
  substract_m1n1(136,8,13) substract_m1n1(144,8,16)\
  substract_m1n1(152,8,19) substract_m1n1(160,8,22)\
  substract_m1n1(168,8,25) substract_m1n1(176,8,28) substract_m1n1(184,8,31) "prefetcht0 1024(%0);"\
  solve_lt_m1n1(200,11)\
  substract_m1n1(208,11,14) substract_m1n1(216,11,17)\
  substract_m1n1(224,11,20)\
  substract_m1n1(232,11,23) substract_m1n1(240,11,26)\
  substract_m1n1(248,11,29) substract_m1n1(256,11,9) "prefetcht0 1088(%0);"\
  substract_m1n1(264,11,12) substract_m1n1(272,11,15)\
  substract_m1n1(280,11,18) substract_m1n1(288,11,21)\
  substract_m1n1(296,11,24) substract_m1n1(304,11,27)\
  substract_m1n1(312,11,30) substract_m1n1(320,11,10) "prefetcht0 1152(%0);"\
  substract_m1n1(328,11,13) substract_m1n1(336,11,16)\
  substract_m1n1(344,11,19) substract_m1n1(352,11,22)\
  substract_m1n1(360,11,25) substract_m1n1(368,11,28) substract_m1n1(376,11,31) "prefetcht0 1216(%0);"\
  solve_lt_m1n1(400,14)\
  substract_m1n1(408,14,17) substract_m1n1(416,14,20)\
  substract_m1n1(424,14,23)\
  substract_m1n1(432,14,26) substract_m1n1(440,14,29)\
  substract_m1n1(448,14,9) "prefetcht0 1280(%0);" substract_m1n1(456,14,12)\
  substract_m1n1(464,14,15) substract_m1n1(472,14,18)\
  substract_m1n1(480,14,21) substract_m1n1(488,14,24)\
  substract_m1n1(496,14,27) substract_m1n1(504,14,30)\
  substract_m1n1(512,14,10) "prefetcht0 1344(%0);" substract_m1n1(520,14,13)\
  substract_m1n1(528,14,16) substract_m1n1(536,14,19)\
  substract_m1n1(544,14,22) substract_m1n1(552,14,25)\
  substract_m1n1(560,14,28) substract_m1n1(568,14,31) "prefetcht0 1408(%0);"\
  solve_lt_m1n1(600,17)\
  substract_m1n1(608,17,20) substract_m1n1(616,17,23)\
  substract_m1n1(624,17,26)\
  substract_m1n1(632,17,29) substract_m1n1(640,17,9) "prefetcht0 1472(%0);"\
  substract_m1n1(648,17,12) substract_m1n1(656,17,15)\
  substract_m1n1(664,17,18) substract_m1n1(672,17,21)\
  substract_m1n1(680,17,24) substract_m1n1(688,17,27)\
  substract_m1n1(696,17,30) substract_m1n1(704,17,10) "prefetcht0 1536(%0);"\
  substract_m1n1(712,17,13) substract_m1n1(720,17,16)\
  substract_m1n1(728,17,19) substract_m1n1(736,17,22)\
  substract_m1n1(744,17,25) substract_m1n1(752,17,28)\
  substract_m1n1(760,17,31) "prefetcht0 1600(%0);"\
  solve_lt_m1n1(800,20)\
  substract_m1n1(808,20,23) substract_m1n1(816,20,26)\
  substract_m1n1(824,20,29)\
  substract_m1n1(832,20,9) "prefetcht0 1664(%0);" substract_m1n1(840,20,12)\
  substract_m1n1(848,20,15) substract_m1n1(856,20,18)\
  substract_m1n1(864,20,21) substract_m1n1(872,20,24)\
  substract_m1n1(880,20,27) substract_m1n1(888,20,30)\
  substract_m1n1(896,20,10) "prefetcht0 1728(%0);" substract_m1n1(904,20,13)\
  substract_m1n1(912,20,16) substract_m1n1(920,20,19)\
  substract_m1n1(928,20,22) substract_m1n1(936,20,25)\
  substract_m1n1(944,20,28) substract_m1n1(952,20,31) "prefetcht0 1792(%0);"\
  solve_lt_m1n1(1000,23)\
  substract_m1n1(1008,23,26) substract_m1n1(1016,23,29)\
  substract_m1n1(1024,23,9) "prefetcht0 1856(%0);"\
  substract_m1n1(1032,23,12) substract_m1n1(1040,23,15)\
  substract_m1n1(1048,23,18) substract_m1n1(1056,23,21)\
  substract_m1n1(1064,23,24) substract_m1n1(1072,23,27)\
  substract_m1n1(1080,23,30) substract_m1n1(1088,23,10) "prefetcht0 1920(%0);"\
  substract_m1n1(1096,23,13) substract_m1n1(1104,23,16)\
  substract_m1n1(1112,23,19) substract_m1n1(1120,23,22)\
  substract_m1n1(1128,23,25) substract_m1n1(1136,23,28)\
  substract_m1n1(1144,23,31) "prefetcht0 1984(%0);"\
  solve_lt_m1n1(1200,26)\
  substract_m1n1(1208,26,29) substract_m1n1(1216,26,9) "prefetcht0 2048(%0);"\
  substract_m1n1(1224,26,12)\
  substract_m1n1(1232,26,15) substract_m1n1(1240,26,18)\
  substract_m1n1(1248,26,21) substract_m1n1(1256,26,24)\
  substract_m1n1(1264,26,27) substract_m1n1(1272,26,30)\
  substract_m1n1(1280,26,10) "prefetcht0 2112(%0);" substract_m1n1(1288,26,13)\
  substract_m1n1(1296,26,16) substract_m1n1(1304,26,19)\
  substract_m1n1(1312,26,22) substract_m1n1(1320,26,25)\
  substract_m1n1(1328,26,28) substract_m1n1(1336,26,31) "prefetcht0 2176(%0);"\
  solve_lt_m1n1(1400,29)\
  substract_m1n1(1408,29,9) "prefetcht0 2240(%0);" substract_m1n1(1416,29,12)\
  substract_m1n1(1424,29,15)\
  substract_m1n1(1432,29,18) substract_m1n1(1440,29,21)\
  substract_m1n1(1448,29,24) substract_m1n1(1456,29,27)\
  substract_m1n1(1464,29,30) substract_m1n1(1472,29,10) "prefetcht0 2304(%0);"\
  substract_m1n1(1480,29,13) substract_m1n1(1488,29,16)\
  substract_m1n1(1496,29,19) substract_m1n1(1504,29,22)\
  substract_m1n1(1512,29,25) substract_m1n1(1520,29,28)\
  substract_m1n1(1528,29,31) "prefetcht0 2368(%0);"\
  solve_lt_m1n1(1600,9) "prefetcht0 2432(%0);"\
  substract_m1n1(1608,9,12) substract_m1n1(1616,9,15)\
  substract_m1n1(1624,9,18)\
  substract_m1n1(1632,9,21) substract_m1n1(1640,9,24)\
  substract_m1n1(1648,9,27) substract_m1n1(1656,9,30)\
  substract_m1n1(1664,9,10) "prefetcht0 2496(%0);" substract_m1n1(1672,9,13)\
  substract_m1n1(1680,9,16) substract_m1n1(1688,9,19)\
  substract_m1n1(1696,9,22) substract_m1n1(1704,9,25)\
  substract_m1n1(1712,9,28) substract_m1n1(1720,9,31) "prefetcht0 2560(%0);"\
  solve_lt_m1n1(1800,12) "prefetcht0 2624(%0);"\
  substract_m1n1(1808,12,15) substract_m1n1(1816,12,18)\
  substract_m1n1(1824,12,21)\
  substract_m1n1(1832,12,24) substract_m1n1(1840,12,27)\
  substract_m1n1(1848,12,30) substract_m1n1(1856,12,10) "prefetcht0 2688(%0);"\
  substract_m1n1(1864,12,13) substract_m1n1(1872,12,16)\
  substract_m1n1(1880,12,19) substract_m1n1(1888,12,22)\
  substract_m1n1(1896,12,25) substract_m1n1(1904,12,28)\
  substract_m1n1(1912,12,31) "prefetcht0 2752(%0);"\
  solve_lt_m1n1(2000,15) "prefetcht0 2816(%0);"\
  substract_m1n1(2008,15,18) substract_m1n1(2016,15,21)\
  substract_m1n1(2024,15,24)\
  substract_m1n1(2032,15,27) substract_m1n1(2040,15,30)\
  substract_m1n1(2048,15,10) "prefetcht0 2880(%0);" substract_m1n1(2056,15,13)\
  substract_m1n1(2064,15,16) substract_m1n1(2072,15,19)\
  substract_m1n1(2080,15,22) substract_m1n1(2088,15,25)\
  substract_m1n1(2096,15,28) substract_m1n1(2104,15,31) "prefetcht0 2944(%0);"\
  solve_lt_m1n1(2200,18) "prefetcht0 3008(%0);"\
  substract_m1n1(2208,18,21) substract_m1n1(2216,18,24)\
  substract_m1n1(2224,18,27)\
  substract_m1n1(2232,18,30) substract_m1n1(2240,18,10) "prefetcht0 3072(%0);"\
  substract_m1n1(2248,18,13) substract_m1n1(2256,18,16)\
  substract_m1n1(2264,18,19) substract_m1n1(2272,18,22)\
  substract_m1n1(2280,18,25) substract_m1n1(2288,18,28)\
  substract_m1n1(2296,18,31) "prefetcht0 3136(%0);"\
  solve_lt_m1n1(2400,21) "prefetcht0 3200(%0);"\
  substract_m1n1(2408,21,24) substract_m1n1(2416,21,27)\
  substract_m1n1(2424,21,30)\
  substract_m1n1(2432,21,10) "prefetcht0 3264(%0);" substract_m1n1(2440,21,13)\
  substract_m1n1(2448,21,16) substract_m1n1(2456,21,19)\
  substract_m1n1(2464,21,22) substract_m1n1(2472,21,25)\
  substract_m1n1(2480,21,28) substract_m1n1(2488,21,31) "prefetcht0 3328(%0);"\
  solve_lt_m1n1(2600,24) "prefetcht0 3392(%0);"\
  substract_m1n1(2608,24,27) substract_m1n1(2616,24,30)\
  substract_m1n1(2624,24,10)\
  substract_m1n1(2632,24,13) "prefetcht0 3456(%0);" substract_m1n1(2640,24,16)\
  substract_m1n1(2648,24,19) substract_m1n1(2656,24,22)\
  substract_m1n1(2664,24,25) substract_m1n1(2672,24,28)\
  substract_m1n1(2680,24,31) "prefetcht0 3520(%0);"\
  solve_lt_m1n1(2800,27)\
  substract_m1n1(2808,27,30) substract_m1n1(2816,27,10)\
  substract_m1n1(2824,27,13) "prefetcht0 3584(%0);"\
  substract_m1n1(2832,27,16) substract_m1n1(2840,27,19)\
  substract_m1n1(2848,27,22) substract_m1n1(2856,27,25)\
  substract_m1n1(2864,27,28) substract_m1n1(2872,27,31) "prefetcht0 3648(%0);"\
  solve_lt_m1n1(3000,30)\
  substract_m1n1(3008,30,10) substract_m1n1(3016,30,13) "prefetcht0 3712(%0);"\
  substract_m1n1(3024,30,16)\
  substract_m1n1(3032,30,19) substract_m1n1(3040,30,22)\
  substract_m1n1(3048,30,25) substract_m1n1(3056,30,28)\
  substract_m1n1(3064,30,31) "prefetcht0 3776(%0);"\
  solve_lt_m1n1(3200,10) "prefetcht0 3840(%0);"\
  substract_m1n1(3208,10,13) substract_m1n1(3216,10,16)\
  substract_m1n1(3224,10,19)\
  substract_m1n1(3232,10,22) substract_m1n1(3240,10,25)\
  substract_m1n1(3248,10,28) substract_m1n1(3256,10,31) "prefetcht0 3904(%0);"\
  solve_lt_m1n1(3400,13)\
  substract_m1n1(3408,13,16) substract_m1n1(3416,13,19) "prefetcht0 3968(%0);"\
  substract_m1n1(3424,13,22)\
  substract_m1n1(3432,13,25) substract_m1n1(3440,13,28)\
  substract_m1n1(3448,13,31) "prefetcht0 4032(%0);"\
  solve_lt_m1n1(3600,16) "prefetcht0 4096(%0);"\
  substract_m1n1(3608,16,19) "prefetcht0 4160(%0);" substract_m1n1(3616,16,22)\
  substract_m1n1(3624,16,25) "prefetcht0 4224(%0);"\
  substract_m1n1(3632,16,28) substract_m1n1(3640,16,31)\
  solve_lt_m1n1(3800,19)\
  substract_m1n1(3808,19,22) substract_m1n1(3816,19,25)\
  substract_m1n1(3824,19,28)\
  substract_m1n1(3832,19,31)\
  solve_lt_m1n1(4000,22)\
  substract_m1n1(4008,22,25) substract_m1n1(4016,22,28)\
  substract_m1n1(4024,22,31)\
  solve_lt_m1n1(4200,25)\
  substract_m1n1(4208,25,28) substract_m1n1(4216,25,31)\
  solve_lt_m1n1(4400,28)\
  substract_m1n1(4408,28,31)\
  solve_lt_m1n1(4600,31)\


#define solve_lt_m8n8 \
  solve_lt_m1n1(0,8) \
  substract_m1n1(8,8,11) substract_m1n1(16,8,14)\
  substract_m1n1(24,8,17) substract_m1n1(32,8,20)\
  substract_m1n1(40,8,23) substract_m1n1(48,8,26)\
  substract_m1n1(56,8,29)\
  solve_lt_m1n1(72,11)\
  substract_m1n1(80,11,14) substract_m1n1(88,11,17)\
  substract_m1n1(96,11,20)\
  substract_m1n1(104,11,23) substract_m1n1(112,11,26)\
  substract_m1n1(120,11,29)\
  solve_lt_m1n1(144,14)\
  substract_m1n1(152,14,17) substract_m1n1(160,14,20)\
  substract_m1n1(168,14,23)\
  substract_m1n1(176,14,26) substract_m1n1(184,14,29)\
  solve_lt_m1n1(216,17)\
  substract_m1n1(224,17,20) substract_m1n1(232,17,23)\
  substract_m1n1(240,17,26)\
  substract_m1n1(248,17,29)\
  solve_lt_m1n1(288,20)\
  substract_m1n1(296,20,23) substract_m1n1(304,20,26)\
  substract_m1n1(312,20,29)\
  solve_lt_m1n1(360,23)\
  substract_m1n1(368,23,26) substract_m1n1(376,23,29)\
  solve_lt_m1n1(432,26)\
  substract_m1n1(440,26,29)\
  solve_lt_m1n1(504,29)


#define solve_lt_m1n1_m2n8(offset,c1)\
  "vmovddup "#offset"(%0),%%xmm0; vmulpd %%xmm0, %%xmm"#c1", %%xmm"#c1";"
#define substract_m1n1_m2n8(offset,c1,c2)\
  "vmovddup "#offset"(%0),%%xmm0; vfnmadd231pd %%xmm0,%%xmm"#c1",%%xmm"#c2";"

#define solve_lt_m2n8 \
  solve_lt_m1n1_m2n8(0,8)\
  solve_lt_m1n1_m2n8(0,10)\
  solve_lt_m1n1_m2n8(0,12)\
  solve_lt_m1n1_m2n8(0,14)\
  substract_m1n1_m2n8(8,8,9)\
  substract_m1n1_m2n8(8,10,11)\
  substract_m1n1_m2n8(8,12,13)\
  substract_m1n1_m2n8(8,14,15)\
  solve_lt_m1n1_m2n8(24,9)\
  solve_lt_m1n1_m2n8(24,11)\
  solve_lt_m1n1_m2n8(24,13)\
  solve_lt_m1n1_m2n8(24,15)

#define solve_lt_m24n8_npf \
  solve_lt_m1n1(0,8) \
  substract_m1n1(8,8,11) substract_m1n1(16,8,14)\
  substract_m1n1(24,8,17) substract_m1n1(32,8,20)\
  substract_m1n1(40,8,23) substract_m1n1(48,8,26)\
  substract_m1n1(56,8,29) substract_m1n1(64,8,9) \
  substract_m1n1(72,8,12) substract_m1n1(80,8,15)\
  substract_m1n1(88,8,18) substract_m1n1(96,8,21)\
  substract_m1n1(104,8,24) substract_m1n1(112,8,27)\
  substract_m1n1(120,8,30) substract_m1n1(128,8,10) \
  substract_m1n1(136,8,13) substract_m1n1(144,8,16)\
  substract_m1n1(152,8,19) substract_m1n1(160,8,22)\
  substract_m1n1(168,8,25) substract_m1n1(176,8,28) substract_m1n1(184,8,31) \
  solve_lt_m1n1(200,11)\
  substract_m1n1(208,11,14) substract_m1n1(216,11,17)\
  substract_m1n1(224,11,20)\
  substract_m1n1(232,11,23) substract_m1n1(240,11,26)\
  substract_m1n1(248,11,29) substract_m1n1(256,11,9) \
  substract_m1n1(264,11,12) substract_m1n1(272,11,15)\
  substract_m1n1(280,11,18) substract_m1n1(288,11,21)\
  substract_m1n1(296,11,24) substract_m1n1(304,11,27)\
  substract_m1n1(312,11,30) substract_m1n1(320,11,10) \
  substract_m1n1(328,11,13) substract_m1n1(336,11,16)\
  substract_m1n1(344,11,19) substract_m1n1(352,11,22)\
  substract_m1n1(360,11,25) substract_m1n1(368,11,28) substract_m1n1(376,11,31) \
  solve_lt_m1n1(400,14)\
  substract_m1n1(408,14,17) substract_m1n1(416,14,20)\
  substract_m1n1(424,14,23)\
  substract_m1n1(432,14,26) substract_m1n1(440,14,29)\
  substract_m1n1(448,14,9) substract_m1n1(456,14,12)\
  substract_m1n1(464,14,15) substract_m1n1(472,14,18)\
  substract_m1n1(480,14,21) substract_m1n1(488,14,24)\
  substract_m1n1(496,14,27) substract_m1n1(504,14,30)\
  substract_m1n1(512,14,10) substract_m1n1(520,14,13)\
  substract_m1n1(528,14,16) substract_m1n1(536,14,19)\
  substract_m1n1(544,14,22) substract_m1n1(552,14,25)\
  substract_m1n1(560,14,28) substract_m1n1(568,14,31) \
  solve_lt_m1n1(600,17)\
  substract_m1n1(608,17,20) substract_m1n1(616,17,23)\
  substract_m1n1(624,17,26)\
  substract_m1n1(632,17,29) substract_m1n1(640,17,9) \
  substract_m1n1(648,17,12) substract_m1n1(656,17,15)\
  substract_m1n1(664,17,18) substract_m1n1(672,17,21)\
  substract_m1n1(680,17,24) substract_m1n1(688,17,27)\
  substract_m1n1(696,17,30) substract_m1n1(704,17,10) \
  substract_m1n1(712,17,13) substract_m1n1(720,17,16)\
  substract_m1n1(728,17,19) substract_m1n1(736,17,22)\
  substract_m1n1(744,17,25) substract_m1n1(752,17,28)\
  substract_m1n1(760,17,31) \
  solve_lt_m1n1(800,20)\
  substract_m1n1(808,20,23) substract_m1n1(816,20,26)\
  substract_m1n1(824,20,29)\
  substract_m1n1(832,20,9) substract_m1n1(840,20,12)\
  substract_m1n1(848,20,15) substract_m1n1(856,20,18)\
  substract_m1n1(864,20,21) substract_m1n1(872,20,24)\
  substract_m1n1(880,20,27) substract_m1n1(888,20,30)\
  substract_m1n1(896,20,10) substract_m1n1(904,20,13)\
  substract_m1n1(912,20,16) substract_m1n1(920,20,19)\
  substract_m1n1(928,20,22) substract_m1n1(936,20,25)\
  substract_m1n1(944,20,28) substract_m1n1(952,20,31)\
  solve_lt_m1n1(1000,23)\
  substract_m1n1(1008,23,26) substract_m1n1(1016,23,29)\
  substract_m1n1(1024,23,9)\
  substract_m1n1(1032,23,12) substract_m1n1(1040,23,15)\
  substract_m1n1(1048,23,18) substract_m1n1(1056,23,21)\
  substract_m1n1(1064,23,24) substract_m1n1(1072,23,27)\
  substract_m1n1(1080,23,30) substract_m1n1(1088,23,10)\
  substract_m1n1(1096,23,13) substract_m1n1(1104,23,16)\
  substract_m1n1(1112,23,19) substract_m1n1(1120,23,22)\
  substract_m1n1(1128,23,25) substract_m1n1(1136,23,28)\
  substract_m1n1(1144,23,31)\
  solve_lt_m1n1(1200,26)\
  substract_m1n1(1208,26,29) substract_m1n1(1216,26,9)\
  substract_m1n1(1224,26,12)\
  substract_m1n1(1232,26,15) substract_m1n1(1240,26,18)\
  substract_m1n1(1248,26,21) substract_m1n1(1256,26,24)\
  substract_m1n1(1264,26,27) substract_m1n1(1272,26,30)\
  substract_m1n1(1280,26,10) substract_m1n1(1288,26,13)\
  substract_m1n1(1296,26,16) substract_m1n1(1304,26,19)\
  substract_m1n1(1312,26,22) substract_m1n1(1320,26,25)\
  substract_m1n1(1328,26,28) substract_m1n1(1336,26,31)\
  solve_lt_m1n1(1400,29)\
  substract_m1n1(1408,29,9) substract_m1n1(1416,29,12)\
  substract_m1n1(1424,29,15)\
  substract_m1n1(1432,29,18) substract_m1n1(1440,29,21)\
  substract_m1n1(1448,29,24) substract_m1n1(1456,29,27)\
  substract_m1n1(1464,29,30) substract_m1n1(1472,29,10)\
  substract_m1n1(1480,29,13) substract_m1n1(1488,29,16)\
  substract_m1n1(1496,29,19) substract_m1n1(1504,29,22)\
  substract_m1n1(1512,29,25) substract_m1n1(1520,29,28)\
  substract_m1n1(1528,29,31)\
  solve_lt_m1n1(1600,9)\
  substract_m1n1(1608,9,12) substract_m1n1(1616,9,15)\
  substract_m1n1(1624,9,18)\
  substract_m1n1(1632,9,21) substract_m1n1(1640,9,24)\
  substract_m1n1(1648,9,27) substract_m1n1(1656,9,30)\
  substract_m1n1(1664,9,10) substract_m1n1(1672,9,13)\
  substract_m1n1(1680,9,16) substract_m1n1(1688,9,19)\
  substract_m1n1(1696,9,22) substract_m1n1(1704,9,25)\
  substract_m1n1(1712,9,28) substract_m1n1(1720,9,31)\
  solve_lt_m1n1(1800,12)\
  substract_m1n1(1808,12,15) substract_m1n1(1816,12,18)\
  substract_m1n1(1824,12,21)\
  substract_m1n1(1832,12,24) substract_m1n1(1840,12,27)\
  substract_m1n1(1848,12,30) substract_m1n1(1856,12,10)\
  substract_m1n1(1864,12,13) substract_m1n1(1872,12,16)\
  substract_m1n1(1880,12,19) substract_m1n1(1888,12,22)\
  substract_m1n1(1896,12,25) substract_m1n1(1904,12,28)\
  substract_m1n1(1912,12,31)\
  solve_lt_m1n1(2000,15)\
  substract_m1n1(2008,15,18) substract_m1n1(2016,15,21)\
  substract_m1n1(2024,15,24)\
  substract_m1n1(2032,15,27) substract_m1n1(2040,15,30)\
  substract_m1n1(2048,15,10) substract_m1n1(2056,15,13)\
  substract_m1n1(2064,15,16) substract_m1n1(2072,15,19)\
  substract_m1n1(2080,15,22) substract_m1n1(2088,15,25)\
  substract_m1n1(2096,15,28) substract_m1n1(2104,15,31)\
  solve_lt_m1n1(2200,18)\
  substract_m1n1(2208,18,21) substract_m1n1(2216,18,24)\
  substract_m1n1(2224,18,27)\
  substract_m1n1(2232,18,30) substract_m1n1(2240,18,10)\
  substract_m1n1(2248,18,13) substract_m1n1(2256,18,16)\
  substract_m1n1(2264,18,19) substract_m1n1(2272,18,22)\
  substract_m1n1(2280,18,25) substract_m1n1(2288,18,28)\
  substract_m1n1(2296,18,31)\
  solve_lt_m1n1(2400,21)\
  substract_m1n1(2408,21,24) substract_m1n1(2416,21,27)\
  substract_m1n1(2424,21,30)\
  substract_m1n1(2432,21,10) substract_m1n1(2440,21,13)\
  substract_m1n1(2448,21,16) substract_m1n1(2456,21,19)\
  substract_m1n1(2464,21,22) substract_m1n1(2472,21,25)\
  substract_m1n1(2480,21,28) substract_m1n1(2488,21,31)\
  solve_lt_m1n1(2600,24)\
  substract_m1n1(2608,24,27) substract_m1n1(2616,24,30)\
  substract_m1n1(2624,24,10)\
  substract_m1n1(2632,24,13) substract_m1n1(2640,24,16)\
  substract_m1n1(2648,24,19) substract_m1n1(2656,24,22)\
  substract_m1n1(2664,24,25) substract_m1n1(2672,24,28)\
  substract_m1n1(2680,24,31)\
  solve_lt_m1n1(2800,27)\
  substract_m1n1(2808,27,30) substract_m1n1(2816,27,10)\
  substract_m1n1(2824,27,13)\
  substract_m1n1(2832,27,16) substract_m1n1(2840,27,19)\
  substract_m1n1(2848,27,22) substract_m1n1(2856,27,25)\
  substract_m1n1(2864,27,28) substract_m1n1(2872,27,31)\
  solve_lt_m1n1(3000,30)\
  substract_m1n1(3008,30,10) substract_m1n1(3016,30,13)\
  substract_m1n1(3024,30,16)\
  substract_m1n1(3032,30,19) substract_m1n1(3040,30,22)\
  substract_m1n1(3048,30,25) substract_m1n1(3056,30,28)\
  substract_m1n1(3064,30,31)\
  solve_lt_m1n1(3200,10)\
  substract_m1n1(3208,10,13) substract_m1n1(3216,10,16)\
  substract_m1n1(3224,10,19)\
  substract_m1n1(3232,10,22) substract_m1n1(3240,10,25)\
  substract_m1n1(3248,10,28) substract_m1n1(3256,10,31)\
  solve_lt_m1n1(3400,13)\
  substract_m1n1(3408,13,16) substract_m1n1(3416,13,19)\
  substract_m1n1(3424,13,22)\
  substract_m1n1(3432,13,25) substract_m1n1(3440,13,28)\
  substract_m1n1(3448,13,31)\
  solve_lt_m1n1(3600,16)\
  substract_m1n1(3608,16,19) substract_m1n1(3616,16,22)\
  substract_m1n1(3624,16,25)\
  substract_m1n1(3632,16,28) substract_m1n1(3640,16,31)\
  solve_lt_m1n1(3800,19)\
  substract_m1n1(3808,19,22) substract_m1n1(3816,19,25)\
  substract_m1n1(3824,19,28)\
  substract_m1n1(3832,19,31)\
  solve_lt_m1n1(4000,22)\
  substract_m1n1(4008,22,25) substract_m1n1(4016,22,28)\
  substract_m1n1(4024,22,31)\
  solve_lt_m1n1(4200,25)\
  substract_m1n1(4208,25,28) substract_m1n1(4216,25,31)\
  solve_lt_m1n1(4400,28)\
  substract_m1n1(4408,28,31)\
  solve_lt_m1n1(4600,31)\

#define PREF_TRSM_ADD_C_COL_m24 \
  "prefetcht0 0(%3);addq %4,%3;"

#define LOAD_TRSM_ADD_C_COL_m24(c1,c2,c3) \
  "vaddpd (%3),%%zmm"#c1",%%zmm"#c1";"\
  "vaddpd 64(%3),%%zmm"#c2",%%zmm"#c2";"\
  "vaddpd 128(%3),%%zmm"#c3",%%zmm"#c3";addq %4,%3;"

#define LOAD_TRSM_ADD_C_COL_m8(c1) \
  "vaddpd (%3),%%zmm"#c1",%%zmm"#c1";addq %4,%3;"

#define LOAD_TRSM_ADD_C_COL_m2(c1) \
  "vaddpd (%3),%%xmm"#c1",%%xmm"#c1";addq %4,%3;"

#define SAVE_TRSM_C_COL_m24n2(c1,c2,c3,c4,c5,c6) \
  "prefetcht2 384(%3); vmovups %%zmm"#c1",(%3);prefetcht2 448(%3);vmovups %%zmm"#c2",64(%3);prefetcht2 512(%3);vmovups %%zmm"#c3",128(%3);"\
  "prefetcht2 384(%3,%4,1); vmovups %%zmm"#c4",(%3,%4,1);prefetcht2 448(%3,%4,1);vmovups %%zmm"#c5",64(%3,%4,1);prefetcht2 512(%3,%4,1);vmovups %%zmm"#c6",128(%3,%4,1);leaq (%3,%4,2),%3;"

#define SAVE_TRSM_C_COL_m8n2(c1,c4) \
  "prefetcht2 64(%3); vmovups %%zmm"#c1",(%3);"\
  "prefetcht2 64(%3,%4,1); vmovups %%zmm"#c4",(%3,%4,1);leaq (%3,%4,2),%3;"

#define SAVE_TRSM_C_COL_m2n2(c1,c4) \
  "vmovups %%xmm"#c1",(%3);"\
  "vmovups %%xmm"#c4",(%3,%4,1);leaq (%3,%4,2),%3;"

#define add_c_reorder_m2n8 \
  LOAD_TRSM_ADD_C_COL_m2(8)\
  LOAD_TRSM_ADD_C_COL_m2(9)\
  LOAD_TRSM_ADD_C_COL_m2(10)\
  LOAD_TRSM_ADD_C_COL_m2(11)\
  LOAD_TRSM_ADD_C_COL_m2(12)\
  LOAD_TRSM_ADD_C_COL_m2(13)\
  LOAD_TRSM_ADD_C_COL_m2(14)\
  LOAD_TRSM_ADD_C_COL_m2(15)\
  TRANS_2X8(8,9,10,11,12,13,14,15)\

#define add_c_reorder_m8n8 \
  LOAD_TRSM_ADD_C_COL_m8(8)\
  LOAD_TRSM_ADD_C_COL_m8(11)\
  LOAD_TRSM_ADD_C_COL_m8(14)\
  LOAD_TRSM_ADD_C_COL_m8(17)\
  LOAD_TRSM_ADD_C_COL_m8(20)\
  LOAD_TRSM_ADD_C_COL_m8(23)\
  LOAD_TRSM_ADD_C_COL_m8(26)\
  LOAD_TRSM_ADD_C_COL_m8(29)\
  TRANS_8X8(8,11,14,17,20,23,26,29)\

#define add_c_reorder_m24n8 \
  LOAD_TRSM_ADD_C_COL_m24(8,9,10)\
  LOAD_TRSM_ADD_C_COL_m24(11,12,13)\
  LOAD_TRSM_ADD_C_COL_m24(14,15,16)\
  LOAD_TRSM_ADD_C_COL_m24(17,18,19)\
  LOAD_TRSM_ADD_C_COL_m24(20,21,22)\
  LOAD_TRSM_ADD_C_COL_m24(23,24,25)\
  LOAD_TRSM_ADD_C_COL_m24(26,27,28)\
  LOAD_TRSM_ADD_C_COL_m24(29,30,31)\
  TRANS_8X8(8,11,14,17,20,23,26,29)\
  TRANS_8X8(9,12,15,18,21,24,27,30)\
  TRANS_8X8(10,13,16,19,22,25,28,31)
#define add_c_reorder_m24n4 \
  LOAD_TRSM_ADD_C_COL_m24(8,9,10)\
  LOAD_TRSM_ADD_C_COL_m24(11,12,13)\
  LOAD_TRSM_ADD_C_COL_m24(14,15,16)\
  LOAD_TRSM_ADD_C_COL_m24(17,18,19)\
  init_m24n1_para(20,21,22)\
  init_m24n1_para(23,24,25)\
  init_m24n1_para(26,27,28)\
  init_m24n1_para(29,30,31)\
  TRANS_8X8(8,11,14,17,20,23,26,29)\
  TRANS_8X8(9,12,15,18,21,24,27,30)\
  TRANS_8X8(10,13,16,19,22,25,28,31)
#define SOLVE_LT_m8n8 \
  "movq %2, %3;" add_c_reorder_m8n8\
  solve_lt_m8n8\
  TRANS_4X8_B(8,11,14,17) TRANS_4X8_B(20,23,26,29)\
  TRANS_8X8(8,11,14,17,20,23,26,29)\
  "movq %2, %3; addq $64, %2;" SAVE_TRSM_C_COL_m8n2(8,11)\
  SAVE_TRSM_C_COL_m8n2(14,17)\
  SAVE_TRSM_C_COL_m8n2(20,23)\
  SAVE_TRSM_C_COL_m8n2(26,29)

#define SOLVE_LT_m2n8 \
  "movq %2, %3;" add_c_reorder_m2n8\
  solve_lt_m2n8\
  TRANS_1X8_B(8,10,12,14) TRANS_1X8_B(9,11,13,15)\
  TRANS_2X8(8,9,10,11,12,13,14,15)\
  "movq %2, %3; addq $16, %2;" SAVE_TRSM_C_COL_m2n2(8,9)\
  SAVE_TRSM_C_COL_m2n2(10,11)\
  SAVE_TRSM_C_COL_m2n2(12,13)\
  SAVE_TRSM_C_COL_m2n2(14,15)

#define SOLVE_LT_m24n8 \
  "movq %2, %3;" add_c_reorder_m24n8\
  solve_lt_m24n8\
  TRANS_4X8_B(8,11,14,17) TRANS_4X8_B(20,23,26,29)\
  TRANS_4X8_B(9,12,15,18) TRANS_4X8_B(21,24,27,30)\
  TRANS_4X8_B(10,13,16,19) TRANS_4X8_B(22,25,28,31)\
  TRANS_8X8(8,11,14,17,20,23,26,29)\
  TRANS_8X8(9,12,15,18,21,24,27,30)\
  TRANS_8X8(10,13,16,19,22,25,28,31)\
  "movq %2, %3; addq $192, %2;" SAVE_TRSM_C_COL_m24n2(8,9,10,11,12,13)\
  SAVE_TRSM_C_COL_m24n2(14,15,16,17,18,19)\
  SAVE_TRSM_C_COL_m24n2(20,21,22,23,24,25)\
  SAVE_TRSM_C_COL_m24n2(26,27,28,29,30,31)
#define SOLVE_LT_m24n4 \
  "movq %2, %3;" add_c_reorder_m24n4\
  solve_lt_m24n8\
  TRANS_4X8_B(8,11,14,17) TRANS_4X8_B(20,23,26,29)\
  TRANS_4X8_B(9,12,15,18) TRANS_4X8_B(21,24,27,30)\
  TRANS_4X8_B(10,13,16,19) TRANS_4X8_B(22,25,28,31)\
  TRANS_8X8(8,11,14,17,20,23,26,29)\
  TRANS_8X8(9,12,15,18,21,24,27,30)\
  TRANS_8X8(10,13,16,19,22,25,28,31)\
  "movq %2, %3; addq $192, %2;" SAVE_TRSM_C_COL_m24n2(8,9,10,11,12,13)\
  SAVE_TRSM_C_COL_m24n2(14,15,16,17,18,19)\
  SAVE_TRSM_C_COL_m24n2(20,21,22,23,24,25)\
  SAVE_TRSM_C_COL_m24n2(26,27,28,29,30,31)
#define COMPUTE_trsm_n8 {\
  __asm__ __volatile__(\
    "movq %0,%%r15; movq %1,%%r14; movq %8,%%r13; movq %7,%%r12; salq $3,%%r12; movq %9,%%r11;"\
    "cmpq $24,%%r11; jb 8772f;"\
    "8771:\n\t"\
    GEMM_LT_m24n8 SOLVE_LT_m24n8 "subq $24,%%r11; cmpq $24,%%r11; jnb 8771b;"\
    "8772:\n\t"\
    "cmpq $8,%%r11; jb 8774f;"\
    "8773:\n\t"\
    GEMM_LT_m8n8 SOLVE_LT_m8n8 "subq $8,%%r11; cmpq $8,%%r11; jnb 8773b;"\
    "8774:\n\t"\
    "cmpq $2,%%r11; jb 8776f;"\
    "8775:\n\t"\
    GEMM_LT_m2n8 SOLVE_LT_m2n8 "subq $2,%%r11; cmpq $2,%%r11; jnb 8775b;"\
    "8776:\n\t"\
    "movq %%r15,%0; movq %%r14,%1;"\
  :"+r"(a_ptr),"+r"(b_ptr),"+r"(c_ptr),"+r"(c_tmp),"+r"(ldc_bytes),"+r"(k_cnt),"+r"(b_tmp):"m"(K),"m"(OFF),"m"(M)\
  :"r10","r11","r12","r13","r14","r15","cc","memory",\
  "zmm0","zmm1","zmm2","zmm3","zmm4","zmm5","zmm6","zmm7","zmm8","zmm9","zmm10","zmm11","zmm12","zmm13","zmm14","zmm15",\
    "zmm16","zmm17","zmm18","zmm19","zmm20","zmm21","zmm22","zmm23","zmm24","zmm25","zmm26","zmm27","zmm28","zmm29","zmm30","zmm31");\
  a_ptr -= M * K; b_ptr += 8 * K; c_ptr += ldc * 8 - M;\
}

// #define COMPUTE_trsm_n4 {\
//   __asm__ __volatile__(\
//     "movq %0,%%r15; movq %1,%%r14; movq %8,%%r13; movq %7,%%r12; salq $3,%%r12; movq %9,%%r11;"\
//     "cmpq $24,%%r11; jb 4772f;"\
//     "4771:\n\t"\
//     GEMM_LT_m24n4 SOLVE_LT_m24n4 "subq $24,%%r11; cmpq $24,%%r11; jnb 4771b;"\
//     "4772:\n\t"\
//     "cmpq $8,%%r11; jb 4774f;"\
//     "4773:\n\t"\
//     GEMM_LT_m8n4 SOLVE_LT_m8n4 "subq $8,%%r11; cmpq $8,%%r11; jnb 4773b;"\
//     "4774:\n\t"\
//     "cmpq $2,%%r11; jb 4776f;"\
//     "4775:\n\t"\
//     GEMM_LT_m2n4 SOLVE_LT_m2n4 "subq $2,%%r11; cmpq $2,%%r11; jnb 4775b;"\
//     "4776:\n\t"\
//     "movq %%r15,%0; movq %%r14,%1;"\
//   :"+r"(a_ptr),"+r"(b_ptr),"+r"(c_ptr),"+r"(c_tmp),"+r"(ldc_bytes),"+r"(k_cnt),"+r"(b_tmp):"m"(K),"m"(OFF),"m"(M)\
//   :"r10","r11","r12","r13","r14","r15","cc","memory",\
//   "zmm0","zmm1","zmm2","zmm3","zmm4","zmm5","zmm6","zmm7","zmm8","zmm9","zmm10","zmm11","zmm12","zmm13","zmm14","zmm15",\
//     "zmm16","zmm17","zmm18","zmm19","zmm20","zmm21","zmm22","zmm23","zmm24","zmm25","zmm26","zmm27","zmm28","zmm29","zmm30","zmm31");\
//   a_ptr -= M * K; b_ptr += 4 * K; c_ptr += ldc * 4 - M;\
// }



void macro_kernel_trsm(int m, int n, int k, double *a, double *b, double *c, int ldc, int offset){
  if (m==0||n==0||k==0) return;
  double *a_ptr = a, *b_ptr = b, *c_ptr = c, *c_tmp = c, *b_tmp;
  double one[8] = {1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0};
  double zero[8] = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
  int64_t ldc_bytes = (int64_t)ldc * sizeof(double), K = (int64_t)k, M = (int64_t)m, OFF = (int64_t)offset, k_cnt = 0;
  // int n_count = n;
  // for(;n_count>11;n_count-=12) COMPUTE(12)
  // for(;n_count>7;n_count-=8) COMPUTE(8)
  // for(;n_count>3;n_count-=4) COMPUTE(4)
  // for(;n_count>1;n_count-=2) { COMPUTE_EDGE_1_nchunk(m,2,a_ptr,b_ptr,c_ptr,ldc,k,offset); b_ptr += 2*k; c_ptr += ldc*2;}
  // if(n_count>0) COMPUTE_EDGE_1_nchunk(m,1,a_ptr,b_ptr,c_ptr,ldc,k,offset);
  // return 0;
  int n_count,n_count_sub;
  // printf("INNER: m = %d, n = %d, k = %d\n",m,n,k);
  // printf("m= %d, n=%d, k = %d\n",m,n,k);
  
  for (n_count_sub=n,n_count=0;n_count_sub>7;n_count_sub-=8,n_count+=8){
      //call the m layer with n=8;
      // printf("!!! **** offset = %d\n",offset);
      COMPUTE_trsm_n8
  }
  for (;n_count_sub>3;n_count_sub-=4,n_count+=4){
      //call the m layer with n=4
      //COMPUTE_trsm_n4
  }
  for (;n_count_sub>1;n_count_sub-=2,n_count+=2){
    //call the m layer with n=2
  }
  for (;n_count_sub>0;n_count_sub-=1,n_count+=1){
    //call the m layer with n=1
  }
}


void ftblas_dtrsm_ori(\
    int M, \
    int N, \
    double alpha, \
    double *A, \
    int LDA, \
    double *B, \
    int LDB)\
{
    int i,j,k;
#ifdef TIMER
    double ts=0.,tm=0.;
    double as = 0., am = 0.;
    double t_tot=0.,t_copy_a=0.,t_copy_b=0.,t_scale_c=0., t0, t1;
#endif
    int K=M;
    if (alpha == 0.||K==0) return;
    int M4,N8=N&-8,K4;
    
    double *b_buffer = (double *)aligned_alloc(4096,K_BLOCKING*N_BLOCKING*sizeof(double));
    double *a_buffer = (double *)aligned_alloc(4096,K_BLOCKING*M_BLOCKING*sizeof(double));
    int second_m_count,second_n_count,second_m_inc,second_n_inc;
    int m_count,n_count,k_count;
    int m_inc,n_inc,k_inc;
    for (n_count=0;n_count<N;n_count+=n_inc){
        n_inc=(N-n_count>N_BLOCKING)?N_BLOCKING:N-n_count;
        for (k_count=0;k_count<K;k_count+=k_inc){
            k_inc=(K-k_count>K_BLOCKING)?K_BLOCKING:K-k_count;
            m_inc=k_inc>M_BLOCKING?M_BLOCKING:k_inc;m_count=k_count;
#ifdef TIMER
            t0=get_sec();
#endif
            packing_a_24x8_trsm(A+m_count+k_count*LDA,a_buffer,LDA,m_inc,k_inc,m_count-k_count);
#ifdef TIMER
            t1=get_sec();
            t_copy_a+=(t1-t0);
#endif
            for (second_n_count=n_count;second_n_count<n_count+n_inc;second_n_count+=second_n_inc){
              second_n_inc=(n_count+n_inc-second_n_count>8)?8:n_count+n_inc-second_n_count;
#ifdef TIMER
              t0=get_sec();
#endif
              packing_b_24x8(B+k_count+second_n_count*LDB,b_buffer+(second_n_count-n_count)*k_inc,LDB,k_inc,second_n_inc);
#ifdef TIMER
              t1=get_sec();
              t_copy_b+=(t1-t0);
              t0=get_sec();
#endif
              macro_kernel_trsm(m_inc,second_n_inc,k_inc,a_buffer,b_buffer+(second_n_count-n_count)*k_inc,&B(m_count,second_n_count),LDB,m_count-k_count);
#ifdef TIMER
              t1=get_sec();
              ts += (t1 - t0); as += 1. * m_inc*second_n_inc*k_inc;
#endif
            }
            m_count+=m_inc;m_inc=k_inc-m_inc;
            if (m_inc>0){
#ifdef TIMER
              t0=get_sec();
#endif
              packing_a_24x8_trsm(A+m_count+k_count*LDA,a_buffer,LDA,m_inc,k_inc,m_count-k_count);
#ifdef TIMER
              t1=get_sec();
              t_copy_a+=(t1-t0);
              t0=get_sec();
#endif
              macro_kernel_trsm(m_inc,n_inc,k_inc,a_buffer,b_buffer,&B(m_count,n_count),LDB,m_count-k_count);
#ifdef TIMER
              t1=get_sec();
              ts += (t1 - t0); as += 1. * m_inc*n_inc*k_inc;
#endif
            }
            for (m_count=k_count+k_inc;m_count<M;m_count+=m_inc){
              m_inc=(M-m_count>M_BLOCKING)?M_BLOCKING:M-m_count;
#ifdef TIMER
              t0=get_sec();
#endif
              packing_a_lower(A+m_count+k_count*LDA,a_buffer,LDA,m_inc,k_inc);
#ifdef TIMER
              t1=get_sec();
              t_copy_a+=(t1-t0);
              t0=get_sec();
#endif
              macro_kernel_gemm(a_buffer,b_buffer,m_inc,n_inc,k_inc,&B(m_count,n_count),LDB);
#ifdef TIMER
              t1=get_sec();
              tm += (t1 - t0); am += 2. * m_inc*n_inc*k_inc;
#endif
            }
        }
    }
#ifdef TIMER
    printf("Time in TRSM: %f, perf=%f GFLOPS\n", ts,1e-9*as/ts);
    printf("Time in GEMM: %f, perf=%f GFLOPS\n", tm,1e-9*am/tm);
    printf("Time in copy A: %f, perf = %f GFLOPS\n", t_copy_a, 1.*1e-9*M*K/t_copy_a);
    printf("Time in copy B: %f, perf = %f GFLOPS\n", t_copy_b, 2.*1e-9*N*K/t_copy_b);
    printf("Total: %f, perf = %f GFLOPS\n", ts+tm+t_copy_b+t_copy_a, 1.*1e-9*M*N*K/(ts+tm+t_copy_b+t_copy_a));
#endif
    free(a_buffer);free(b_buffer);
}