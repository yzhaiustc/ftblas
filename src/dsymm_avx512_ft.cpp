#include <immintrin.h>

#include "../include/ftblas.h"

#define A(i,j) A[(i)+(j)*lda]
//#define TIMING 1
//#define PRINT 1
#define KERNEL_h_k1m24n1 \
  "vmovaps (%0),%%zmm1; vmovaps 64(%0),%%zmm2; vmovaps 128(%0),%%zmm3; addq $192,%0;"\
  "vbroadcastsd (%1),%%zmm4; vfmadd231pd %%zmm1,%%zmm4,%%zmm8; vfmadd231pd %%zmm2,%%zmm4,%%zmm9; vfmadd231pd %%zmm3,%%zmm4,%%zmm10;"
#define KERNEL_k1m24n1 KERNEL_h_k1m24n1 "addq $8,%1;"
#define KERNEL_h_k1m24n2 KERNEL_h_k1m24n1\
  "vbroadcastsd 8(%1),%%zmm5; vfmadd231pd %%zmm1,%%zmm5,%%zmm11; vfmadd231pd %%zmm2,%%zmm5,%%zmm12; vfmadd231pd %%zmm3,%%zmm5,%%zmm13;"
#define KERNEL_k1m24n2 KERNEL_h_k1m24n2 "addq $16,%1;"
#define unit_acc_m24n2(c1_no,c2_no,c3_no,c4_no,c5_no,c6_no,...)\
  "vbroadcastsd ("#__VA_ARGS__"),%%zmm4; vfmadd231pd %%zmm1,%%zmm4,%%zmm"#c1_no"; vfmadd231pd %%zmm2,%%zmm4,%%zmm"#c2_no"; vfmadd231pd %%zmm3,%%zmm4,%%zmm"#c3_no";"\
  "vbroadcastsd 8("#__VA_ARGS__"),%%zmm5; vfmadd231pd %%zmm1,%%zmm5,%%zmm"#c4_no"; vfmadd231pd %%zmm2,%%zmm5,%%zmm"#c5_no"; vfmadd231pd %%zmm3,%%zmm5,%%zmm"#c6_no";"
#define KERNEL_h_k1m24n4 KERNEL_h_k1m24n2 "prefetcht0 384(%0);" unit_acc_m24n2(14,15,16,17,18,19,%1,%%r12,1)
#define KERNEL_k1m24n4 KERNEL_h_k1m24n4 "addq $16,%1;"
#define KERNEL_k1m24n6 KERNEL_h_k1m24n4 unit_acc_m24n2(20,21,22,23,24,25,%1,%%r12,2) "addq $16,%1;"
#define KERNEL_h_k1m24n8 KERNEL_k1m24n6 "prefetcht0 448(%0);" unit_acc_m24n2(26,27,28,29,30,31,%%r15)
#define KERNEL_k1m24n8 KERNEL_h_k1m24n8 "addq $16,%%r15;"
#define unit_init_3zmm(c1_no,c2_no,c3_no) "vpxorq %%zmm"#c1_no",%%zmm"#c1_no",%%zmm"#c1_no"; vpxorq %%zmm"#c2_no",%%zmm"#c2_no",%%zmm"#c2_no"; vpxorq %%zmm"#c3_no",%%zmm"#c3_no",%%zmm"#c3_no";"
#define unit_init_6zmm(c1_no,c2_no,c3_no,c4_no,c5_no,c6_no) unit_init_3zmm(c1_no,c2_no,c3_no) unit_init_3zmm(c4_no,c5_no,c6_no)
#define INIT_m24n1 unit_init_3zmm(8,9,10)
#define INIT_m24n2 unit_init_6zmm(8,9,10,11,12,13)
#define INIT_m24n4 INIT_m24n2 unit_init_6zmm(14,15,16,17,18,19)
#define INIT_m24n6 INIT_m24n4 unit_init_6zmm(20,21,22,23,24,25)
#define INIT_m24n8 INIT_m24n6 unit_init_6zmm(26,27,28,29,30,31)
#define CLEAR_reg_n8 unit_init_6zmm(0,1,2,3,4,5) "vpxorq %%zmm6,%%zmm6,%%zmm6;vpxorq %%zmm7,%%zmm7,%%zmm7;"
#define add_row_chksm_m8n8(c1,c2,c3,c4,c5,c6,c7,c8)\
  "vaddpd %%zmm"#c1",%%zmm0,%%zmm0;%%zmm"#c2",%%zmm1,%%zmm1;vaddpd %%zmm"#c3",%%zmm2,%%zmm2;%%zmm"#c4",%%zmm3,%%zmm3;"\
  "vaddpd %%zmm"#c5",%%zmm4,%%zmm4;%%zmm"#c6",%%zmm5,%%zmm5;vaddpd %%zmm"#c7",%%zmm6,%%zmm6;%%zmm"#c8",%%zmm7,%%zmm7;"

#define save_init_m24 "movq %2,%3; addq $192,%2;" unit_init_3zmm(0,1,2) unit_init_3zmm(3,4,5) "vpxorq %%zmm6,%%zmm6,%%zmm6;vpxorq %%zmm7,%%zmm7,%%zmm7;"
#define SAVE_m24n1\
  "vaddpd (%2),%%zmm8,%%zmm8; vmovupd %%zmm8,(%2); vaddpd 64(%2),%%zmm9,%%zmm9; vmovupd %%zmm9,64(%2); vaddpd 128(%2),%%zmm10,%%zmm10; vmovupd %%zmm10,128(%2); addq $192,%2;"\
  "vaddpd %%zmm0, %%zmm8, %%zmm0;vaddpd %%zmm0, %%zmm9, %%zmm0;vaddpd %%zmm0, %%zmm10, %%zmm0;"
#define unit_save_m24n2(c1_no,c2_no,c3_no,c4_no,c5_no,c6_no,z1,z2)\
  "vaddpd (%3),%%zmm"#c1_no",%%zmm"#c1_no"; vmovupd %%zmm"#c1_no",(%3); vaddpd 64(%3),%%zmm"#c2_no",%%zmm"#c2_no"; vmovupd %%zmm"#c2_no",64(%3); vaddpd 128(%3),%%zmm"#c3_no",%%zmm"#c3_no"; vmovupd %%zmm"#c3_no",128(%3);"\
  "vaddpd (%3,%4,1),%%zmm"#c4_no",%%zmm"#c4_no"; vmovupd %%zmm"#c4_no",(%3,%4,1); vaddpd 64(%3,%4,1),%%zmm"#c5_no",%%zmm"#c5_no"; vmovupd %%zmm"#c5_no",64(%3,%4,1); vaddpd 128(%3,%4,1),%%zmm"#c6_no",%%zmm"#c6_no"; vmovupd %%zmm"#c6_no",128(%3,%4,1); leaq (%3,%4,2),%3;"\
  "vaddpd %%zmm"#c1_no", %%zmm"#z1",%%zmm"#z1";vaddpd %%zmm"#c2_no", %%zmm"#z1",%%zmm"#z1";vaddpd %%zmm"#c3_no", %%zmm"#z1",%%zmm"#z1";"\
  "vaddpd %%zmm"#c4_no", %%zmm"#z2",%%zmm"#z2";vaddpd %%zmm"#c5_no", %%zmm"#z2",%%zmm"#z2";vaddpd %%zmm"#c6_no", %%zmm"#z2",%%zmm"#z2";"
#define SAVE_m24n2 save_init_m24 unit_save_m24n2(8,9,10,11,12,13,0,1)
#define SAVE_m24n4 SAVE_m24n2 unit_save_m24n2(14,15,16,17,18,19,2,3)
#define SAVE_m24n6 SAVE_m24n4 unit_save_m24n2(20,21,22,23,24,25,4,5)
#define SAVE_m24n8 CLEAR_reg_n8 SAVE_m24n6 unit_save_m24n2(26,27,28,29,30,31,6,7)

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
  "vaddpd %%zmm8,%%zmm12,%%zmm8; vshuff64x2 $0xd8,%%zmm8,%%zmm8,%%zmm8;vaddpd (%8),%%zmm8,%%zmm8;vmovaps %%zmm8,(%8);"

#define add_chksum_m24_1col_0(c4_no,c5_no,c6_no)\
  "vaddpd %%zmm"#c4_no", %%zmm8, %%zmm8; vaddpd %%zmm"#c5_no", %%zmm9, %%zmm9; vaddpd %%zmm"#c6_no", %%zmm10, %%zmm10;"
#define add_chksum_m24_1col(c1_no,c2_no,c3_no,c4_no,c5_no,c6_no)\
  "vaddpd %%zmm"#c1_no", %%zmm8, %%zmm8; vaddpd %%zmm"#c2_no", %%zmm9, %%zmm9; vaddpd %%zmm"#c3_no", %%zmm10, %%zmm10;"\
  "vaddpd %%zmm"#c4_no", %%zmm8, %%zmm8; vaddpd %%zmm"#c5_no", %%zmm9, %%zmm9; vaddpd %%zmm"#c6_no", %%zmm10, %%zmm10;"
#define save_chksum_m24_1col(c1_no,c2_no,c3_no)\
  "vaddpd (%7), %%zmm"#c1_no", %%zmm"#c1_no"; vaddpd 64(%7), %%zmm"#c2_no", %%zmm"#c2_no"; vaddpd 128(%7), %%zmm"#c3_no", %%zmm"#c3_no";"\
  "vmovaps %%zmm"#c1_no", (%7); vmovaps %%zmm"#c2_no", 64(%7); vmovaps %%zmm"#c3_no", 128(%7); addq $192, %7;"
#define SAVE_CHKSUM_m24n1 save_chksum_m24_1col(8,9,10)
#define SAVE_CHKSUM_m24n2_h add_chksum_m24_1col_0(11,12,13)
#define SAVE_CHKSUM_m24n2 SAVE_CHKSUM_m24n2_h save_chksum_m24_1col(8,9,10)
#define SAVE_CHKSUM_m24n4_h SAVE_CHKSUM_m24n2_h add_chksum_m24_1col(14,15,16,17,18,19)
#define SAVE_CHKSUM_m24n4 SAVE_CHKSUM_m24n4_h save_chksum_m24_1col(8,9,10)
#define SAVE_CHKSUM_m24n6_h SAVE_CHKSUM_m24n4_h add_chksum_m24_1col(20,21,22,23,24,25)
#define SAVE_CHKSUM_m24n6 SAVE_CHKSUM_m24n6_h save_chksum_m24_1col(8,9,10)
#define SAVE_CHKSUM_m24n8_h SAVE_CHKSUM_m24n6_h add_chksum_m24_1col(26,27,28,29,30,31)
#define SAVE_CHKSUM_m24n8 SAVE_CHKSUM_m24n8_h save_chksum_m24_1col(8,9,10) trans_and_save_8X8

#define KERNEL_h_k1m16n1 \
  "vmovaps (%0),%%zmm1; vmovaps 64(%0),%%zmm2; addq $128,%0;"\
  "vbroadcastsd (%1),%%zmm3; vfmadd231pd %%zmm1,%%zmm3,%%zmm8; vfmadd231pd %%zmm2,%%zmm3,%%zmm9;"
#define KERNEL_k1m16n1 KERNEL_h_k1m16n1 "addq $8,%1;"
#define KERNEL_h_k1m16n2 KERNEL_h_k1m16n1\
  "vbroadcastsd 8(%1),%%zmm4; vfmadd231pd %%zmm1,%%zmm4,%%zmm10; vfmadd231pd %%zmm2,%%zmm4,%%zmm11;"
#define KERNEL_k1m16n2 KERNEL_h_k1m16n2 "addq $16,%1;"
#define unit_acc_m16n2(c1_no,c2_no,c3_no,c4_no,...)\
  "vbroadcastsd ("#__VA_ARGS__"),%%zmm3; vfmadd231pd %%zmm1,%%zmm3,%%zmm"#c1_no"; vfmadd231pd %%zmm2,%%zmm3,%%zmm"#c2_no";"\
  "vbroadcastsd 8("#__VA_ARGS__"),%%zmm4; vfmadd231pd %%zmm1,%%zmm4,%%zmm"#c3_no"; vfmadd231pd %%zmm2,%%zmm4,%%zmm"#c4_no";"
#define KERNEL_h_k1m16n4 KERNEL_h_k1m16n2 "prefetcht0 384(%0);" unit_acc_m16n2(12,13,14,15,%1,%%r12,1)
#define KERNEL_k1m16n4 KERNEL_h_k1m16n4 "addq $16,%1;"
#define KERNEL_k1m16n6 KERNEL_h_k1m16n4 unit_acc_m16n2(16,17,18,19,%1,%%r12,2) "addq $16,%1;"
#define KERNEL_h_k1m16n8 KERNEL_k1m16n6 "prefetcht0 448(%0);" unit_acc_m16n2(20,21,22,23,%%r15)
#define KERNEL_k1m16n8 KERNEL_h_k1m16n8 "addq $16,%%r15;"
#define save_init_m16 "movq %2,%3; addq $128,%2;"
#define SAVE_m16n1 "vaddpd (%2),%%zmm8,%%zmm8; vmovupd %%zmm8,(%2); vaddpd 64(%2),%%zmm9,%%zmm9; vmovupd %%zmm9,64(%2); addq $128,%2;"

#define unit_save_m16n2(c1_no,c2_no,c3_no,c4_no,z1,z2)\
  "vaddpd (%3),%%zmm"#c1_no",%%zmm"#c1_no"; vmovupd %%zmm"#c1_no",(%3); vaddpd 64(%3),%%zmm"#c2_no",%%zmm"#c2_no"; vmovupd %%zmm"#c2_no",64(%3);"\
  "vaddpd (%3,%4,1),%%zmm"#c3_no",%%zmm"#c3_no"; vmovupd %%zmm"#c3_no",(%3,%4,1); vaddpd 64(%3,%4,1),%%zmm"#c4_no",%%zmm"#c4_no"; vmovupd %%zmm"#c4_no",64(%3,%4,1); leaq (%3,%4,2),%3;"\
  "vaddpd %%zmm"#c1_no", %%zmm"#z1",%%zmm"#z1";vaddpd %%zmm"#c2_no", %%zmm"#z1",%%zmm"#z1";"\
  "vaddpd %%zmm"#c3_no", %%zmm"#z2",%%zmm"#z2";vaddpd %%zmm"#c4_no", %%zmm"#z2",%%zmm"#z2";"

// #define unit_save_m16n2(c1_no,c2_no,c3_no,c4_no)\
//   "vaddpd (%3),%%zmm"#c1_no",%%zmm"#c1_no"; vmovupd %%zmm"#c1_no",(%3); vaddpd 64(%3),%%zmm"#c2_no",%%zmm"#c2_no"; vmovupd %%zmm"#c2_no",64(%3);"\
//   "vaddpd (%3,%4,1),%%zmm"#c3_no",%%zmm"#c3_no"; vmovupd %%zmm"#c3_no",(%3,%4,1); vaddpd 64(%3,%4,1),%%zmm"#c4_no",%%zmm"#c4_no"; vmovupd %%zmm"#c4_no",64(%3,%4,1); leaq (%3,%4,2),%3;"
#define SAVE_m16n2 save_init_m16 unit_save_m16n2(8,9,10,11,0,1)
#define SAVE_m16n4 SAVE_m16n2 unit_save_m16n2(12,13,14,15,2,3)
#define SAVE_m16n6 SAVE_m16n4 unit_save_m16n2(16,17,18,19,4,5)
#define SAVE_m16n8 CLEAR_reg_n8 SAVE_m16n6 unit_save_m16n2(20,21,22,23,6,7)
#define unit_init_2zmm(c1_no,c2_no) "vpxorq %%zmm"#c1_no",%%zmm"#c1_no",%%zmm"#c1_no"; vpxorq %%zmm"#c2_no",%%zmm"#c2_no",%%zmm"#c2_no";"
#define unit_init_4zmm(c1_no,c2_no,c3_no,c4_no) unit_init_2zmm(c1_no,c2_no) unit_init_2zmm(c3_no,c4_no)
#define INIT_m16n1 unit_init_2zmm(8,9)
#define INIT_m16n2 unit_init_4zmm(8,9,10,11)
#define INIT_m16n4 INIT_m16n2 unit_init_4zmm(12,13,14,15)
#define INIT_m16n6 INIT_m16n4 unit_init_4zmm(16,17,18,19)
#define INIT_m16n8 INIT_m16n6 unit_init_4zmm(20,21,22,23)


#define add_chksum_m16_1col(c1_no,c2_no,c3_no,c4_no)\
  "vaddpd %%zmm"#c1_no", %%zmm"#c3_no", %%zmm"#c1_no"; vaddpd %%zmm"#c2_no", %%zmm"#c4_no", %%zmm"#c2_no";"
#define save_chksum_m16_1col(c1_no,c2_no)\
  "vaddpd (%7), %%zmm"#c1_no", %%zmm"#c1_no"; vaddpd 64(%7), %%zmm"#c2_no", %%zmm"#c2_no";"\
  "vmovaps %%zmm"#c1_no", (%7); vmovaps %%zmm"#c2_no", 64(%7); addq $128, %7;"
#define SAVE_CHKSUM_m16n1 save_chksum_m16_1col(8,9)
#define SAVE_CHKSUM_m16n2_h add_chksum_m16_1col(8,9,10,11)
#define SAVE_CHKSUM_m16n2 SAVE_CHKSUM_m16n2_h save_chksum_m16_1col(8,9)
#define SAVE_CHKSUM_m16n4_h SAVE_CHKSUM_m16n2_h add_chksum_m16_1col(12,13,14,15) add_chksum_m16_1col(8,9,12,13)
#define SAVE_CHKSUM_m16n4 SAVE_CHKSUM_m16n4_h save_chksum_m16_1col(8,9)
#define SAVE_CHKSUM_m16n6_h SAVE_CHKSUM_m16n4_h add_chksum_m16_1col(16,17,18,19) add_chksum_m16_1col(8,9,16,17)
#define SAVE_CHKSUM_m16n6 SAVE_CHKSUM_m16n6_h save_chksum_m16_1col(8,9)
#define SAVE_CHKSUM_m16n8_h SAVE_CHKSUM_m16n6_h add_chksum_m16_1col(20,21,22,23) add_chksum_m16_1col(8,9,20,21)
#define SAVE_CHKSUM_m16n8 SAVE_CHKSUM_m16n8_h save_chksum_m16_1col(8,9) trans_and_save_8X8


#define KERNEL_k1m8n1 \
  "vbroadcastsd (%1),%%zmm1; addq $8,%1;"\
  "vfmadd231pd (%0),%%zmm1,%%zmm8; addq $64,%0;"
//#define unit_acc_m8n2(c1_no,c2_no,...)\
//  "vbroadcastf32x4 ("#__VA_ARGS__"),%%zmm3; vfmadd231pd %%zmm1,%%zmm3,%%zmm"#c1_no"; vfmadd231pd %%zmm2,%%zmm3,%%zmm"#c2_no";"
#define unit_acc_m8n2(c1_no,c2_no,...)\
  "vbroadcastsd ("#__VA_ARGS__"),%%zmm3; vfmadd231pd %%zmm1,%%zmm3,%%zmm"#c1_no";vbroadcastsd 8("#__VA_ARGS__"),%%zmm4; vfmadd231pd %%zmm1,%%zmm4,%%zmm"#c2_no";"
#define KERNEL_h_k1m8n2 \
  "vmovaps (%0),%%zmm1; addq $64,%0;" unit_acc_m8n2(8,9,%1)
#define KERNEL_k1m8n2 KERNEL_h_k1m8n2 "addq $16,%1;"
#define KERNEL_h_k1m8n4 KERNEL_h_k1m8n2 unit_acc_m8n2(10,11,%1,%%r12,1)
#define KERNEL_k1m8n4 KERNEL_h_k1m8n4 "addq $16,%1;"
#define KERNEL_k1m8n6 KERNEL_h_k1m8n4 unit_acc_m8n2(12,13,%1,%%r12,2) "addq $16,%1;"
#define KERNEL_h_k1m8n8 KERNEL_k1m8n6 unit_acc_m8n2(14,15,%%r15)
#define KERNEL_k1m8n8 KERNEL_h_k1m8n8 "addq $16,%%r15;"
#define save_init_m8 "movq %2,%3; addq $64,%2;"
#define SAVE_m8n1 "vaddpd (%2),%%zmm8,%%zmm8; vaddpd %%zmm8, %%zmm0, %%zmm0;vmovupd %%zmm8,(%2); addq $64,%2;"

#define unit_save_m8n2(c1_no,c2_no,z1,z2)\
  "vaddpd (%3),%%zmm"#c1_no",%%zmm"#c1_no"; vmovups %%zmm"#c1_no",(%3);"\
  "vaddpd (%3,%4,1),%%zmm"#c2_no",%%zmm"#c2_no"; vmovups %%zmm"#c2_no",(%3,%4,1);leaq (%3,%4,2),%3;"\
  "vaddpd %%zmm"#c1_no", %%zmm"#z1",%%zmm"#z1";"\
  "vaddpd %%zmm"#c2_no", %%zmm"#z2",%%zmm"#z2";"

// #define unit_save_m8n2(c1_no,c2_no)\
//   "vaddpd (%3),%%zmm"#c1_no",%%zmm"#c1_no"; vmovups %%zmm"#c1_no",(%3);"\
//   "vaddpd (%3,%4,1),%%zmm"#c2_no",%%zmm"#c2_no"; vmovups %%zmm"#c2_no",(%3,%4,1);leaq (%3,%4,2),%3;"
#define SAVE_m8n2 save_init_m8 unit_save_m8n2(8,9,0,1)
#define SAVE_m8n4 SAVE_m8n2 unit_save_m8n2(10,11,2,3)
#define SAVE_m8n6 SAVE_m8n4 unit_save_m8n2(12,13,4,5)
#define SAVE_m8n8 CLEAR_reg_n8 SAVE_m8n6 unit_save_m8n2(14,15,6,7) trans_and_save_8X8
#define INIT_m8n1 "vpxorq %%zmm8,%%zmm8,%%zmm8;"
#define INIT_m8n2 unit_init_2zmm(8,9)
#define INIT_m8n4 INIT_m8n2 unit_init_2zmm(10,11)
#define INIT_m8n6 INIT_m8n4 unit_init_2zmm(12,13)
#define INIT_m8n8 INIT_m8n6 unit_init_2zmm(14,15)


#define add_chksum_m8_1col(c1_no,c2_no)\
  "vaddpd %%zmm"#c1_no", %%zmm"#c2_no", %%zmm"#c1_no";"
#define save_chksum_m8_1col\
  "vaddpd (%7), %%zmm8, %%zmm8;vmovaps %%zmm8, (%7); addq $64, %7;"
#define SAVE_CHKSUM_m8n1 save_chksum_m8_1col
#define SAVE_CHKSUM_m8n2_h add_chksum_m8_1col(8,9)
#define SAVE_CHKSUM_m8n2 SAVE_CHKSUM_m8n2_h save_chksum_m8_1col
#define SAVE_CHKSUM_m8n4_h SAVE_CHKSUM_m8n2_h add_chksum_m8_1col(10,11) add_chksum_m8_1col(8,10)
#define SAVE_CHKSUM_m8n4 SAVE_CHKSUM_m8n4_h save_chksum_m8_1col
#define SAVE_CHKSUM_m8n6_h SAVE_CHKSUM_m8n4_h add_chksum_m8_1col(12,13) add_chksum_m8_1col(8,12)
#define SAVE_CHKSUM_m8n6 SAVE_CHKSUM_m8n6_h save_chksum_m8_1col
#define SAVE_CHKSUM_m8n8_h SAVE_CHKSUM_m8n6_h add_chksum_m8_1col(14,15) add_chksum_m8_1col(8,14)
#define SAVE_CHKSUM_m8n8 SAVE_CHKSUM_m8n8_h save_chksum_m8_1col


#define KERNEL_k1m4n1 \
  "vbroadcastsd (%1),%%ymm1; addq $8,%1;"\
  "vfmadd231pd (%0),%%ymm1,%%ymm4; addq $32,%0;"
//#define unit_acc_m4n2(c1_no,c2_no,...)\
//  "vbroadcastf128 ("#__VA_ARGS__"),%%ymm3; vfmadd231pd %%ymm1,%%ymm3,%%ymm"#c1_no"; vfmadd231pd %%ymm2,%%ymm3,%%ymm"#c2_no";"
//#define KERNEL_h_k1m4n2 \
//  "vmovddup (%0),%%ymm1; vmovddup 8(%0),%%ymm2; addq $32,%0;" unit_acc_m4n2(4,5,%1)
#define unit_acc_m4n2(c1_no,c2_no,...)\
  "vbroadcastsd ("#__VA_ARGS__"),%%ymm2; vfmadd231pd %%ymm1,%%ymm2,%%ymm"#c1_no";vbroadcastsd 8("#__VA_ARGS__"),%%ymm3; vfmadd231pd %%ymm1,%%ymm3,%%ymm"#c2_no";"
#define KERNEL_h_k1m4n2 \
  "vmovaps (%0),%%ymm1; addq $32,%0;" unit_acc_m4n2(4,5,%1)
#define KERNEL_k1m4n2 KERNEL_h_k1m4n2 "addq $16,%1;"
#define KERNEL_h_k1m4n4 KERNEL_h_k1m4n2 unit_acc_m4n2(6,7,%1,%%r12,1)
#define KERNEL_k1m4n4 KERNEL_h_k1m4n4 "addq $16,%1;"
#define KERNEL_k1m4n6 KERNEL_h_k1m4n4 unit_acc_m4n2(8,9,%1,%%r12,2) "addq $16,%1;"
#define KERNEL_h_k1m4n8 KERNEL_k1m4n6 unit_acc_m4n2(10,11,%%r15)
#define KERNEL_k1m4n8 KERNEL_h_k1m4n8 "addq $16,%%r15;"
#define save_init_m4 "movq %2,%3; addq $32,%2;"
#define SAVE_m4n1 "vaddpd (%2),%%ymm4,%%ymm4; vmovupd %%ymm4,(%2); addq $32,%2;"
//#define unit_save_m4n2(c1_no,c2_no)\
//  "vunpcklpd %%ymm"#c2_no",%%ymm"#c1_no",%%ymm1; vfmadd213pd (%3),%%ymm0,%%ymm1; vmovupd %%ymm1,(%3);"\
//  "vunpckhpd %%ymm"#c2_no",%%ymm"#c1_no",%%ymm2; vfmadd213pd (%3,%4,1),%%ymm0,%%ymm2; vmovupd %%ymm2,(%3,%4,1); leaq (%3,%4,2),%3;"
#define unit_save_m4n2(c1_no,c2_no)\
  "vaddpd (%3),%%ymm"#c1_no",%%ymm"#c1_no"; vmovups %%ymm"#c1_no",(%3);"\
  "vaddpd (%3,%4,1),%%ymm"#c2_no",%%ymm"#c2_no"; vmovups %%ymm"#c2_no",(%3,%4,1);leaq (%3,%4,2),%3;"
#define SAVE_m4n2 save_init_m4 unit_save_m4n2(4,5)
#define SAVE_m4n4 SAVE_m4n2 unit_save_m4n2(6,7)
#define SAVE_m4n6 SAVE_m4n4 unit_save_m4n2(8,9)
#define SAVE_m4n8 SAVE_m4n6 unit_save_m4n2(10,11)
#define INIT_m4n1 "vpxor %%ymm4,%%ymm4,%%ymm4;"
#define unit_init_2ymm(c1_no,c2_no) "vpxor %%ymm"#c1_no",%%ymm"#c1_no",%%ymm"#c1_no"; vpxor %%ymm"#c2_no",%%ymm"#c2_no",%%ymm"#c2_no";"
#define INIT_m4n2 unit_init_2ymm(4,5)
#define INIT_m4n4 INIT_m4n2 unit_init_2ymm(6,7)
#define INIT_m4n6 INIT_m4n4 unit_init_2ymm(8,9)
#define INIT_m4n8 INIT_m4n6 unit_init_2ymm(10,11)


#define add_chksum_m4_1col(c1_no,c2_no)\
  "vaddpd %%ymm"#c1_no", %%ymm"#c2_no", %%ymm"#c1_no";"
#define save_chksum_m4_1col\
  "vaddpd (%7), %%ymm4, %%ymm4;vmovaps %%ymm4, (%7); addq $32, %7;"
#define SAVE_CHKSUM_m4n1 save_chksum_m4_1col
#define SAVE_CHKSUM_m4n2_h add_chksum_m4_1col(4,5)
#define SAVE_CHKSUM_m4n2 SAVE_CHKSUM_m4n2_h save_chksum_m4_1col
#define SAVE_CHKSUM_m4n4_h SAVE_CHKSUM_m4n2_h add_chksum_m4_1col(6,7) add_chksum_m4_1col(4,6)
#define SAVE_CHKSUM_m4n4 SAVE_CHKSUM_m4n4_h save_chksum_m4_1col
#define SAVE_CHKSUM_m4n6_h SAVE_CHKSUM_m4n4_h add_chksum_m4_1col(8,9) add_chksum_m4_1col(4,8)
#define SAVE_CHKSUM_m4n6 SAVE_CHKSUM_m4n6_h save_chksum_m4_1col
#define SAVE_CHKSUM_m4n8_h SAVE_CHKSUM_m4n6_h add_chksum_m4_1col(10,11) add_chksum_m4_1col(4,10)
#define SAVE_CHKSUM_m4n8 SAVE_CHKSUM_m4n8_h save_chksum_m4_1col


#define KERNEL_k1m2n1 \
  "vmovddup (%1),%%xmm1; addq $8,%1;"\
  "vfmadd231pd (%0),%%xmm1,%%xmm4; addq $16,%0;"
//#define unit_acc_m2n2(c1_no,c2_no,...)\
//  "vmovupd ("#__VA_ARGS__"),%%xmm3; vfmadd231pd %%xmm1,%%xmm3,%%xmm"#c1_no"; vfmadd231pd %%xmm2,%%xmm3,%%xmm"#c2_no";"
//#define KERNEL_h_k1m2n2 \
//  "vmovddup (%0),%%xmm1; vmovddup 8(%0),%%xmm2; addq $16,%0;" unit_acc_m2n2(4,5,%1)
#define unit_acc_m2n2(c1_no,c2_no,...)\
  "vmovddup ("#__VA_ARGS__"),%%xmm2; vfmadd231pd %%xmm1,%%xmm2,%%xmm"#c1_no";vmovddup 8("#__VA_ARGS__"),%%xmm3; vfmadd231pd %%xmm1,%%xmm3,%%xmm"#c2_no";"
#define KERNEL_h_k1m2n2 \
  "vmovaps (%0),%%xmm1; addq $16,%0;" unit_acc_m2n2(4,5,%1)
#define KERNEL_k1m2n2 KERNEL_h_k1m2n2 "addq $16,%1;"
#define KERNEL_h_k1m2n4 KERNEL_h_k1m2n2 unit_acc_m2n2(6,7,%1,%%r12,1)
#define KERNEL_k1m2n4 KERNEL_h_k1m2n4 "addq $16,%1;"
#define KERNEL_k1m2n6 KERNEL_h_k1m2n4 unit_acc_m2n2(8,9,%1,%%r12,2) "addq $16,%1;"
#define KERNEL_h_k1m2n8 KERNEL_k1m2n6 unit_acc_m2n2(10,11,%%r15)
#define KERNEL_k1m2n8 KERNEL_h_k1m2n8 "addq $16,%%r15;"
#define save_init_m2 "movq %2,%3; addq $16,%2;"
#define SAVE_m2n1 "vaddpd (%2),%%xmm4,%%xmm4; vmovupd %%xmm4,(%2); addq $16,%2;"
//#define unit_save_m2n2(c1_no,c2_no)\
//  "vunpcklpd %%xmm"#c2_no",%%xmm"#c1_no",%%xmm1; vfmadd213pd (%3),%%xmm0,%%xmm1; vmovupd %%xmm1,(%3);"\
//  "vunpckhpd %%xmm"#c2_no",%%xmm"#c1_no",%%xmm2; vfmadd213pd (%3,%4,1),%%xmm0,%%xmm2; vmovupd %%xmm2,(%3,%4,1); leaq (%3,%4,2),%3;"
#define unit_save_m2n2(c1_no,c2_no)\
  "vaddpd (%3),%%xmm"#c1_no",%%xmm"#c1_no"; vmovups %%xmm"#c1_no",(%3);"\
  "vaddpd (%3,%4,1),%%xmm"#c2_no",%%xmm"#c2_no"; vmovups %%xmm"#c2_no",(%3,%4,1);leaq (%3,%4,2),%3;"
#define SAVE_m2n2 save_init_m2 unit_save_m2n2(4,5)
#define SAVE_m2n4 SAVE_m2n2 unit_save_m2n2(6,7)
#define SAVE_m2n6 SAVE_m2n4 unit_save_m2n2(8,9)
#define SAVE_m2n8 SAVE_m2n6 unit_save_m2n2(10,11)
#define INIT_m2n1 "vpxor %%xmm4,%%xmm4,%%xmm4;"
#define unit_init_2xmm(c1_no,c2_no) "vpxor %%xmm"#c1_no",%%xmm"#c1_no",%%xmm"#c1_no"; vpxor %%xmm"#c2_no",%%xmm"#c2_no",%%xmm"#c2_no";"
#define INIT_m2n2 unit_init_2xmm(4,5)
#define INIT_m2n4 INIT_m2n2 unit_init_2xmm(6,7)
#define INIT_m2n6 INIT_m2n4 unit_init_2xmm(8,9)
#define INIT_m2n8 INIT_m2n6 unit_init_2xmm(10,11)


#define add_chksum_m2_1col(c1_no,c2_no)\
  "vaddpd %%xmm"#c1_no", %%xmm"#c2_no", %%xmm"#c1_no";"
#define save_chksum_m2_1col\
  "vaddpd (%7), %%xmm4, %%xmm4;vmovaps %%xmm4, (%7); addq $16, %7;"
#define SAVE_CHKSUM_m2n1 save_chksum_m2_1col
#define SAVE_CHKSUM_m2n2_h add_chksum_m2_1col(4,5)
#define SAVE_CHKSUM_m2n2 SAVE_CHKSUM_m2n2_h save_chksum_m2_1col
#define SAVE_CHKSUM_m2n4_h SAVE_CHKSUM_m2n2_h add_chksum_m2_1col(6,7) add_chksum_m2_1col(4,6)
#define SAVE_CHKSUM_m2n4 SAVE_CHKSUM_m2n4_h save_chksum_m2_1col
#define SAVE_CHKSUM_m2n6_h SAVE_CHKSUM_m2n4_h add_chksum_m2_1col(8,9) add_chksum_m2_1col(4,8)
#define SAVE_CHKSUM_m2n6 SAVE_CHKSUM_m2n6_h save_chksum_m2_1col
#define SAVE_CHKSUM_m2n8_h SAVE_CHKSUM_m2n6_h add_chksum_m2_1col(10,11) add_chksum_m2_1col(4,10)
#define SAVE_CHKSUM_m2n8 SAVE_CHKSUM_m2n8_h save_chksum_m2_1col


#define KERNEL_k1m1n1 \
  "vmovsd (%1),%%xmm1; addq $8,%1;"\
  "vfmadd231sd (%0),%%xmm1,%%xmm4; addq $8,%0;"
//#define KERNEL_h_k1m1n2 \
//  "vmovddup (%0),%%xmm1; addq $8,%0;"\
//  "vfmadd231pd (%1),%%xmm1,%%xmm4;"
#define unit_acc_m1n2(c1_no,c2_no,...)\
  "vmovsd ("#__VA_ARGS__"),%%xmm2; vfmadd231sd %%xmm1,%%xmm2,%%xmm"#c1_no";vmovsd 8("#__VA_ARGS__"),%%xmm3; vfmadd231sd %%xmm1,%%xmm3,%%xmm"#c2_no";"
#define KERNEL_h_k1m1n2 \
  "vmovsd (%0),%%xmm1; addq $8,%0;" unit_acc_m1n2(4,5,%1)
#define KERNEL_k1m1n2 KERNEL_h_k1m1n2 "addq $16,%1;"
#define KERNEL_h_k1m1n4 KERNEL_h_k1m1n2 unit_acc_m1n2(6,7,%1,%%r12,1)
#define KERNEL_k1m1n4 KERNEL_h_k1m1n4 "addq $16,%1;"
#define KERNEL_k1m1n6 KERNEL_h_k1m1n4 unit_acc_m1n2(8,9,%1,%%r12,2) "addq $16,%1;"
#define KERNEL_h_k1m1n8 KERNEL_k1m1n6 unit_acc_m1n2(10,11,%%r15)
#define KERNEL_k1m1n8 KERNEL_h_k1m1n8 "addq $16,%%r15;"
//#define KERNEL_k1m1n2 KERNEL_h_k1m1n2 "addq $16,%1;"
//#define KERNEL_h_k1m1n4 KERNEL_h_k1m1n2 "vfmadd231pd (%1,%%r12,1),%%xmm1,%%xmm5;"
//#define KERNEL_k1m1n4 KERNEL_h_k1m1n4 "addq $16,%1;"
//#define KERNEL_k1m1n6 KERNEL_h_k1m1n4 "vfmadd231pd (%1,%%r12,2),%%xmm1,%%xmm6; addq $16,%1;"
//#define KERNEL_h_k1m1n8 KERNEL_k1m1n6 "vfmadd231pd (%%r15),%%xmm1,%%xmm7;"
//#define KERNEL_k1m1n8 KERNEL_h_k1m1n8 "addq $16,%%r15;"
#define save_init_m1 "movq %2,%3; addq $8,%2;"
#define SAVE_m1n1 "vaddsd (%2),%%xmm4,%%xmm4; vmovsd %%xmm4,(%2); addq $8,%2;"
//#define unit_save_m1n2(c1_no)\
//  "vmovsd (%3),%%xmm2; vmovhpd (%3,%4,1),%%xmm2,%%xmm2; vfmadd231pd %%xmm"#c1_no",%%xmm0,%%xmm2; vmovsd %%xmm2,(%3); vmovhpd %%xmm2,(%3,%4,1); leaq (%3,%4,2),%3;"
#define unit_save_m1n2(c1_no,c2_no)\
  "vaddsd (%3),%%xmm"#c1_no",%%xmm"#c1_no"; vmovsd %%xmm"#c1_no",(%3);"\
  "vaddsd (%3,%4,1),%%xmm"#c2_no",%%xmm"#c2_no"; vmovsd %%xmm"#c2_no",(%3,%4,1);leaq (%3,%4,2),%3;"
#define SAVE_m1n2 save_init_m1 unit_save_m1n2(4,5)
#define SAVE_m1n4 SAVE_m1n2 unit_save_m1n2(6,7)
#define SAVE_m1n6 SAVE_m1n4 unit_save_m1n2(8,9)
#define SAVE_m1n8 SAVE_m1n6 unit_save_m1n2(10,11)
//#define INIT_m1n1 "vpxor %%xmm4,%%xmm4,%%xmm4;"
//#define INIT_m1n2 INIT_m1n1
//#define INIT_m1n4 INIT_m1n2 "vpxor %%xmm5,%%xmm5,%%xmm5;"
//#define INIT_m1n6 INIT_m1n4 "vpxor %%xmm6,%%xmm6,%%xmm6;"
//#define INIT_m1n8 INIT_m1n6 "vpxor %%xmm7,%%xmm7,%%xmm7;"
#define INIT_m1n1 "vpxor %%xmm4,%%xmm4,%%xmm4;"
#define INIT_m1n2 unit_init_2xmm(4,5)
#define INIT_m1n4 INIT_m1n2 unit_init_2xmm(6,7)
#define INIT_m1n6 INIT_m1n4 unit_init_2xmm(8,9)
#define INIT_m1n8 INIT_m1n6 unit_init_2xmm(10,11)

#define add_chksum_m1_1col(c1_no,c2_no)\
  "vaddsd %%xmm"#c1_no", %%xmm"#c2_no", %%xmm"#c1_no";"
#define save_chksum_m1_1col\
  "vaddsd (%7), %%xmm4, %%xmm4;vmovsd %%xmm4, (%7); addq $8, %7;"
#define SAVE_CHKSUM_m1n1 save_chksum_m1_1col
#define SAVE_CHKSUM_m1n2_h add_chksum_m1_1col(4,5)
#define SAVE_CHKSUM_m1n2 SAVE_CHKSUM_m1n2_h save_chksum_m1_1col
#define SAVE_CHKSUM_m1n4_h SAVE_CHKSUM_m1n2_h add_chksum_m1_1col(6,7) add_chksum_m1_1col(4,6)
#define SAVE_CHKSUM_m1n4 SAVE_CHKSUM_m1n4_h save_chksum_m1_1col
#define SAVE_CHKSUM_m1n6_h SAVE_CHKSUM_m1n4_h add_chksum_m1_1col(8,9) add_chksum_m1_1col(4,8)
#define SAVE_CHKSUM_m1n6 SAVE_CHKSUM_m1n6_h save_chksum_m1_1col
#define SAVE_CHKSUM_m1n8_h SAVE_CHKSUM_m1n6_h add_chksum_m1_1col(10,11) add_chksum_m1_1col(4,10)
#define SAVE_CHKSUM_m1n8 SAVE_CHKSUM_m1n8_h save_chksum_m1_1col

#define COMPUTE_SIMPLE(mdim,ndim)\
  INIT_m##mdim##n##ndim "testq %%r13,%%r13; jz 7"#mdim"7"#ndim"9f;"\
  "movq %%r13,%5; movq %%r14,%1; leaq (%%r14,%%r12,2),%%r15; addq %%r12,%%r15;"\
  "7"#mdim"7"#ndim"1:\n\t"\
  KERNEL_k1m##mdim##n##ndim "decq %5; jnz 7"#mdim"7"#ndim"1b;"\
  "7"#mdim"7"#ndim"9:\n\t"\
  SAVE_m##mdim##n##ndim\
  SAVE_CHKSUM_m##mdim##n##ndim
#define COMPUTE_m24n1 COMPUTE_SIMPLE(24,1)
#define COMPUTE_m24n2 COMPUTE_SIMPLE(24,2)
#define COMPUTE_m24n4 COMPUTE_SIMPLE(24,4)
#define COMPUTE_m24n6 COMPUTE_SIMPLE(24,6)
//#define COMPUTE_m24n8 COMPUTE_SIMPLE(24,8)
#define COMPUTE_m16n1 COMPUTE_SIMPLE(16,1)
#define COMPUTE_m16n2 COMPUTE_SIMPLE(16,2)
#define COMPUTE_m16n4 COMPUTE_SIMPLE(16,4)
#define COMPUTE_m16n6 COMPUTE_SIMPLE(16,6)
#define COMPUTE_m16n8 COMPUTE_SIMPLE(16,8)
#define COMPUTE_m8n1 COMPUTE_SIMPLE(8,1)
#define COMPUTE_m8n2 COMPUTE_SIMPLE(8,2)
#define COMPUTE_m8n4 COMPUTE_SIMPLE(8,4)
#define COMPUTE_m8n6 COMPUTE_SIMPLE(8,6)
#define COMPUTE_m8n8 COMPUTE_SIMPLE(8,8)

#define COMPUTE_m24n8 \
  INIT_m24n8 "movq %%r13,%5; movq %%r14,%1; leaq (%%r14,%%r12,2),%%r15; addq %%r12,%%r15; movq %2,%3;"\
  "cmpq $16,%5; jb 724783f; movq $16,%5;"\
  "724781:\n\t"\
  KERNEL_k1m24n8 "addq $4,%5; testq $12,%5; movq $172,%%r10; cmovz %4,%%r10;"\
  KERNEL_k1m24n8 "prefetcht1 (%3); subq $129,%3; addq %%r10,%3;"\
  KERNEL_k1m24n8 "prefetcht1 (%6); addq $32,%6; cmpq $208,%5; cmoveq %2,%3;"\
  KERNEL_k1m24n8 "cmpq %5,%%r13; jnb 724781b;"\
  "movq %2,%3; negq %5; leaq 16(%%r13,%5,1),%5;"\
  "724783:\n\t"\
  "testq %5,%5; jz 724789f;"\
  "724785:\n\t"\
  "prefetcht0 (%3); prefetcht0 64(%3); prefetcht0 128(%3);"\
  KERNEL_k1m24n8 "addq %4,%3; decq %5;jnz 724785b;"\
  "724789:\n\t"\
  "prefetcht0 (%%r14); prefetcht0 (%7); prefetcht0 64(%7); prefetcht0 128(%7);" SAVE_m24n8 SAVE_CHKSUM_m24n8

#define COMPUTE(ndim) {\
  b_pref = b_ptr + ndim * K;\
  __asm__ __volatile__(\
    "vbroadcastsd %10,%%zmm0; movq %9,%%r11; movq %1,%%r14; movq %5,%%r13; movq %5,%%r12; salq $4,%%r12;"\
    "cmpq $24,%%r11; jb "#ndim"33101f;"\
    #ndim"33100:\n\t"\
    COMPUTE_m24n##ndim "subq $24,%%r11; cmpq $24,%%r11; jnb "#ndim"33100b;"\
    #ndim"33101:\n\t"\
    "cmpq $16,%%r11; jb "#ndim"33102f;"\
    COMPUTE_SIMPLE(16,ndim) "subq $16,%%r11;"\
    #ndim"33102:\n\t"\
    "cmpq $8,%%r11; jb "#ndim"33103f;"\
    COMPUTE_SIMPLE(8,ndim) "subq $8,%%r11;"\
    #ndim"33103:\n\t"\
    "cmpq $4,%%r11; jb "#ndim"33104f;"\
    COMPUTE_SIMPLE(4,ndim) "subq $4,%%r11;"\
    #ndim"33104:\n\t"\
    "cmpq $2,%%r11; jb "#ndim"33105f;"\
    COMPUTE_SIMPLE(2,ndim) "subq $2,%%r11;"\
    #ndim"33105:\n\t"\
    "testq %%r11,%%r11; jz "#ndim"33106f;"\
    COMPUTE_SIMPLE(1,ndim) "subq $1,%%r11;"\
    #ndim"33106:\n\t"\
    "movq %%r14,%1; movq %%r13,%5;"\
  :"+r"(a_ptr),"+r"(b_ptr),"+r"(c_ptr),"+r"(c_tmp),"+r"(ldc_in_bytes),"+r"(K),"+r"(b_pref),"+r"(chk_c_ptr_pos),"+r"(chk_c_row_ptr_pos):"m"(M),"m"(ALPHA):"r10","r11","r12","r13","r14","r15","cc","memory",\
    "zmm0","zmm1","zmm2","zmm3","zmm4","zmm5","zmm6","zmm7","zmm8","zmm9","zmm10","zmm11","zmm12","zmm13","zmm14","zmm15",\
    "zmm16","zmm17","zmm18","zmm19","zmm20","zmm21","zmm22","zmm23","zmm24","zmm25","zmm26","zmm27","zmm28","zmm29","zmm30","zmm31");\
  a_ptr -= M * K; b_ptr += ndim * K; c_ptr += ndim * ldc - M;\
}
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define BLASLONG int
int __attribute__ ((noinline))
CNAME(BLASLONG m, BLASLONG n, BLASLONG k, double alpha, double * __restrict__ A, double * __restrict__ B, double * __restrict__ C, BLASLONG ldc, double * __restrict__ chk_c_col_ptr,double * __restrict__  chk_c_row_ptr)
{
    if(m==0||n==0||k==0||alpha==0.0) return 0;
    int64_t ldc_in_bytes = (int64_t)ldc * sizeof(double); double ALPHA = alpha;
    int64_t M = (int64_t)m, K = (int64_t)k;
    BLASLONG n_count = n;
    double *a_ptr = A,*b_ptr = B,*c_ptr = C,*c_tmp = C,*b_pref = B, *chk_c_ptr_pos = chk_c_col_ptr, *chk_c_row_ptr_pos = chk_c_row_ptr;
    for(;n_count>7;n_count-=8) {
      chk_c_ptr_pos = chk_c_col_ptr;
      COMPUTE(8)
      chk_c_row_ptr_pos+=8;
    }
    for(;n_count>5;n_count-=6) {
      chk_c_ptr_pos = chk_c_col_ptr;
      COMPUTE(6)
      chk_c_row_ptr_pos+=6;
    }
    for(;n_count>3;n_count-=4) {
      chk_c_ptr_pos = chk_c_col_ptr;
      COMPUTE(4)
      chk_c_row_ptr_pos+=4;
    }
    for(;n_count>1;n_count-=2) {
      chk_c_ptr_pos = chk_c_col_ptr;
      COMPUTE(2)
      chk_c_row_ptr_pos+=2;
    }
    if(n_count>0) {
      chk_c_ptr_pos = chk_c_col_ptr;
      COMPUTE(1)
    }
    return 0;
}

static void dgemm_tcopy_2(double *src, double *dst, BLASLONG lead_dim, BLASLONG dim_first, BLASLONG dim_second){
//src_leading_dim parallel with dst_tile_leading_dim
    if(dim_first==0 || dim_second==0) return;
    BLASLONG count_first,count_second;
    double *tosrc,*todst;
    for(count_second=0;count_second<dim_second;count_second++){
      tosrc = src + count_second * lead_dim;
      todst = dst + count_second * 2;
      for(count_first=dim_first;count_first>1;count_first-=2){
        todst[0]=tosrc[0];todst[1]=tosrc[1];
        tosrc+=2;todst+=2*dim_second;
      }
      todst -= count_second;
      if(count_first>0) *todst=*tosrc;
    }
}
static void dgemm_ncopy_2(double *src, double *dst, BLASLONG lead_dim, BLASLONG dim_first, BLASLONG dim_second,double *chk_A,double *chk_B,double *chk_online_C_row){
//src_leading_dim perpendicular to dst_tile_leading_dim
    if(dim_first==0 || dim_second==0) return;
    BLASLONG count_first,count_second,tosrc_inc;
    double *tosrc1,*tosrc2;
    double *todst=dst;
    double *ptr_chk_b = chk_B, *ptr_chk_a = chk_A, *ptr_chk_c = chk_online_C_row;
    double ld1,ld2,ld_a,ld_c1,ld_c2;
    tosrc1=src;tosrc2=tosrc1+lead_dim;
    tosrc_inc=2*lead_dim-dim_first;
    for(count_second=dim_second;count_second>1;count_second-=2){
      ptr_chk_b = chk_B;ptr_chk_a = chk_A;
      ld_c1 = 0.;ld_c2 = 0.;
      for(count_first=0;count_first<dim_first;count_first++){
        ld1 = *tosrc1;ld2 = *tosrc2;ld_a = *ptr_chk_a;
        ld_c1 += (ld1 * ld_a);ld_c2 += (ld2 * ld_a);
        todst[0]=ld1;tosrc1++;todst[1]=ld2;tosrc2++;
        *ptr_chk_b += (ld1 + ld2);
        todst+=2;ptr_chk_b++;ptr_chk_a++;
      }
      ptr_chk_c[0]+=ld_c1;ptr_chk_c[1]+=ld_c2;
      tosrc1+=tosrc_inc;tosrc2+=tosrc_inc;ptr_chk_c+=2;
    }
    tosrc_inc-=lead_dim;
    if(count_second>0){
      ptr_chk_b = chk_B;ptr_chk_a = chk_A;
      ld_c1 = 0.;
      for(count_first=0;count_first<dim_first;count_first++){
        ld1 = *tosrc1;
        ld_c1 += (ld1 * ld_a);
        todst[0]=ld1;tosrc1++;
        *ptr_chk_b += ld1;
        todst++;ptr_chk_b++;ptr_chk_a++;
      }
      ptr_chk_c[0]+=ld_c1;
    }
}

void ocopy_symm(double *src, double *dst, int lead_dim, int dim_first, int dim_second, int m_start, int n_start,double *chk_b, double *chk_c){
    if(dim_first==0 || dim_second==0) return;
    int count_first,count_second;
    int c24,c16,c8,c4,c2,c1;
    int remain = dim_first;
    int count_zero, count_inner;
    c24=remain/24;
    remain -= 24*c24;
    c16 = (remain>15)?remain/16:0;
    remain -= 16*c16;
    c8 = (remain>7)?1:0;
    remain -= 8*c8;
    c4 = (remain>3)?1:0;
    remain -= 4*c4;
    c2 = (remain>1)?1:0;
    remain -= 2*c2;
    c1 = remain;
    double *todst,*tosrc_v,*tosrc_h;
    double *ptr_24 = dst + 24*dim_second*c24;
    double *ptr_16 = ptr_24 +16*dim_second*c16;
    double *ptr_8 = ptr_16+8*dim_second*c8;
    double *ptr_4 = ptr_8+4*dim_second*c4;
    double *ptr_2 = ptr_4+2*dim_second*c2;
    double *ptr_chk_c, ld_a;
    int count_third;
    for(count_second=0,count_third=n_start;count_second<dim_second;count_second++,count_third++){
      tosrc_h = src + m_start + n_start*lead_dim + count_second * lead_dim;
      tosrc_v = src + m_start*lead_dim + n_start + count_second;
      todst = dst + count_second * 24;
      count_zero=m_start;
      ptr_chk_c = chk_c;
      double ld_chk_b = *(chk_b + count_second);
      for(count_first=dim_first;count_first>23;count_first-=24){
        // __m512d ld1 = _mm512_loadu_pd(tosrc);
        // __m512d ld2 = _mm512_loadu_pd(tosrc+8);
        // __m512d ld3 = _mm512_loadu_pd(tosrc+16);
        // _mm512_storeu_pd(todst,ld1);
        // _mm512_storeu_pd(todst+8,ld2);
        // _mm512_storeu_pd(todst+16,ld3);
        for (count_inner=0;count_inner<24;count_inner++){
            ld_a = (count_third<count_zero)?*tosrc_h:*tosrc_v;
            todst[count_inner]= ld_a;
            *ptr_chk_c += ld_a * ld_chk_b;
            tosrc_h++;
            ptr_chk_c++;
            tosrc_v+=lead_dim;
            count_zero++;
        }
        // tosrc+=24;
        todst+=24*dim_second;
      }
      // todst = dst + 24*dim_second*c24+16*count_second;
      todst = ptr_24 + 16*count_second;
      for(;count_first>15;count_first-=16){
        // __m512d ld1 = _mm512_loadu_pd(tosrc);
        // __m512d ld2 = _mm512_loadu_pd(tosrc+8);
        // _mm512_storeu_pd(todst,ld1);
        // _mm512_storeu_pd(todst+8,ld2);
        // tosrc+=16;
        for (count_inner=0;count_inner<16;count_inner++){
            ld_a = (count_third<count_zero)?*tosrc_h:*tosrc_v;
            todst[count_inner]= ld_a;
            *ptr_chk_c += ld_a * ld_chk_b;
            tosrc_h++;
            ptr_chk_c++;
            tosrc_v+=lead_dim;
            count_zero++;
        }
        todst+=16*dim_second;
      }
      // todst = dst + 24*dim_second*c24+16*dim_second*c16+8*count_second;
      todst = ptr_16 + +8*count_second;
      for(;count_first>7;count_first-=8){
        // __m512d ld1 = _mm512_loadu_pd(tosrc);
        // _mm512_storeu_pd(todst,ld1);
        // tosrc+=8;
        for (count_inner=0;count_inner<8;count_inner++){
            ld_a = (count_third<count_zero)?*tosrc_h:*tosrc_v;
            todst[count_inner]= ld_a;
            *ptr_chk_c += ld_a * ld_chk_b;
            tosrc_h++;
            ptr_chk_c++;
            tosrc_v+=lead_dim;
            count_zero++;
        }
        todst+=8*dim_second;
      }
      // todst = dst + 24*dim_second*c24+16*dim_second*c16+8*dim_second*c8+4*count_second;
      todst = ptr_8 + 4*count_second;
      for(;count_first>3;count_first-=4){
        // __m256d ld1 = _mm256_loadu_pd(tosrc);
        // _mm256_storeu_pd(todst,ld1);
        // tosrc+=4;
        for (count_inner=0;count_inner<4;count_inner++){
            ld_a = (count_third<count_zero)?*tosrc_h:*tosrc_v;
            todst[count_inner]= ld_a;
            *ptr_chk_c += ld_a * ld_chk_b;
            tosrc_h++;
            ptr_chk_c++;
            tosrc_v+=lead_dim;
            count_zero++;
        }
        todst+=4*dim_second;
      }
      // todst = dst + 24*dim_second*c24+16*dim_second*c16+8*dim_second*c8+4*dim_second*c4+2*count_second;
      todst = ptr_4 + 2*count_second;
      for(;count_first>1;count_first-=2){
        // double ld1 = tosrc[0],ld2 = tosrc[1];
        // todst[0]=ld1;todst[1]=ld2;
        // tosrc+=2;
        for (count_inner=0;count_inner<2;count_inner++){
            ld_a = (count_third<count_zero)?*tosrc_h:*tosrc_v;
            todst[count_inner]= ld_a;
            *ptr_chk_c += ld_a * ld_chk_b;
            tosrc_h++;
            ptr_chk_c++;
            tosrc_v+=lead_dim;
            count_zero++;
        }
        todst+=2*dim_second;
      }
      // todst = dst + 24*dim_second*c24+16*dim_second*c16+8*dim_second*c8+4*dim_second*c4+2*dim_second*c2+count_second;
      todst = ptr_2 + count_second;
      if(count_first>0) {
        // double ld1 = tosrc[0];
        // *todst=ld1;
        ld_a = (count_third<count_zero)?*tosrc_h:*tosrc_v;
        *todst= ld_a;
        *ptr_chk_c += ld_a * ld_chk_b;
        tosrc_h++;
        ptr_chk_c++;
        tosrc_v+=lead_dim;
        count_zero++;
      }

    }
}

static void dgemm_tcopy_16(double *src, double *dst, BLASLONG lead_dim, BLASLONG dim_first, BLASLONG dim_second){
    if(dim_first==0 || dim_second==0) return;
    BLASLONG count_first,count_second;
    BLASLONG c24,c16,c8,c4,c2,c1;
    BLASLONG remain = dim_first;
    c24=remain/24;
    remain -= 24*c24;
    c16 = (remain>15)?remain/16:0;
    remain -= 16*c16;
    c8 = (remain>7)?1:0;
    remain -= 8*c8;
    c4 = (remain>3)?1:0;
    remain -= 4*c4;
    c2 = (remain>1)?1:0;
    remain -= 2*c2;
    c1 = remain;
    double *tosrc,*todst;
    double *ptr_24 = dst + 24*dim_second*c24;
    double *ptr_16 = ptr_24 +16*dim_second*c16;
    double *ptr_8 = ptr_16+8*dim_second*c8;
    double *ptr_4 = ptr_8+4*dim_second*c4;
    double *ptr_2 = ptr_4+2*dim_second*c2;
    for(count_second=0;count_second<dim_second;count_second++){
      tosrc = src + count_second * lead_dim;
      todst = dst + count_second * 24;
      for(count_first=dim_first;count_first>23;count_first-=24){
        __m512d ld1 = _mm512_loadu_pd(tosrc);
        __m512d ld2 = _mm512_loadu_pd(tosrc+8);
        __m512d ld3 = _mm512_loadu_pd(tosrc+16);
        _mm512_storeu_pd(todst,ld1);
        _mm512_storeu_pd(todst+8,ld2);
        _mm512_storeu_pd(todst+16,ld3);
        tosrc+=24;todst+=24*dim_second;
      }
      // todst = dst + 24*dim_second*c24+16*count_second;
      todst = ptr_24 + 16*count_second;
      for(;count_first>15;count_first-=16){
        __m512d ld1 = _mm512_loadu_pd(tosrc);
        __m512d ld2 = _mm512_loadu_pd(tosrc+8);
        _mm512_storeu_pd(todst,ld1);
        _mm512_storeu_pd(todst+8,ld2);
        tosrc+=16;todst+=16*dim_second;
      }
      // todst = dst + 24*dim_second*c24+16*dim_second*c16+8*count_second;
      todst = ptr_16 + +8*count_second;
      for(;count_first>7;count_first-=8){
        __m512d ld1 = _mm512_loadu_pd(tosrc);
        _mm512_storeu_pd(todst,ld1);
        tosrc+=8;todst+=8*dim_second;
      }
      // todst = dst + 24*dim_second*c24+16*dim_second*c16+8*dim_second*c8+4*count_second;
      todst = ptr_8 + 4*count_second;
      for(;count_first>3;count_first-=4){
        __m256d ld1 = _mm256_loadu_pd(tosrc);
        _mm256_storeu_pd(todst,ld1);
        tosrc+=4;todst+=4*dim_second;
      }
      // todst = dst + 24*dim_second*c24+16*dim_second*c16+8*dim_second*c8+4*dim_second*c4+2*count_second;
      todst = ptr_4 + 2*count_second;
      for(;count_first>1;count_first-=2){
        double ld1 = tosrc[0],ld2 = tosrc[1];
        todst[0]=ld1;todst[1]=ld2;
        tosrc+=2;todst+=2*dim_second;
      }
      // todst = dst + 24*dim_second*c24+16*dim_second*c16+8*dim_second*c8+4*dim_second*c4+2*dim_second*c2+count_second;
      todst = ptr_2 + count_second;
      if(count_first>0) {
        double ld1 = tosrc[0];
        *todst=ld1;
      }

    }
}
static void dgemm_ncopy_16(double *src, double *dst, BLASLONG lead_dim, BLASLONG dim_first, BLASLONG dim_second){
//src_leading_dim perpendicular to dst_tile_leading_dim
    if(dim_first==0 || dim_second==0) return;
    BLASLONG count_first,count_second,tosrc_inc;
    double *tosrc1,*tosrc2,*tosrc3,*tosrc4,*tosrc5,*tosrc6,*tosrc7,*tosrc8;
    double *tosrc9,*tosrc10,*tosrc11,*tosrc12,*tosrc13,*tosrc14,*tosrc15,*tosrc16;
    double *tosrc17,*tosrc18,*tosrc19,*tosrc20,*tosrc21,*tosrc22,*tosrc23,*tosrc24;
    double *todst=dst;
    tosrc1=src;tosrc2=tosrc1+lead_dim;tosrc3=tosrc2+lead_dim;tosrc4=tosrc3+lead_dim;
    tosrc5=tosrc4+lead_dim;tosrc6=tosrc5+lead_dim;tosrc7=tosrc6+lead_dim;tosrc8=tosrc7+lead_dim;
    tosrc9=tosrc8+lead_dim;tosrc10=tosrc9+lead_dim;tosrc11=tosrc10+lead_dim;tosrc12=tosrc11+lead_dim;
    tosrc13=tosrc12+lead_dim;tosrc14=tosrc13+lead_dim;tosrc15=tosrc14+lead_dim;tosrc16=tosrc15+lead_dim;
    tosrc17=tosrc16+lead_dim;tosrc18=tosrc17+lead_dim;tosrc19=tosrc18+lead_dim;tosrc20=tosrc19+lead_dim;
    tosrc21=tosrc20+lead_dim;tosrc22=tosrc21+lead_dim;tosrc23=tosrc22+lead_dim;tosrc24=tosrc23+lead_dim;
    tosrc_inc=24*lead_dim-dim_first;
    for(count_second=dim_second;count_second>23;count_second-=24){
      for(count_first=0;count_first<dim_first;count_first++){
        todst[0]=*tosrc1;tosrc1++;todst[1]=*tosrc2;tosrc2++;
        todst[2]=*tosrc3;tosrc3++;todst[3]=*tosrc4;tosrc4++;
        todst[4]=*tosrc5;tosrc5++;todst[5]=*tosrc6;tosrc6++;
        todst[6]=*tosrc7;tosrc7++;todst[7]=*tosrc8;tosrc8++;
        todst[8]=*tosrc9;tosrc9++;todst[9]=*tosrc10;tosrc10++;
        todst[10]=*tosrc11;tosrc11++;todst[11]=*tosrc12;tosrc12++;
        todst[12]=*tosrc13;tosrc13++;todst[13]=*tosrc14;tosrc14++;
        todst[14]=*tosrc15;tosrc15++;todst[15]=*tosrc16;tosrc16++;
        todst[16]=*tosrc17;tosrc17++;todst[17]=*tosrc18;tosrc18++;
        todst[18]=*tosrc19;tosrc19++;todst[19]=*tosrc20;tosrc20++;
        todst[20]=*tosrc21;tosrc21++;todst[21]=*tosrc22;tosrc22++;
        todst[22]=*tosrc23;tosrc23++;todst[23]=*tosrc24;tosrc24++;
        todst+=24;
      }
      tosrc1+=tosrc_inc;tosrc2+=tosrc_inc;tosrc3+=tosrc_inc;tosrc4+=tosrc_inc;
      tosrc5+=tosrc_inc;tosrc6+=tosrc_inc;tosrc7+=tosrc_inc;tosrc8+=tosrc_inc;
      tosrc9+=tosrc_inc;tosrc10+=tosrc_inc;tosrc11+=tosrc_inc;tosrc12+=tosrc_inc;
      tosrc13+=tosrc_inc;tosrc14+=tosrc_inc;tosrc15+=tosrc_inc;tosrc16+=tosrc_inc;
      tosrc17+=tosrc_inc;tosrc18+=tosrc_inc;tosrc19+=tosrc_inc;tosrc20+=tosrc_inc;
      tosrc21+=tosrc_inc;tosrc22+=tosrc_inc;tosrc23+=tosrc_inc;tosrc24+=tosrc_inc;
    }
    tosrc_inc=16*lead_dim-dim_first;
    for(;count_second>15;count_second-=16){
      for(count_first=0;count_first<dim_first;count_first++){
        todst[0]=*tosrc1;tosrc1++;todst[1]=*tosrc2;tosrc2++;
        todst[2]=*tosrc3;tosrc3++;todst[3]=*tosrc4;tosrc4++;
        todst[4]=*tosrc5;tosrc5++;todst[5]=*tosrc6;tosrc6++;
        todst[6]=*tosrc7;tosrc7++;todst[7]=*tosrc8;tosrc8++;
        todst[8]=*tosrc9;tosrc9++;todst[9]=*tosrc10;tosrc10++;
        todst[10]=*tosrc11;tosrc11++;todst[11]=*tosrc12;tosrc12++;
        todst[12]=*tosrc13;tosrc13++;todst[13]=*tosrc14;tosrc14++;
        todst[14]=*tosrc15;tosrc15++;todst[15]=*tosrc16;tosrc16++;
        todst+=16;
      }
      tosrc1+=tosrc_inc;tosrc2+=tosrc_inc;tosrc3+=tosrc_inc;tosrc4+=tosrc_inc;
      tosrc5+=tosrc_inc;tosrc6+=tosrc_inc;tosrc7+=tosrc_inc;tosrc8+=tosrc_inc;
      tosrc9+=tosrc_inc;tosrc10+=tosrc_inc;tosrc11+=tosrc_inc;tosrc12+=tosrc_inc;
      tosrc13+=tosrc_inc;tosrc14+=tosrc_inc;tosrc15+=tosrc_inc;tosrc16+=tosrc_inc;
    }
    tosrc_inc=8*lead_dim-dim_first;
    for(;count_second>7;count_second-=8){
      for(count_first=0;count_first<dim_first;count_first++){
        todst[0]=*tosrc1;tosrc1++;todst[1]=*tosrc2;tosrc2++;
        todst[2]=*tosrc3;tosrc3++;todst[3]=*tosrc4;tosrc4++;
        todst[4]=*tosrc5;tosrc5++;todst[5]=*tosrc6;tosrc6++;
        todst[6]=*tosrc7;tosrc7++;todst[7]=*tosrc8;tosrc8++;
        todst+=8;
      }
      tosrc1+=tosrc_inc;tosrc2+=tosrc_inc;tosrc3+=tosrc_inc;tosrc4+=tosrc_inc;
      tosrc5+=tosrc_inc;tosrc6+=tosrc_inc;tosrc7+=tosrc_inc;tosrc8+=tosrc_inc;
    }
    tosrc_inc=4*lead_dim-dim_first;
    for(;count_second>3;count_second-=4){
      for(count_first=0;count_first<dim_first;count_first++){
        todst[0]=*tosrc1;tosrc1++;todst[1]=*tosrc2;tosrc2++;
        todst[2]=*tosrc3;tosrc3++;todst[3]=*tosrc4;tosrc4++;
        todst+=4;
      }
      tosrc1+=tosrc_inc;tosrc2+=tosrc_inc;tosrc3+=tosrc_inc;tosrc4+=tosrc_inc;
    }
    tosrc_inc=2*lead_dim-dim_first;
    for(;count_second>1;count_second-=2){
      for(count_first=0;count_first<dim_first;count_first++){
        todst[0]=*tosrc1;tosrc1++;todst[1]=*tosrc2;tosrc2++;
        todst+=2;
      }
      tosrc1+=tosrc_inc;tosrc2+=tosrc_inc;
    }
    tosrc_inc=lead_dim-dim_first;
    if(count_second>0){
      for(count_first=0;count_first<dim_first;count_first++){
        todst[0]=*tosrc1;tosrc1++;
        todst++;
      }
    }
}


void symm_checksum_encoding_A_col_major(double *A, int m, int lda, double *chksm_A_col){
  int i,j,inner_count;
  int m4=m&-4;
  double *ptr_a1, *ptr_a2,*ptr_a3,*ptr_a4;
  double tmp1,tmp2,tmp3,tmp4;
  for (j=0;j<m4;j+=4){
      i = j;
      int i_left = m4 - j - 4;
      int i_left_8 = i_left & -8;
      // printf("j = %d, remain = %d, i_left_8 = %d\n", j, i_left - i_left_8, i_left_8);
      double ai1j = A(i+1,j), ai2j = A(i+2,j), ai3j = A(i+3,j), ai2j1 = A(i+2,j+1), ai3j1 = A(i+3,j+1), ai3j2 = A(i+3,j+2);
      double tmp1=0.,tmp2=0.,tmp3=0.,tmp4=0.;
      tmp1 += (A(i,j)+ai1j+ai2j+ai3j);
      tmp2 += (ai1j+A(i+1,j+1)+ai2j1+ai3j1);
      tmp3 += (ai2j+ai2j1+A(i+2,j+2)+ai3j2);
      tmp4 += (ai3j+ai3j1+ai3j2+A(i+3,j+3));
      i+=4;
      if (i_left - i_left_8 > 0){
        __m256d v1 = _mm256_loadu_pd(&A(i,j));
        __m256d v2 = _mm256_loadu_pd(&A(i,j+1));
        __m256d v3 = _mm256_loadu_pd(&A(i,j+2));
        __m256d v4 = _mm256_loadu_pd(&A(i,j+3));
        _mm256_storeu_pd(chksm_A_col+i, v1+v2+v3+v4+_mm256_loadu_pd(chksm_A_col+i));
        __m128d halfv1 = _mm_add_pd(_mm256_extractf128_pd(v1, 0), _mm256_extractf128_pd(v1, 1));
        __m128d halfv2 = _mm_add_pd(_mm256_extractf128_pd(v2, 0), _mm256_extractf128_pd(v2, 1));
        __m128d halfv3 = _mm_add_pd(_mm256_extractf128_pd(v3, 0), _mm256_extractf128_pd(v3, 1));
        __m128d halfv4 = _mm_add_pd(_mm256_extractf128_pd(v4, 0), _mm256_extractf128_pd(v4, 1));
        halfv1 = _mm_hadd_pd(halfv1, halfv1);
        halfv2 = _mm_hadd_pd(halfv2, halfv2);
        halfv3 = _mm_hadd_pd(halfv3, halfv3);
        halfv4 = _mm_hadd_pd(halfv4, halfv4);
        // printf("%f,%f,%f,%f\n",halfv1[0],halfv2[0],halfv3[0],halfv4[0]);
        tmp1+=halfv1[0];
        tmp2+=halfv2[0];
        tmp3+=halfv3[0];
        tmp4+=halfv4[0];
      }
      if (i_left_8>0){
        i+=(i_left - i_left_8);
        __m512d hv1 = _mm512_setzero_pd();
        __m512d hv2 = _mm512_setzero_pd();
        __m512d hv3 = _mm512_setzero_pd();
        __m512d hv4 = _mm512_setzero_pd();
        for (inner_count=0;inner_count<i_left_8;inner_count+=8){
          __m512d lv1 = _mm512_loadu_pd(&A(i,j));
          __m512d lv2 = _mm512_loadu_pd(&A(i,j+1));
          __m512d lv3 = _mm512_loadu_pd(&A(i,j+2));
          __m512d lv4 = _mm512_loadu_pd(&A(i,j+3));
          _mm512_storeu_pd(chksm_A_col+i, lv1+lv2+lv3+lv4+_mm512_loadu_pd(chksm_A_col+i));
          hv1 = _mm512_add_pd(hv1,lv1);
          hv2 = _mm512_add_pd(hv2,lv2);
          hv3 = _mm512_add_pd(hv3,lv3);
          hv4 = _mm512_add_pd(hv4,lv4);
          i+=8;
        }
        tmp1+=_mm512_reduce_add_pd(hv1);
        tmp2+=_mm512_reduce_add_pd(hv2);
        tmp3+=_mm512_reduce_add_pd(hv3);
        tmp4+=_mm512_reduce_add_pd(hv4);
      }
      chksm_A_col[j]+=tmp1;
      chksm_A_col[j+1]+=tmp2;
      chksm_A_col[j+2]+=tmp3;
      chksm_A_col[j+3]+=tmp4;
  }
  if (m4==m) return;
  for (j=0;j<m4;j++){
    double tmp = 0.;
    for (i=m4;i<m;i++){
      double aij = A(i,j);
      tmp += aij;
      chksm_A_col[i] += aij;
    }
    chksm_A_col[j] += tmp;
  }
  for (j=m4;j<m;j++){
    double tmp = A(j,j);
    for (i=j+1;i<m;i++){
      double aij = A(i,j);
      tmp += aij;
      chksm_A_col[i] += aij;
    }
    chksm_A_col[j] += tmp;
  }
}

static void SCALE_MULT(double *dat,double scale, BLASLONG lead_dim, BLASLONG dim_first, BLASLONG dim_second){
//dim_first parallel with leading dim; dim_second perpendicular to leading dim.
    if(dim_first==0 || dim_second==0 || scale==1.0) return;
    double *current_dat = dat;
    BLASLONG count_first,count_second;
    for(count_second=0;count_second<dim_second;count_second++){
      for(count_first=0;count_first<dim_first;count_first++){
        *current_dat *= scale; current_dat++;
      }
      current_dat += lead_dim - dim_first;
    }
}

#define GEMM_KERNEL(m_from,m_dim,n_from,n_dim,k_dim,ALPHA,sa,sb,C,LDC,chk_c_col_ptr,chk_c_row_ptr) CNAME(m_dim,n_dim,k_dim,ALPHA,sa,sb,(C)+(int64_t)(LDC)*(int64_t)(n_from)+(m_from),LDC,chk_c_col_ptr,chk_c_row_ptr)
#define GEMM_BETA(C,LDC,BETA,M,N) SCALE_MULT(C,BETA,LDC,M,N)
void GEMM_ICOPY(int m_from,int m_dim,int k_from,int k_dim,double *A,int lda,char transa,double *sa){
  if(transa=='N' || transa=='n') dgemm_tcopy_16(A+(int64_t)lda*(int64_t)k_from+m_from,sa,lda,m_dim,k_dim);
  else dgemm_ncopy_16(A+(int64_t)lda*(int64_t)m_from+k_from,sa,lda,k_dim,m_dim);
}
void GEMM_OCOPY(int k_from,int k_dim,int n_from,int n_dim,double *B,int ldb,double *sb,double *chk_A,double *chk_B,double *chk_online_C_row){
  dgemm_ncopy_2(B+(int64_t)ldb*(int64_t)n_from+k_from,sb,ldb,k_dim,n_dim,chk_A+k_from,chk_B+k_from,chk_online_C_row+n_from);
}

#define GEMM_R_M 1152
#define GEMM_R_N 9216
#define GEMM_D_M 192 // should be a multiple of 16
#define GEMM_D_N 96 // should be a multiple of 12
#define GEMM_D_K 384
#define COPY_MIN_DIM 192 //should be a multiple of 16

int get_copy_task(int32_t *dim_left, int *dim_start, int *dim_end){
  int acquire_status, dimend, taskdim; int32_t dimstart, dimleftload;
  //do{
    dimend = *dim_left;
    if(dimend<=0) return 0;
    dimleftload = dimend;
    taskdim = dimend % COPY_MIN_DIM; if(taskdim==0) taskdim = COPY_MIN_DIM;
    dimstart = dimend - taskdim;
    (*dim_left)=dimstart;
  *dim_start = dimstart; *dim_end = dimend;
  return 1;
}
int get_gemm_task(uint64_t *task_end, int *m_start, int *n_start, int *m_end, int *n_end, int m_min, int n_min, int m_max, int n_max){
  int acquire_status, mstart, mend, nstart, nend, nextm, nextn, mload, nload, nstride; uint64_t ll, load, write;
  do{
    ll = *task_end; mstart = ll & 0xFFFFFFFF; nstart = ll >> 32;
    if(mstart>=m_max || nstart>=n_max) return 0;
    nstride = GEMM_D_N;
    nend = nstart + nstride; mend = mstart + GEMM_D_M; if(mend>m_max) mend = m_max;
    nextm = mstart; nextn = nend;
    if(nend>=n_max){
      nend = n_max; nextm = mend; nextn = n_min;
    }
    load = ll; write = ((uint64_t)nextn << 32) | (uint64_t)nextm;
    __asm__ __volatile__("lock cmpxchgq %1,(%2);":"+a"(load):"r"(write),"r"(task_end):"cc","memory");
    if(load == ll) acquire_status = 1;
    else acquire_status = 0;
  }while(!acquire_status);
    *m_start = mstart; *n_start = nstart; *m_end = mend; *n_end = nend;
  return 1;
}
void ftblas_dsymm_ft(int m,int n,int k,double alpha,double *A,int lda,double *B,int ldb,double beta,double *C,int ldc){
  const int M = m, N = n, K = k, LDA = lda, LDB = ldb, LDC = ldc;
  const double ALPHA = alpha, BETA = beta;
#ifdef TIMING
  double t_tot=0.,t_copy_a=0.,t_copy_b=0.,t_copy_c=0.,t_A_encoding=0.,t_my=0.;
  double t0, t1;
#endif
  
  if(M<1||N<1) return;

  int MNK, MN;
  MN = (M<N)?N:M; MNK = (MN<K)?K:MNK;
  double *sa = (double *)aligned_alloc(4096,GEMM_D_K*GEMM_R_M*sizeof(double));
  double *sb = (double *)aligned_alloc(4096,GEMM_D_K*GEMM_R_N*sizeof(double));
  double *chk_A = (double *)calloc(K,sizeof(double));
  double *chk_B = (double *)calloc(K,sizeof(double));
  double *chk_online_C_row = (double *)calloc(N,sizeof(double));
  double *chk_online_C_col = (double *)calloc(M,sizeof(double));
#ifdef TIMING
  t0=get_sec();
#endif
  symm_checksum_encoding_A_col_major(A,M,LDA,chk_A);
#ifdef TIMING
  t1=get_sec();
  t_my+=(t1-t0);
#endif
  double *chk_c_col = (double *)aligned_alloc(4096,M*sizeof(double));
  double *chk_c_row = (double *)aligned_alloc(4096,N*sizeof(double));
  uint64_t task_end; int32_t adimleft, bdimleft;

    int m_count, n_count, k_count, m_inc, n_inc, k_inc;
    int m_start, n_start, m_end, n_end, copystart, copyend;
    for(k_count=0; k_count<K; k_count+=k_inc){
      k_inc = K-k_count; if(k_inc>GEMM_D_K) k_inc = GEMM_D_K;
      for(n_count=0; n_count<N; n_count+=n_inc){
        n_inc = N-n_count; if(n_inc>GEMM_R_N) n_inc = GEMM_R_N;
        bdimleft = n_inc;
        while(get_copy_task(&bdimleft,&copystart,&copyend)){
#ifdef TIMING
          t0=get_sec();
#endif
          GEMM_OCOPY(k_count,k_inc,n_count+copystart,copyend-copystart,B,LDB,sb+copystart*k_inc,chk_A,chk_B,chk_online_C_row);
#ifdef TIMING
          t1=get_sec();
          t_copy_b+=(t1-t0);
#endif
        }
        for(m_count=0; m_count<M; m_count+=m_inc){
          m_inc = M-m_count; if(m_inc>GEMM_R_M) m_inc = GEMM_R_M;
	  adimleft = m_inc;
          while(get_copy_task(&adimleft,&copystart,&copyend)){
#ifdef TIMING
            t0=get_sec();
#endif
            ocopy_symm(A,sa+copystart*k_inc,LDA,copyend-copystart,k_inc,m_count+copystart,k_count,chk_B+k_count,chk_online_C_col+m_count+copystart);
#ifdef TIMING
            t1=get_sec();
            t_copy_a+=(t1-t0);
#endif
          }
          task_end = (uint64_t)m_count | ((uint64_t)n_count<<32);
	  while(get_gemm_task(&task_end,&m_start,&n_start,&m_end,&n_end,m_count,n_count,m_count+m_inc,n_count+n_inc)){
            if(k_count == 0 && BETA != 1.0) {
#ifdef TIMING
              t0=get_sec();
#endif
              GEMM_BETA(C+n_start*LDC+m_start,LDC,BETA,m_end-m_start,n_end-n_start);
#ifdef TIMING
              t1=get_sec();
              t_copy_c+=(t1-t0);
#endif
            }
#ifdef TIMING
            t0=get_sec();
#endif
            if(ALPHA != 0.0) GEMM_KERNEL(m_start,m_end-m_start,n_start,n_end-n_start,k_inc,ALPHA,sa+(m_start-m_count)*k_inc,sb+(n_start-n_count)*k_inc,C,LDC,chk_c_col+m_count,chk_c_row+n_count);
#ifdef TIMING
            t1=get_sec();
            t_tot+=(t1-t0);
#endif
          }
        }
      }
    }
#ifdef PRINT
  printf("\n****** A checksum *******\n");
  print_vector(chk_A, K);
  printf("\n****** B checksum *******\n");
  print_vector(chk_B, K);
  printf("\n****** my C checksum row *******\n");
  print_vector(chk_online_C_row, N);
  printf("\n****** real C checksum row *******\n");
  print_vector(chk_c_row, N);
  printf("\n****** my C checksum col *******\n");
  print_vector(chk_online_C_col, M);
  printf("\n****** real C checksum col *******\n");
  print_vector(chk_c_col, M);
  free(sb); free(sa); free(chk_c_col); free(chk_c_row); 
  free(chk_A); free(chk_B); free(chk_online_C_col); free(chk_online_C_row);
#endif
#ifdef TIMING
  printf("Time in A chksm encoding: %f, perf: %f\n", t_my, 2*M*K*1e-9/t_my);
  printf("Time in major loop: %f\n", t_tot);
  printf("Time in copy A: %f, perf = %f GFLOPS\n", t_copy_a, 2.*1e-9*M*K/t_copy_a);
  printf("Time in copy B: %f, perf = %f GFLOPS\n", t_copy_b, 2.*1e-9*N*K/t_copy_b);
  printf("Time in copy C: %f, perf = %f GFLOPS\n", t_copy_c, 2.*1e-9*M*N/t_copy_c);
  printf("Total: %f\n", t_tot+t_copy_b+t_copy_c+t_copy_a+t_my);
#endif
}