static void scopy_kernel(long n, float *x, float *y)
{

	long i = 0;

	__asm__ __volatile__(
			".p2align 4				            			 \n\t"
			"1:				            	    			 \n\t"
			"vmovups	  0(%2, %0, 4) , %%zmm16	     	 \n\t"
			"vmovups	 64(%2, %0, 4) , %%zmm17	    	 \n\t"
			"vmovups	128(%2, %0, 4) , %%zmm18	     	 \n\t"
			"vmovups	192(%2, %0, 4) , %%zmm19	     	 \n\t"

			"vmovups   		   %%zmm16,	  0(%3, %0, 4)	     \n\t"
			"vmovups   		   %%zmm17,  64(%3, %0, 4)	     \n\t"
			"vmovups   		   %%zmm18, 128(%3, %0, 4)	     \n\t"
			"vmovups   		   %%zmm19, 192(%3, %0, 4)	     \n\t"
			
			"addq		$64, %0		  	 	    	 \n\t"
			"subq	    $64, %1			             \n\t"
			"jnz		1b		             	     \n\t"

			"vzeroupper					    \n\t"

			: "+r"(i),		// 0
			  "+r"(n)		// 1
			: "r"(x), 		// 2
			  "r"(y)		// 3
			: "cc",
//				"%xmm0", "%xmm1", "%xmm2", "%xmm3",
//				"%xmm4", "%xmm5", "%xmm6", "%xmm7",
//				"%xmm8", "%xmm9", "%xmm10", "%xmm11",
//				"%xmm12", "%xmm13", "%xmm14", "%xmm15",
				"memory");
}

int ft_scopy(long n, float *x, long inc_x, float *y, long inc_y) {
  long i = 0;
  long ix = 0, iy = 0;

  if (n <= 0)
    return 0;
/*
  printf("memory addr: %ld, mem_addr & 511 = %ld\n", (long)y, (long)y & 511);

  long bias = (long)y & (long)511;

  printf("bias = %ld\n", bias);

  void *ptr_dest_1 = (void *)(y + bias);
  void *ptr_dest_2 = (void *)y;
  FLOAT *new_dest  = (FLOAT *)ptr_dest_2;
  ptr_dest_2 += bias;
  printf("ADDR 1: %ld, ADDR 2: %ld\n", (long)ptr_dest_1, (long)ptr_dest_2);
  printf("diff1 = %ld, diff2 = %ld\n", (long)ptr_dest_1 - (long)y, (long)ptr_dest_2 - (long)y);
*/
  if ((inc_x == 1) && (inc_y == 1)) {

    long n1 = n & -64;
    if (n1 > 0) {
      scopy_kernel(n1, x, y);
      i = n1;
    }

    while (i < n) {
      y[i] = x[i];
      i++;

    }

  } else {

    while (i < n) {

      y[iy] = x[ix];
      ix += inc_x;
      iy += inc_y;
      i++;

    }

  }
  return 0;

}

void ftblas_scopy(const int n, const float *x, const int inc_x, const float *y, const int inc_y){
    ft_scopy(n, (float *)x, inc_x, (float *)y, inc_y);
}