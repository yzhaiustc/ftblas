static void dcopy_kernel(long n, double *x, double *y)
{

	long i = 0;

	__asm__ __volatile__(
            ".p2align 4				            			 \n\t"
			"1:				            	    			 \n\t"
			"vmovupd	  0(%2, %0, 8) , %%zmm16	     	 \n\t"
			"vmovupd	 64(%2, %0, 8) , %%zmm17	    	 \n\t"
			"vmovupd	128(%2, %0, 8) , %%zmm18	     	 \n\t"
			"vmovupd	192(%2, %0, 8) , %%zmm19	     	 \n\t"

			"vmovupd   		   %%zmm16,	  0(%3, %0, 8)	     \n\t"
			"vmovupd   		   %%zmm17,  64(%3, %0, 8)	     \n\t"
			"vmovupd   		   %%zmm18, 128(%3, %0, 8)	     \n\t"
			"vmovupd   		   %%zmm19, 192(%3, %0, 8)	     \n\t"
			
			"addq		$32, %0		  	 	    	 \n\t"
			"subq	    $32, %1			             \n\t"
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

void ft_dcopy(int n, double *x, int inc_x, double *y, int inc_y) {
  long i = 0;
  long ix = 0, iy = 0;
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

    long n1 = n & -32;
    if (n1 > 0) {
      dcopy_kernel(n1, x, y);
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
}

void ftblas_dcopy(const int n, const double *x, const int inc_x, const double *y, const int inc_y){
    ft_dcopy(n, (double *)x, inc_x, (double *)y, inc_y);
}