cmake_minimum_required(VERSION 3.17)

project (ftblas)

set(CMAKE_VERBOSE_MAKEFILE ON)

add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(test)

link_directories(/home/yzhai015/intel/oneapi/mkl/latest/lib)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -lmkl_core -Wall -DMKL_ILP64 -m64 -lmkl_intel_lp64 -lmkl_intel_thread -liomp5 -lpthread -lm")

add_executable(ft_dnrm2_avx2 test/test_ft_dnrm2_avx2.c)
add_executable(ft_ddot_avx2 test/test_ft_ddot_avx2.c)
add_executable(dnrm2_avx2 test/test_dnrm2_avx2.c)
add_executable(ddot_avx2 test/test_ddot_avx2.c)
add_executable(daxpy_avx2 test/test_daxpy_avx2.c)
add_executable(dscal_avx2 test/test_dscal_avx2.c)

include_directories(/home/yzhai015/intel/oneapi/mkl/latest/include)
target_link_libraries(ft_dnrm2_avx2 IncludeFuncs SrcFuncs TestFuncs)
target_link_libraries(ft_ddot_avx2 IncludeFuncs SrcFuncs TestFuncs)
target_link_libraries(dnrm2_avx2 IncludeFuncs SrcFuncs TestFuncs)
target_link_libraries(ddot_avx2 IncludeFuncs SrcFuncs TestFuncs)
target_link_libraries(daxpy_avx2 IncludeFuncs SrcFuncs TestFuncs)
target_link_libraries(dscal_avx2 IncludeFuncs SrcFuncs TestFuncs)
