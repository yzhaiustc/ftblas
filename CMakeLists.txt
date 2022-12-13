cmake_minimum_required(VERSION 3.17)

project (ftblas)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_VERBOSE_MAKEFILE ON)

add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(test)

find_package(MKL CONFIG REQUIRED)


link_directories(/home/yzhai015/intel/oneapi/mkl/latest/lib)
include_directories(/home/yzhai015/intel/oneapi/mkl/latest/include)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -lmkl_core -Wall -DMKL_ILP64 -m64 -lmkl_intel_lp64 -lmkl_intel_thread -liomp5 -lpthread -lm")

add_executable(ft_dnrm2_avx2 test/test_ft_dnrm2_avx2.cpp)
add_executable(ft_ddot_avx2 test/test_ft_ddot_avx2.cpp)
add_executable(dnrm2_avx2 test/test_dnrm2_avx2.cpp)
add_executable(ddot_avx2 test/test_ddot_avx2.cpp)
add_executable(daxpy_avx2 test/test_daxpy_avx2.cpp)
add_executable(dscal_avx2 test/test_dscal_avx2.cpp)

target_compile_options(ft_dnrm2_avx2 PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
target_include_directories(ft_dnrm2_avx2 PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(ft_dnrm2_avx2 PUBLIC $<LINK_ONLY:MKL::MKL> IncludeFuncs SrcFuncs TestFuncs)

target_compile_options(ft_ddot_avx2 PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
target_include_directories(ft_ddot_avx2 PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(ft_ddot_avx2 PUBLIC $<LINK_ONLY:MKL::MKL> IncludeFuncs SrcFuncs TestFuncs)

target_compile_options(dnrm2_avx2 PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
target_include_directories(dnrm2_avx2 PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(dnrm2_avx2 PUBLIC $<LINK_ONLY:MKL::MKL> IncludeFuncs SrcFuncs TestFuncs)

target_compile_options(ddot_avx2 PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
target_include_directories(ddot_avx2 PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(ddot_avx2 PUBLIC $<LINK_ONLY:MKL::MKL> IncludeFuncs SrcFuncs TestFuncs)

target_compile_options(daxpy_avx2 PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
target_include_directories(daxpy_avx2 PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(daxpy_avx2 PUBLIC $<LINK_ONLY:MKL::MKL> IncludeFuncs SrcFuncs TestFuncs)

target_compile_options(dscal_avx2 PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_COMPILE_OPTIONS>)
target_include_directories(dscal_avx2 PUBLIC $<TARGET_PROPERTY:MKL::MKL,INTERFACE_INCLUDE_DIRECTORIES>)
target_link_libraries(dscal_avx2 PUBLIC $<LINK_ONLY:MKL::MKL> IncludeFuncs SrcFuncs TestFuncs)
