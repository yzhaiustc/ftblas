cmake_minimum_required(VERSION 3.17)

project (ftblas)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_VERBOSE_MAKEFILE ON)

add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(test)


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fopenmp -lpthread -lm")

add_executable(ddot_avx2 test/test_ddot_avx2.cpp)
target_link_libraries(ddot_avx2 IncludeFuncs SrcFuncs TestFuncs)