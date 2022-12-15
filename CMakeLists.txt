cmake_minimum_required(VERSION 3.17)

project (ftblas)

set(CMAKE_CXX_STANDARD 11)

add_subdirectory(include)
add_subdirectory(src)
add_subdirectory(test)

option(USE_FAULT_TOLERANT  OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -O3 -march=skylake-avx512 -fopenmp -lpthread -lm")

if(USE_FAULT_TOLERANT)
    message("Enable the fault tolerant functionality")
    target_compile_definitions(SrcFuncs PRIVATE FT_ENABLED=1)
endif(USE_FAULT_TOLERANT)

add_executable(ddot test/test_ddot.cpp)
target_link_libraries(ddot IncludeFuncs SrcFuncs TestFuncs)

add_executable(dgemm test/test_dgemm.cpp)
target_link_libraries(dgemm IncludeFuncs SrcFuncs TestFuncs)

if(USE_FAULT_TOLERANT)
    target_compile_definitions(ddot PRIVATE FT_ENABLED=1)
    target_compile_definitions(dgemm PRIVATE FT_ENABLED=1)
endif(USE_FAULT_TOLERANT)
