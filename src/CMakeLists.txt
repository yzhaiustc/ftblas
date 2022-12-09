aux_source_directory(. DIR_LIB_SRCS)

set(CMAKE_VERBOSE_MAKEFILE ON)

add_library(SrcFuncs ${DIR_LIB_SRCS})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -g -march=native -lmkl_core -Wall -DMKL_ILP64 -m64 -lmkl_intel_lp64 -lmkl_intel_thread -liomp5 -lpthread -lm")
include_directories(/home/yzhai015/intel/oneapi/mkl/latest/include)
link_directories(/home/yzhai015/intel/oneapi/mkl/latest/lib/intel64)

