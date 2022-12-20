aux_source_directory(. DIR_LIB_SRCS)
#remember to delete -g when push
set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -O3 -g -march=skylake-avx512 -fopenmp -lpthread -lm")
add_library(SrcFuncs ${DIR_LIB_SRCS})
