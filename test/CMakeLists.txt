aux_source_directory(. DIR_LIB_SRCS)

add_library(TestFuncs ${DIR_LIB_SRCS})


include_directories(/home/yzhai015/intel/oneapi/mkl/latest/include)
target_link_libraries(TestFuncs IncludeFuncs SrcFuncs)
link_directories(/home/yzhai015/intel/oneapi/mkl/latest/lib/intel64)

