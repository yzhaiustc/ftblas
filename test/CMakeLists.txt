aux_source_directory(. DIR_LIB_SRCS)
add_library(TestFuncs ${DIR_LIB_SRCS})
target_link_libraries(TestFuncs IncludeFuncs SrcFuncs)