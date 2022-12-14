file(GLOB headers *.h)
add_library(IncludeFuncs INTERFACE)
target_sources(IncludeFuncs INTERFACE ${headers})
