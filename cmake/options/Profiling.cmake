option(PROFILING "Poor man's profiling of snippets of code, where added." OFF)

if (PROFILING)
  add_definitions(-DPROFILING)
  message("PROFILING = ON")
else()
  message("PROFILING = OFF")  
endif()