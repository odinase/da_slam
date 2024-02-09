option(BUILD_SANDBOX "Compile sandbox code for testing" OFF)

if (BUILD_SANDBOX)
message("BUILD_SANDBOX = ON")
else()
message("BUILD_SANDBOX = OFF")  

endif()
