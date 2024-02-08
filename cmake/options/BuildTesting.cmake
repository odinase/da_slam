option(BUILD_TESTING "Compile testing suite" OFF)

if (BUILD_TESTING)
message("BUILD_TESTING = ON")
else()
message("BUILD_TESTING = OFF")  

endif()
