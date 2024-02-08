cmake_minimum_required(VERSION 3.22)
project(da-slam)

set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(GTSAM REQUIRED)
find_package(GTSAM_UNSTABLE REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(range-v3 REQUIRED)

include(cmake/AddOptions.cmake)

include(cmake/FindVisualization.cmake)
if(VISUALIZATION_AVAILABLE)
  add_subdirectory(3rdparty)
  message("VISUALIZATION_AVAILABLE = YES")
  message("VISUALIZATION BACKEND = ${VISUALIZATION_BACKEND}")
else()
  message("VISUALIZATION_AVAILABLE = NO")
endif()

add_subdirectory(src)

if(BUILD_TESTING)
  find_package(GTest REQUIRED)
  include(GoogleTest)
  enable_testing()
  add_subdirectory(test)
endif()

# add_executable(test_auction_method
#   tests/test_auction_method.cpp
# )

# target_link_libraries(test_auction_method
#   Eigen3::Eigen
#   gtsam
#   gtsam_unstable
#   data_association
# )

# endif()

# set_target_properties(test_auction_method PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/tests" )

# add_executable(test_hungarian_method
#   tests/test_hungarian_method.cpp
# )

# target_link_libraries(test_hungarian_method
#   Eigen3::Eigen
#   gtsam
#   gtsam_unstable
#   data_association
# )
  
# if(glog_FOUND)
# target_link_libraries(test_hungarian_method
#   glog::glog
# )
# endif()

# set_target_properties(test_hungarian_method PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/tests" )

# add_executable(test_hypothesis_compare
#   tests/test_hypothesis_compare.cpp
# )

# target_link_libraries(test_hypothesis_compare
#   # Eigen3::Eigen
#   # gtsam
#   # gtsam_unstable
#   data_association
#   hypothesis
# )

# # if(glog_FOUND)
# # target_link_libraries(test_hungarian_method
# #   glog::glog
# # )
# # endif()

# set_target_properties(test_hypothesis_compare PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/tests" )



# if(VISUALIZATION_AVAILABLE)

# add_executable(test_association_visualization
#   tests/test_association_visualization.cpp
# )

# target_link_libraries(test_association_visualization
#   Eigen3::Eigen
#   gtsam
#   gtsam_unstable
#   data_association
#   hypothesis
#   visualization
# )

# if(glog_FOUND)
# target_link_libraries(test_association_visualization
#   glog::glog
# )
# endif()

# set_target_properties(test_association_visualization PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/tests" )

# endif() # VISUALIZATION
# endif() # WITH_TESTS
