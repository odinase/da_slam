cmake_minimum_required(VERSION 3.10)
project(slam)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

find_package(GTSAM REQUIRED) # Uses installed package
# include_directories(${GTSAM_INCLUDE_DIR})

find_package(Eigen3 REQUIRED)
# set(EIGEN3_INCLUDE_DIR "/usr/local/include/gtsam/3rdparty/Eigen")
include_directories(${EIGEN3_INCLUDE_DIR})

find_package (glog 0.6.0 REQUIRED)

include_directories(
  eigen3
  include
)

### Options


option(HYPOTHESIS_QUALITY "Compute NIS for joint hypothesis for comparision of hypothesis quality" OFF)

if (HYPOTHESIS_QUALITY)
  add_definitions(-DHYPOTHESIS_QUALITY)
  message("HYPOTHESIS_QUALITY = ON")
else()
  message("HYPOTHESIS_QUALITY = OFF")
endif()

option(PROFILING "Poor man's profiling of snippets of code, where added." OFF)

if (PROFILING)
  add_definitions(-DPROFILING)
  message("PROFILING = ON")
else()
  message("PROFILING = OFF")  
endif()

option(LOGGING "Log statistics about where the SLAM system is, what it's doing and results" OFF)

if (LOGGING)
  add_definitions(-DLOGGING)
  message("LOGGING = ON")
else()
  message("LOGGING = OFF")  
endif()

option(HEARTBEAT "Print periodically how far the SLAM system has come through the dataset" ON)

if (HEARTBEAT)
  add_definitions(-DHEARTBEAT)
  message("HEARTBEAT = ON")
else()
  message("HEARTBEAT = OFF")  
endif()


### Libraries


add_library(hypothesis
  # src/jcbb/MarginalMocks.cpp
  src/data_association/Hypothesis.cpp
)

target_link_libraries(hypothesis
  Eigen3::Eigen
  gtsam
)


add_library(data_association
  src/data_association/DataAssociation.cpp
)

target_link_libraries(data_association
  Eigen3::Eigen
)

add_executable(slam_g2o_file
  src/slam/slam_g2o_file.cpp
)

target_link_libraries(slam_g2o_file
  Eigen3::Eigen
  gtsam
  gtsam_unstable
  hypothesis
  glog::glog
  data_association
)

set_target_properties(slam_g2o_file PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR} )

# Not the correct way of doing this, but whatever

add_executable(test_auction_method
  tests/test_auction_method.cpp
)

target_link_libraries(test_auction_method
  Eigen3::Eigen
  gtsam
  gtsam_unstable
  glog::glog
  data_association
)

set_target_properties(test_auction_method PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/tests" )


add_executable(test_hungarian_method
  tests/test_hungarian_method.cpp
)

target_link_libraries(test_hungarian_method
  Eigen3::Eigen
  gtsam
  gtsam_unstable
  glog::glog
  data_association
)

set_target_properties(test_hungarian_method PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/tests" )
