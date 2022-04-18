cmake_minimum_required(VERSION 3.10)
project(da-slam)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)


find_package(GTSAM REQUIRED) # Uses installed package
find_package(OpenCV 3 REQUIRED)
find_package(Eigen3 REQUIRED)


find_package(glfw3)
set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL 3)


find_package (glog 0.6.0)


include_directories(
  eigen3
  include
  ${EIGEN3_INCLUDE_DIR}
)

if(OpenGL_FOUND AND glfw3_FOUND)
set(VISUALIZATION_AVAILABLE TRUE)
else()
set(VISUALIZATION_AVAILABLE FALSE)
endif()

if(VISUALIZATION_AVAILABLE)
include_directories(
  3rdparty/imgui
  3rdparty/implot
  3rdparty/imgui/backends
  ${OPENGL_INCLUDE_DIRS}
)
endif()

if (glog_FOUND)
add_definitions(-DGLOG_AVAILABLE)
message("Building with glog")
else()
message("Building without glog")
endif()

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

option(WITH_TESTS "Compile poor man's testing suite" OFF)

if (WITH_TESTS)
  message("WITH_TESTS = ON")
else()
  message("WITH_TESTS = OFF")  
endif()


if(VISUALIZATION_AVAILABLE)
message("VISUALIZATION_AVAILABLE = YES")
else()
message("VISUALIZATION_AVAILABLE = NO")
endif()

### Libraries


add_library(config
  src/config.cpp
)

target_link_libraries(config
  ${OpenCV_LIBS}
)

add_library(hypothesis
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

if(VISUALIZATION_AVAILABLE)
add_library(imgui_backend
    3rdparty/imgui/backends/imgui_impl_glfw.cpp
    3rdparty/imgui/backends/imgui_impl_opengl3.cpp
)

add_library(imgui
    3rdparty/imgui/imgui_demo.cpp
    3rdparty/imgui/imgui_draw.cpp
    3rdparty/imgui/imgui_tables.cpp
    3rdparty/imgui/imgui_widgets.cpp
    3rdparty/imgui/imgui.cpp
)

target_link_libraries(imgui
    imgui_backend
)

add_library(implot
    3rdparty/implot/implot_demo.cpp
    3rdparty/implot/implot_items.cpp
    3rdparty/implot/implot.cpp
)

target_link_libraries(implot
    imgui_backend
    imgui
)

add_library(visualization
    src/visualization.cpp
)

target_link_libraries(visualization
    imgui_backend
    imgui
    implot
    glfw
    ${OPENGL_LIBRARIES}
    gtsam_unstable
    gtsam
    Eigen3::Eigen
    config
    data_association
)


add_executable(slam_g2o_file_visualization
  src/slam/slam_g2o_file_visualization.cpp
)

target_link_libraries(slam_g2o_file_visualization
  Eigen3::Eigen
  gtsam
  gtsam_unstable
  hypothesis
  data_association
  config
  visualization
)

if(glog_FOUND)
target_link_libraries(slam_g2o_file_visualization
  glog::glog
)
endif()

set_target_properties(slam_g2o_file_visualization PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR} )
endif()

add_executable(slam_g2o_file
  src/slam/slam_g2o_file.cpp
)

target_link_libraries(slam_g2o_file
  Eigen3::Eigen
  gtsam
  gtsam_unstable
  hypothesis
  data_association
  config
)

if(glog_FOUND)
target_link_libraries(slam_g2o_file
  glog::glog
)
endif()


set_target_properties(slam_g2o_file PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR} )


# Not the correct way of doing this, but whatever

if(WITH_TESTS)

add_executable(test_auction_method
  tests/test_auction_method.cpp
)

target_link_libraries(test_auction_method
  Eigen3::Eigen
  gtsam
  gtsam_unstable
  data_association
)
  
if(glog_FOUND)
target_link_libraries(test_auction_method
  glog::glog
)
endif()

set_target_properties(test_auction_method PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/tests" )


add_executable(test_hungarian_method
  tests/test_hungarian_method.cpp
)

target_link_libraries(test_hungarian_method
  Eigen3::Eigen
  gtsam
  gtsam_unstable
  data_association
)
  
if(glog_FOUND)
target_link_libraries(test_hungarian_method
  glog::glog
)
endif()

set_target_properties(test_hungarian_method PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/tests" )

if(VISUALIZATION_AVAILABLE)
add_executable(test_association_visualization
  tests/test_association_visualization.cpp
)

target_link_libraries(test_association_visualization
  Eigen3::Eigen
  gtsam
  gtsam_unstable
  data_association
  hypothesis
  visualization
  )

if(glog_FOUND)
target_link_libraries(test_association_visualization
  glog::glog  
)
endif()

set_target_properties(test_association_visualization PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/tests" )


add_executable(test_read_ground_truth_assos
  tests/test_read_ground_truth_assos.cpp
)

target_link_libraries(test_read_ground_truth_assos
  Eigen3::Eigen
  gtsam
  gtsam_unstable
  data_association
  hypothesis
  visualization
)
  
if(glog_FOUND)
target_link_libraries(test_read_ground_truth_assos
  glog::glog
)
endif()

set_target_properties(test_read_ground_truth_assos PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/tests" )
endif() # VISUALIZATION
endif() # WITH_TESTS