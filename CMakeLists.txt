cmake_minimum_required(VERSION 3.10)
project(da-slam)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
# set(CMAKE_CXX_FLAGS "${CMAKE_C_FLAGS} -target x86_64-apple-macos12.4")


find_package(GTSAM REQUIRED) # Uses installed package
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(range-v3 REQUIRED)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -target x86_64-apple-darwin20.3.0")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -ggdb")


find_package(glfw3)

SET(VISUALIZATION_BACKEND "OPENGL" CACHE STRING "Visualization backend to use")

if (VISUALIZATION_BACKEND STREQUAL "OPENGL")

message("Using visualization backend OpenGL")
set(OpenGL_GL_PREFERENCE GLVND)
find_package(OpenGL 3)
set(VISUALIZATION_BACKEND_OPENGL TRUE)
set(VISUALIZATION_BACKEND_VULKAN FALSE)

elseif(VISUALIZATION_BACKEND STREQUAL "VULKAN")
message("Using visualization backend Vulkan")

find_package(Vulkan REQUIRED)
set(VISUALIZATION_BACKEND_OPENGL FALSE)
set(VISUALIZATION_BACKEND_VULKAN TRUE)

endif()

find_package (glog 0.6.0)


include_directories(
  eigen3
  include
  ${EIGEN3_INCLUDE_DIR}
)

if(VISUALIZATION_BACKEND_OPENGL)
set(VIZ_FOUND OpenGL_FOUND)
elseif(VISUALIZATION_BACKEND_VULKAN)
set(VIZ_FOUND Vulkan_FOUND)
endif()

if(VIZ_FOUND AND glfw3_FOUND)
set(VISUALIZATION_AVAILABLE TRUE)
else()
set(VISUALIZATION_AVAILABLE FALSE)
endif()


if(VISUALIZATION_AVAILABLE)
include_directories(
  3rdparty/imgui
  3rdparty/implot
  3rdparty/imgui/backends
)

if (VISUALIZATION_BACKEND_OPENGL)
include_directories(
  ${OPENGL_INCLUDE_DIRS} 
)
endif()
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
message("VISUALIZATION BACKEND = ${VISUALIZATION_BACKEND}")
else()
message("VISUALIZATION_AVAILABLE = NO")
endif()

### Libraries


add_library(config
  src/config.cpp
)

target_link_libraries(config
  yaml-cpp
  gtsam
  range-v3::range-v3
)

add_library(hypothesis
  src/data_association/Hypothesis.cpp
)

target_link_libraries(hypothesis
  Eigen3::Eigen
  gtsam
  range-v3::range-v3
)

add_library(data_association
  src/data_association/DataAssociation.cpp
)

target_link_libraries(data_association
  Eigen3::Eigen
  hypothesis
  gtsam
  range-v3::range-v3
)

if(VISUALIZATION_AVAILABLE)
if(VISUALIZATION_BACKEND_OPENGL)
add_library(imgui_backend
    3rdparty/imgui/backends/imgui_impl_glfw.cpp
    3rdparty/imgui/backends/imgui_impl_opengl3.cpp
)
elseif(VISUALIZATION_BACKEND_VULKAN)
add_library(imgui_backend
    3rdparty/imgui/backends/imgui_impl_glfw.cpp
    3rdparty/imgui/backends/imgui_impl_vulkan.cpp
)
endif()

target_link_libraries(imgui_backend
    glfw
    range-v3::range-v3
)
if (VISUALIZATION_BACKEND_OPENGL)
target_link_libraries(imgui_backend
  ${OPENGL_LIBRARIES}
)
elseif(VISUALIZATION_BACKEND_VULKAN)
target_link_libraries(imgui_backend
  Vulkan::Vulkan
  range-v3::range-v3
)
endif()
    
add_library(imgui
    3rdparty/imgui/imgui_demo.cpp
    3rdparty/imgui/imgui_draw.cpp
    3rdparty/imgui/imgui_tables.cpp
    3rdparty/imgui/imgui_widgets.cpp
    3rdparty/imgui/imgui.cpp
)

target_link_libraries(imgui
    imgui_backend
    range-v3::range-v3
)

add_library(implot
    3rdparty/implot/implot_demo.cpp
    3rdparty/implot/implot_items.cpp
    3rdparty/implot/implot.cpp
)

target_link_libraries(implot
    imgui_backend
    imgui
    range-v3::range-v3
)

if (VISUALIZATION_BACKEND_OPENGL)
add_library(visualization
    src/drawing.cpp
    src/visualization_opengl.cpp
)
elseif(VISUALIZATION_BACKEND_VULKAN)
add_library(visualization
    src/drawing.cpp
    src/visualization_vulkan.cpp
)
endif()

target_link_libraries(visualization
    imgui_backend
    imgui
    implot
    gtsam_unstable
    gtsam
    Eigen3::Eigen
    config
    data_association
    glfw
    range-v3::range-v3
)

if (VISUALIZATION_BACKEND_OPENGL)
target_link_libraries(visualization
  ${OPENGL_LIBRARIES}
)
elseif(VISUALIZATION_BACKEND_VULKAN)
target_link_libraries(visualization
  Vulkan::Vulkan
)
endif()


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
  imgui_backend
  imgui
  implot
  glfw
  ${OPENGL_LIBRARIES}
  range-v3::range-v3
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
  range-v3::range-v3
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

add_executable(test_hypothesis_compare
  tests/test_hypothesis_compare.cpp
)

target_link_libraries(test_hypothesis_compare
  # Eigen3::Eigen
  # gtsam
  # gtsam_unstable
  data_association
  hypothesis
)
  
# if(glog_FOUND)
# target_link_libraries(test_hungarian_method
#   glog::glog
# )
# endif()

set_target_properties(test_hypothesis_compare PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/tests" )



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

endif() # VISUALIZATION
endif() # WITH_TESTS
