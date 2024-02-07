cmake_minimum_required(VERSION 3.10)
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

add_subdirectory(cmake)

option(WITH_TESTS "Compile poor man's testing suite" OFF)

if (WITH_TESTS)
message("WITH_TESTS = ON")
else()
message("WITH_TESTS = OFF")  
endif()

if(VISUALIZATION_AVAILABLE)
  add_subdirectory(3rdparty)
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
