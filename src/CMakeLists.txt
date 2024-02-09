set(TARGETS_TO_INSTALL "")

add_library(da_slam
    config.cpp
    data_association/Hypothesis.cpp
    data_association/DataAssociation.cpp
)

target_link_libraries(da_slam PUBLIC
    gtsam
    yaml-cpp
    range-v3::range-v3
    Eigen3::Eigen
    spdlog::spdlog
    magic_enum::magic_enum
)

target_include_directories(da_slam PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

add_executable(slam_g2o_file
  slam/slam_g2o_file.cpp
)

target_link_libraries(slam_g2o_file
    da_slam
)

list(APPEND TARGETS_TO_INSTALL "slam_g2o_file")

if(VISUALIZATION_AVAILABLE)
    add_subdirectory(visualization)
    
    add_executable(slam_g2o_file_visualization
      slam/slam_g2o_file_visualization.cpp
    )

    target_link_libraries(slam_g2o_file_visualization PUBLIC
        da_slam
        visualization
    )

    target_compile_definitions(slam_g2o_file_visualization PRIVATE
        VISUALIZATION_AVAILABLE
    )

    list(APPEND TARGETS_TO_INSTALL "slam_g2o_file_visualization")
endif() # VISUALIZATION_AVAILABLE

install(TARGETS ${TARGETS_TO_INSTALL}
    DESTINATION ${CMAKE_SOURCE_DIR}
)