set(TARGETS_TO_INSTALL "")

add_library(da_slam
    config.cpp
    argparse.cpp
    data_association/hypothesis.cpp
    data_association/compatibility.cpp
    data_association/innovation.cpp
    data_association/assignment_solvers/auction.cpp
    data_association/assignment_solvers/hungarian.cpp
    data_association/maximum_likelihood.cpp
    data_association/known_data_association.cpp
    data_association/assignment_solvers/auction.cpp
    data_association/assignment_solvers/hungarian.cpp
    slam/slam.cpp
    slam/utils_g2o.cpp
)

target_link_libraries(da_slam PUBLIC
    gtsam
    yaml-cpp
    range-v3::range-v3
    Eigen3::Eigen
    spdlog::spdlog
    fmt::fmt
    magic_enum::magic_enum    
)

target_include_directories(da_slam PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

add_executable(main
  main.cpp
)

target_link_libraries(main
    da_slam
)

list(APPEND TARGETS_TO_INSTALL "main")

# if(VISUALIZATION_AVAILABLE)
#     add_subdirectory(visualization)
    
#     add_executable(main_viz
#       main_viz.cpp
#     )

#     target_link_libraries(main_viz PUBLIC
#         da_slam
#         visualization
#     )

#     target_compile_definitions(main_viz PRIVATE
#         VISUALIZATION_AVAILABLE
#     )

#     list(APPEND TARGETS_TO_INSTALL "main_viz")
# endif() # VISUALIZATION_AVAILABLE

install(TARGETS ${TARGETS_TO_INSTALL}
    DESTINATION ${CMAKE_SOURCE_DIR}
)