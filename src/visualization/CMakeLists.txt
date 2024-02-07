if(VISUALIZATION_BACKEND_OPENGL)
    set(VISUALIZATION_SRC visualization_opengl.cpp)
elseif(VISUALIZATION_BACKEND_VULKAN)
    set(VISUALIZATION_SRC visualization_vulkan.cpp)
endif()

add_library(visualization
    drawing.cpp
    ${VISUALIZATION_SRC}
)

target_link_libraries(visualization PUBLIC
    gtsam_unstable
    gtsam
    Eigen3::Eigen
    range-v3::range-v3
    da_slam
    visualization_backend
)
