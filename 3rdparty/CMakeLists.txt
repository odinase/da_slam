if(VISUALIZATION_BACKEND_OPENGL)
  set(IMGUI_IMPL_BACKEND_SRC imgui_impl_opengl3.cpp)
  set(VISUALIZATION_BACKEND_LIB OpenGL::OpenGL)
elseif(VISUALIZATION_BACKEND_VULKAN)
  set(IMGUI_IMPL_BACKEND_SRC imgui_impl_vulkan.cpp)
  set(VISUALIZATION_BACKEND_LIB Vulkan::Vulkan)
endif()

add_library(visualization_backend
  imgui/backends/imgui_impl_glfw.cpp
  imgui/backends/${IMGUI_IMPL_BACKEND_SRC}
  imgui/imgui_demo.cpp
  imgui/imgui_draw.cpp
  imgui/imgui_tables.cpp
  imgui/imgui_widgets.cpp
  imgui/imgui.cpp
  implot/implot_demo.cpp
  implot/implot_items.cpp
  implot/implot.cpp
)

target_link_libraries(visualization_backend
    glfw
    ${VISUALIZATION_BACKEND_LIB}
    range-v3::range-v3
)

target_include_directories(visualization_backend PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
