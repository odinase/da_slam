
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

if(VISUALIZATION_BACKEND_OPENGL)
  set(BACKEND_SRC imgui/backends/imgui_impl_opengl3.cpp)
elseif(VISUALIZATION_BACKEND_VULKAN)
  set(BACKEND_SRC imgui/backends/imgui_impl_vulkan.cpp)
endif()

add_library(imgui_backend
    imgui/backends/imgui_impl_glfw.cpp
    ${BACKEND_SRC}
)