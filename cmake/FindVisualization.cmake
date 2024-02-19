set(VISUALIZATION_BACKEND "OPENGL" CACHE STRING "Visualization backend to use")

if(VISUALIZATION_BACKEND STREQUAL "OPENGL")
    set(OpenGL_GL_PREFERENCE GLVND)
    find_package(OpenGL 3)
    set(VISUALIZATION_BACKEND_OPENGL TRUE)
    set(VISUALIZATION_BACKEND_VULKAN FALSE)
elseif(VISUALIZATION_BACKEND STREQUAL "VULKAN")
    find_package(Vulkan REQUIRED)
    set(VISUALIZATION_BACKEND_OPENGL FALSE)
    set(VISUALIZATION_BACKEND_VULKAN TRUE)
endif()

if(VISUALIZATION_BACKEND_OPENGL)
set(VIZ_FOUND OpenGL_FOUND)
elseif(VISUALIZATION_BACKEND_VULKAN)
set(VIZ_FOUND Vulkan_FOUND)
endif()

if(VIZ_FOUND)
    find_package(glfw3)
    set(VISUALIZATION_AVAILABLE TRUE)
else()
    set(VISUALIZATION_AVAILABLE FALSE)
endif()