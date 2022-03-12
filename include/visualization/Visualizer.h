#pragma once

#include <memory>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <stdio.h>
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <GLES2/gl2.h>
#endif
#include <GLFW/glfw3.h> // Will drag system OpenGL headers

// [Win32] Our example includes a copy of glfw3.lib pre-compiled with VS2010 to maximize ease of testing and compatibility with old VS compilers.
// To link with VS2010-era libraries, VS2015+ requires linking with legacy_stdio_definitions.lib, which we do using this pragma.
// Your own project should not be affected, as you are likely to link with a newer binary of GLFW that is adequate for your version of Visual Studio.
#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

#include <gtsam/nonlinear/NonlinearFactorGraph.h>


namespace visualization {

class Visualizer
{
private:
    GLFWwindow *window_;

public:
    typedef std::shared_ptr<Visualizer> shared_ptr;
    void new_frame() const;
    void render();
    inline bool is_initialized() const { return window_ != NULL; }
    inline bool running() const { return is_initialized() && !glfwWindowShouldClose(window_); };
    void progress_bar(int curr_timestep, int tot_timesteps);
    void draw_factor_graph(const gtsam::NonlinearFactorGraph& graph, const gtsam::Values& estimates);
    Visualizer();
    ~Visualizer();
};

} // namespace visualization