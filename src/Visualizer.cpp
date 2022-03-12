#include "imgui.h"
#include "imgui_internal.h"
#include "implot.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <stdio.h>
#include <cmath>
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

#include "visualization/Visualizer.h"
#include "visualization/colors.h"

#include <gtsam_unstable/slam/PoseToPointFactor.h>
#include <gtsam/slam/BetweenFactor.h>

static void glfw_error_callback(int error, const char *description)
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

namespace ImGui
{
    bool BufferingBar(const char *label, float value, const ImVec2 &size_arg, const ImU32 &bg_col, const ImU32 &fg_col)
    {
        ImGuiWindow *window = GetCurrentWindow();
        if (window->SkipItems)
            return false;

        ImGuiContext &g = *GImGui;
        const ImGuiStyle &style = g.Style;
        const ImGuiID id = window->GetID(label);

        ImVec2 pos = window->DC.CursorPos;
        ImVec2 size = size_arg;
        size.x -= style.FramePadding.x * 2;

        const ImRect bb(pos, ImVec2(pos.x + size.x, pos.y + size.y));
        ItemSize(bb, style.FramePadding.y);
        if (!ItemAdd(bb, id))
            return false;

        // Render
        const float circleStart = size.x * 0.7f;
        const float circleEnd = size.x;
        const float circleWidth = circleEnd - circleStart;

        window->DrawList->AddRectFilled(bb.Min, ImVec2(pos.x + circleStart, bb.Max.y), bg_col);
        window->DrawList->AddRectFilled(bb.Min, ImVec2(pos.x + circleStart * value, bb.Max.y), fg_col);

        return true;
    }
}

namespace visualization
{

    void Visualizer::new_frame() const
    {
        // Poll and handle events (inputs, window resize, etc.)
        // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
        // - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application, or clear/overwrite your copy of the mouse data.
        // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application, or clear/overwrite your copy of the keyboard data.
        // Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
    }

    void Visualizer::render()
    {
        using namespace colors;
        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window_, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(CLEAR.x * CLEAR.w, CLEAR.y * CLEAR.w, CLEAR.z * CLEAR.w, CLEAR.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window_);
    }

    void Visualizer::progress_bar(int curr_timestep, int tot_timesteps)
    {
        static const ImU32 col = ImGui::GetColorU32(ImGuiCol_ButtonHovered);
        static const ImU32 bg = ImGui::GetColorU32(ImGuiCol_Button);

        float progress = float(curr_timestep) / tot_timesteps;
        ImGui::BufferingBar("##buffer_bar", progress, ImVec2(400, 6), bg, col);
        // int num_digits = floor(log10(tot_timesteps));
        ImGui::Text("Processed %d / %d timesteps, %.2f%% complete", curr_timestep, tot_timesteps, progress * 100.0);
    }

    Visualizer::Visualizer()
    {
        window_ = NULL;
        // Setup window
        glfwSetErrorCallback(glfw_error_callback);
        if (!glfwInit())
            return;

            // Decide GL+GLSL versions
#if defined(IMGUI_IMPL_OPENGL_ES2)
        // GL ES 2.0 + GLSL 100
        const char *glsl_version = "#version 100";
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
        glfwWindowHint(GLFW_CLIENT_API, GLFW_OPENGL_ES_API);
#elif defined(__APPLE__)
        // GL 3.2 + GLSL 150
        const char *glsl_version = "#version 150";
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // 3.2+ only
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);           // Required on Mac
#else
        // GL 3.0 + GLSL 130
        const char *glsl_version = "#version 130";
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
        // glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);  // 3.2+ only
        // glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);            // 3.0+ only
#endif

        // Create window with graphics context
        window_ = glfwCreateWindow(1280, 720, "Data Association SLAM - Visualization", NULL, NULL);
        if (window_ == NULL)
            return;
        glfwMakeContextCurrent(window_);
        glfwSwapInterval(1); // Enable vsync

        // Setup Dear ImGui context
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImPlot::CreateContext();

        // io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
        // io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

        // Setup Dear ImGui style
        ImGui::StyleColorsDark();
        // ImGui::StyleColorsClassic();

        // Setup Platform/Renderer backends
        ImGui_ImplGlfw_InitForOpenGL(window_, true);
        ImGui_ImplOpenGL3_Init(glsl_version);
    }

    Visualizer::~Visualizer()
    {
        // Cleanup
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImPlot::DestroyContext();
        ImGui::DestroyContext();

        glfwDestroyWindow(window_);
        glfwTerminate();
    }

    // Assumes that factors are PoseToPoint 2D, robot is pose and landmarks are points
    void Visualizer::draw_factor_graph(const gtsam::NonlinearFactorGraph &graph, const gtsam::Values &estimates)
    {
        ImGui::Begin("Factor graph");
        if (ImPlot::BeginPlot("##factor graph", ImVec2(-1, -1)))
        {
            gtsam::PoseToPointFactor<gtsam::Pose2, gtsam::Point2>::shared_ptr meas;
            gtsam::BetweenFactor<gtsam::Pose2>::shared_ptr odom;
            for (const auto &factor_ : graph)
            { // inspired by dataset.cpp/writeG2o
                meas = boost::dynamic_pointer_cast<gtsam::PoseToPointFactor<gtsam::Pose2, gtsam::Point2>>(factor_);
                odom = boost::dynamic_pointer_cast<gtsam::BetweenFactor<gtsam::Pose2>>(factor_);
                double line[4]; // = {0.0, 0.0, 0.0, 0.0};
                if (meas)
                {
                    gtsam::Pose2 x = estimates.at<gtsam::Pose2>(meas->key1());
                    gtsam::Point2 l = estimates.at<gtsam::Point2>(meas->key2());

                    line[0] = x.x();
                    line[1] = l.x();

                    line[2] = x.y();
                    line[3] = l.y();

                    ImPlot::PlotLine("Measurement", line, line + 2, 2);
                    ImPlot::SetNextMarkerStyle(ImPlotMarker_Diamond, 5.0, ImVec4(119.0 / 255.0, 100.0 / 255.0, 182.0 / 255.0, 1.0));
                    ImPlot::PlotScatter("Landmark", &l.x(), &l.y(), 1);
                }
                if (odom)
                {
                    gtsam::Pose2 x_from = estimates.at<gtsam::Pose2>(odom->key1());
                    gtsam::Pose2 x_to = estimates.at<gtsam::Pose2>(odom->key2());

                    line[0] = x_from.x();
                    line[1] = x_to.x();

                    line[2] = x_from.y();
                    line[3] = x_to.y();

                    ImPlot::PlotLine("Odometry", line, line + 2, 2);
                    ImPlot::SetNextMarkerStyle(ImPlotMarker_Diamond, 5.0, ImVec4(19.0 / 255.0, 160.0 / 255.0, 17.0 / 255.0, 1.0));
                    ImPlot::PlotScatter("Poses", line, line + 2, 2);
                }
            }
            ImPlot::EndPlot();
        }
        ImGui::End();
    }

} // namespace visualization