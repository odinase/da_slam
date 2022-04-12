#include "imgui.h"
#include "imgui_internal.h"
#include "implot.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <stdio.h>
#include <cmath>
#include <sstream>
#include <string>
#include <exception>
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

#include "visualization/visualization.h"
#include "visualization/colors.h"

#include <gtsam_unstable/slam/PoseToPointFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/Marginals.h>
#include <Eigen/Core>
#include <Eigen/Cholesky>

static void glfw_error_callback(int error, const char *description)
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}

Eigen::MatrixXd ellipse(const Eigen::Vector2d &mu, const Eigen::Matrix2d &P, const double s = 1.0, const int n = 200)
{
    Eigen::RowVectorXd thetas = Eigen::RowVectorXd::LinSpaced(n, 0, 2.0 * M_PI);
    Eigen::MatrixXd points(2, n);
    points << thetas.array().cos(),
        thetas.array().sin();
    Eigen::LLT<Eigen::Matrix2d> chol(P);
    Eigen::MatrixXd ell = (s * chol.matrixL().toDenseMatrix() * points).colwise() + mu;
    return ell;
}

Eigen::MatrixXd circle(const Eigen::Vector2d &center, const double r = 1.0, const int n = 200)
{
    Eigen::RowVectorXd thetas = Eigen::RowVectorXd::LinSpaced(n, 0, 2.0 * M_PI);
    Eigen::MatrixXd points(2, n);
    points << thetas.array().cos(),
              thetas.array().sin();
    Eigen::MatrixXd circ = points.colwise() + center;
    return circ;
}

// Eigen::MatrixXd ellipse(const Eigen::Vector3d &mu, const Eigen::Matrix3d &P, const double s = 1.0, const int n = 200)
// {
//     Eigen::RowVectorXd thetas = Eigen::RowVectorXd::LinSpaced(n, 0, 2.0 * M_PI);
//     Eigen::MatrixXd points(2, n);
//     points << thetas.array().cos(),
//         thetas.array().sin();
//     Eigen::LLT<Eigen::Matrix2d> chol(P);
//     Eigen::MatrixXd ell = (s * chol.matrixL().toDenseMatrix() * points).colwise() + mu;
//     return ell;
// }

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
    static GLFWwindow *WINDOW = NULL;

    void new_frame()
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

    void render()
    {
        using namespace colors;
        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(WINDOW, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(CLEAR.x * CLEAR.w, CLEAR.y * CLEAR.w, CLEAR.z * CLEAR.w, CLEAR.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(WINDOW);
    }

    void progress_bar(int curr_timestep, int tot_timesteps)
    {
        static const ImU32 col = ImGui::GetColorU32(ImGuiCol_ButtonHovered);
        static const ImU32 bg = ImGui::GetColorU32(ImGuiCol_Button);

        float progress = float(curr_timestep) / tot_timesteps;
        ImGui::BufferingBar("##buffer_bar", progress, ImVec2(400, 6), bg, col);
        // int num_digits = floor(log10(tot_timesteps));
        ImGui::Text("Processed %d / %d timesteps, %.2f%% complete", curr_timestep, tot_timesteps, progress * 100.0);
    }

    bool running()
    {
        return WINDOW != NULL && !glfwWindowShouldClose(WINDOW);
    }

    bool init()
    {
        // Setup window
        glfwSetErrorCallback(glfw_error_callback);
        if (!glfwInit())
            return false;

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
        WINDOW = glfwCreateWindow(1280, 720, "Data Association SLAM - Visualization", NULL, NULL);
        if (WINDOW == NULL)
            return false;
        glfwMakeContextCurrent(WINDOW);
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
        ImGui_ImplGlfw_InitForOpenGL(WINDOW, true);
        ImGui_ImplOpenGL3_Init(glsl_version);

        return true;
    }

    void shutdown()
    {
        // Cleanup
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImPlot::DestroyContext();
        ImGui::DestroyContext();

        glfwDestroyWindow(WINDOW);
        glfwTerminate();
    }

    void draw_factor_graph(const gtsam::NonlinearFactorGraph &graph, const gtsam::Values &estimates, int latest_time_step)
    {
        latest_time_step = std::max(latest_time_step, 0);

        gtsam::PoseToPointFactor<gtsam::Pose2, gtsam::Point2>::shared_ptr meas2d;
        gtsam::BetweenFactor<gtsam::Pose2>::shared_ptr odom2d;

        gtsam::PoseToPointFactor<gtsam::Pose3, gtsam::Point3>::shared_ptr meas3d;
        gtsam::BetweenFactor<gtsam::Pose3>::shared_ptr odom3d;

        std::stringstream ss;
        std::string legend;

        // Add all values first

        double line[4];
        double xp, yp;
        gtsam::KeySet value_keys;
        gtsam::KeySet lmks_to_draw;

        // Draw just factors and add keys that have an associated factor
        for (const auto &factor_ : graph)
        {
            meas2d = boost::dynamic_pointer_cast<gtsam::PoseToPointFactor<gtsam::Pose2, gtsam::Point2>>(factor_);
            odom2d = boost::dynamic_pointer_cast<gtsam::BetweenFactor<gtsam::Pose2>>(factor_);
            meas3d = boost::dynamic_pointer_cast<gtsam::PoseToPointFactor<gtsam::Pose3, gtsam::Point3>>(factor_);
            odom3d = boost::dynamic_pointer_cast<gtsam::BetweenFactor<gtsam::Pose3>>(factor_);

            if (meas2d)
            {
                gtsam::Key x_key = meas2d->key1();
                if (gtsam::symbolIndex(x_key) < latest_time_step) {
                    continue;
                }
                gtsam::Key lmk_key = meas2d->key2();
                lmks_to_draw.insert(lmk_key);

                value_keys.insert(x_key);
                value_keys.insert(lmk_key);

                gtsam::Pose2 x = estimates.at<gtsam::Pose2>(x_key);
                gtsam::Point2 l = estimates.at<gtsam::Point2>(lmk_key);

                line[0] = x.x();
                line[1] = l.x();

                line[2] = x.y();
                line[3] = l.y();

                ImPlot::PlotLine("Measurement", line, line + 2, 2);
            }
            if (odom2d)
            {
                gtsam::Key x_from_key = odom2d->key1();
                if (gtsam::symbolIndex(x_from_key) < latest_time_step) {
                    continue;
                }
                gtsam::Key x_to_key = odom2d->key2();

                gtsam::Pose2 x_from = estimates.at<gtsam::Pose2>(x_from_key);
                gtsam::Pose2 x_to = estimates.at<gtsam::Pose2>(x_to_key);

                value_keys.insert(x_from_key);
                value_keys.insert(x_to_key);

                line[0] = x_from.x();
                line[1] = x_to.x();

                line[2] = x_from.y();
                line[3] = x_to.y();

                ImPlot::PlotLine("Odometry", line, line + 2, 2);
            }
            if (meas3d)
            {
                gtsam::Key x_key = meas3d->key1();
                if (gtsam::symbolIndex(x_key) < latest_time_step) {
                    continue;
                }
                gtsam::Key lmk_key = meas3d->key2();
                lmks_to_draw.insert(lmk_key);

                value_keys.insert(x_key);
                value_keys.insert(lmk_key);

                gtsam::Pose3 x = estimates.at<gtsam::Pose3>(x_key);
                gtsam::Point3 l = estimates.at<gtsam::Point3>(lmk_key);

                line[0] = x.x();
                line[1] = l.x();

                line[2] = x.y();
                line[3] = l.y();

                ImPlot::PlotLine("Measurement", line, line + 2, 2);
            }
            if (odom3d)
            {
                gtsam::Key x_from_key = odom3d->key1();
                if (gtsam::symbolIndex(x_from_key) < latest_time_step) {
                    continue;
                }
                gtsam::Key x_to_key = odom3d->key2();

                gtsam::Pose3 x_from = estimates.at<gtsam::Pose3>(x_from_key);
                gtsam::Pose3 x_to = estimates.at<gtsam::Pose3>(x_to_key);

                value_keys.insert(x_from_key);
                value_keys.insert(x_to_key);

                line[0] = x_from.x();
                line[1] = x_to.x();

                line[2] = x_from.y();
                line[3] = x_to.y();

                ImPlot::PlotLine("Odometry", line, line + 2, 2);
            }
        }

        for (const gtsam::Values::ConstKeyValuePair p : estimates)
        {
            unsigned char chr;
            uint64_t idx;
            if (gtsam::symbolChr(p.key) == 'x' || gtsam::symbolChr(p.key) == '\0')
            {
                chr = 'x';
                idx = gtsam::symbolIndex(p.key);
                if (idx < latest_time_step) {
                    continue;
                }
                try
                {
                    gtsam::Pose2 x = p.value.cast<gtsam::Pose2>();
                    xp = x.x();
                    yp = x.y();
                }
                catch (const std::bad_cast &e)
                {
                    gtsam::Pose3 x = p.value.cast<gtsam::Pose3>();
                    xp = x.x();
                    yp = x.y();
                }
                legend = "Poses";
                ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 5.0, ImVec4(19.0 / 255.0, 160.0 / 255.0, 17.0 / 255.0, 1.0));
            }
            else if (gtsam::symbolChr(p.key) == 'l')
            {
                chr = 'l';
                idx = gtsam::symbolIndex(p.key);
                if (!lmks_to_draw.exists(p.key)) {
                    continue;
                }
                try
                {
                    gtsam::Point2 l = p.value.cast<gtsam::Point2>();
                    xp = l.x();
                    yp = l.y();
                }
                catch (const std::bad_cast &e)
                {
                    gtsam::Point3 l = p.value.cast<gtsam::Point3>();
                    xp = l.x();
                    yp = l.y();
                }
                legend = "Landmarks";
                ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 5.0, ImVec4(119.0 / 255.0, 100.0 / 255.0, 182.0 / 255.0, 1.0));
            }
            else
            {
                std::cerr << "Received key " << gtsam::Symbol(p.key) << " which could not be parsed, skipping\n";
                std::cerr << "Chr: \"" << gtsam::symbolChr(p.key) << "\"\nIndex: " << gtsam::symbolIndex(p.key) << "\n";
                continue; // Should never happen??
            }

            if (!value_keys.exists(p.key))
            { // Found value with no factor attached to it
                ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 10.0, ImVec4(1.0, 0.0, 0.0, 1.0));
            }

            ImPlot::PlotScatter(legend.c_str(), &xp, &yp, 1);
            ss << chr << idx;
            ImPlot::PlotText(ss.str().c_str(), xp, yp, false, ImVec2(15, 15));
            ss.str("");
        }

    }


    void draw_factor_graph_ground_truth(const gtsam::NonlinearFactorGraph &graph, const gtsam::Values &estimates, int latest_time_step)
    {
        latest_time_step = std::max(latest_time_step, 0);

        gtsam::PoseToPointFactor<gtsam::Pose2, gtsam::Point2>::shared_ptr meas2d;
        gtsam::BetweenFactor<gtsam::Pose2>::shared_ptr odom2d;

        gtsam::PoseToPointFactor<gtsam::Pose3, gtsam::Point3>::shared_ptr meas3d;
        gtsam::BetweenFactor<gtsam::Pose3>::shared_ptr odom3d;

        std::stringstream ss;
        std::string legend;

        // Add all values first

        double line[4];
        double xp, yp;
        gtsam::KeySet value_keys;
        gtsam::KeySet lmks_to_draw;

        // Draw just factors and add keys that have an associated factor
        for (const auto &factor_ : graph)
        {
            meas2d = boost::dynamic_pointer_cast<gtsam::PoseToPointFactor<gtsam::Pose2, gtsam::Point2>>(factor_);
            odom2d = boost::dynamic_pointer_cast<gtsam::BetweenFactor<gtsam::Pose2>>(factor_);
            meas3d = boost::dynamic_pointer_cast<gtsam::PoseToPointFactor<gtsam::Pose3, gtsam::Point3>>(factor_);
            odom3d = boost::dynamic_pointer_cast<gtsam::BetweenFactor<gtsam::Pose3>>(factor_);

            if (meas2d)
            {
                gtsam::Key x_key = meas2d->key1();
                if (gtsam::symbolIndex(x_key) < latest_time_step) {
                    continue;
                }
                gtsam::Key lmk_key = meas2d->key2();
                lmks_to_draw.insert(lmk_key);

                value_keys.insert(x_key);
                value_keys.insert(lmk_key);

                gtsam::Pose2 x = estimates.at<gtsam::Pose2>(x_key);
                gtsam::Point2 l = estimates.at<gtsam::Point2>(lmk_key);

                line[0] = x.x();
                line[1] = l.x();

                line[2] = x.y();
                line[3] = l.y();

                ImPlot::PlotLine("Measurement Ground Truth", line, line + 2, 2);
            }
            if (odom2d)
            {
                gtsam::Key x_from_key = odom2d->key1();
                if (gtsam::symbolIndex(x_from_key) < latest_time_step) {
                    continue;
                }
                gtsam::Key x_to_key = odom2d->key2();

                gtsam::Pose2 x_from = estimates.at<gtsam::Pose2>(x_from_key);
                gtsam::Pose2 x_to = estimates.at<gtsam::Pose2>(x_to_key);

                value_keys.insert(x_from_key);
                value_keys.insert(x_to_key);

                line[0] = x_from.x();
                line[1] = x_to.x();

                line[2] = x_from.y();
                line[3] = x_to.y();

                ImPlot::PlotLine("Odometry Ground Truth", line, line + 2, 2);
            }
            if (meas3d)
            {
                gtsam::Key x_key = meas3d->key1();
                if (gtsam::symbolIndex(x_key) < latest_time_step) {
                    continue;
                }
                gtsam::Key lmk_key = meas3d->key2();
                lmks_to_draw.insert(lmk_key);

                value_keys.insert(x_key);
                value_keys.insert(lmk_key);

                gtsam::Pose3 x = estimates.at<gtsam::Pose3>(x_key);
                gtsam::Point3 l = estimates.at<gtsam::Point3>(lmk_key);

                line[0] = x.x();
                line[1] = l.x();

                line[2] = x.y();
                line[3] = l.y();

                ImPlot::SetNextLineStyle(colors::MAGENTA);
                ImPlot::PlotLine("Measurement Ground Truth", line, line + 2, 2);
            }
            if (odom3d)
            {
                gtsam::Key x_from_key = odom3d->key1();
                if (gtsam::symbolIndex(x_from_key) < latest_time_step) {
                    continue;
                }
                gtsam::Key x_to_key = odom3d->key2();

                gtsam::Pose3 x_from = estimates.at<gtsam::Pose3>(x_from_key);
                gtsam::Pose3 x_to = estimates.at<gtsam::Pose3>(x_to_key);

                value_keys.insert(x_from_key);
                value_keys.insert(x_to_key);

                line[0] = x_from.x();
                line[1] = x_to.x();

                line[2] = x_from.y();
                line[3] = x_to.y();

                ImPlot::SetNextLineStyle(colors::CYAN);
                ImPlot::PlotLine("Odometry Ground Truth", line, line + 2, 2);
            }
        }

        for (const gtsam::Values::ConstKeyValuePair p : estimates)
        {
            unsigned char chr;
            uint64_t idx;
            if (gtsam::symbolChr(p.key) == 'x' || gtsam::symbolChr(p.key) == '\0')
            {
                chr = 'x';
                idx = gtsam::symbolIndex(p.key);
                if (idx < latest_time_step) {
                    continue;
                }
                try
                {
                    gtsam::Pose2 x = p.value.cast<gtsam::Pose2>();
                    xp = x.x();
                    yp = x.y();
                }
                catch (const std::bad_cast &e)
                {
                    gtsam::Pose3 x = p.value.cast<gtsam::Pose3>();
                    xp = x.x();
                    yp = x.y();
                }
                legend = "Poses Ground Truth";
                ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 5.0, colors::ROSY_BROWN);
            }
            else if (gtsam::symbolChr(p.key) == 'l')
            {
                chr = 'l';
                idx = gtsam::symbolIndex(p.key);
                if (!lmks_to_draw.exists(p.key)) {
                    continue;
                }
                try
                {
                    gtsam::Point2 l = p.value.cast<gtsam::Point2>();
                    xp = l.x();
                    yp = l.y();
                }
                catch (const std::bad_cast &e)
                {
                    gtsam::Point3 l = p.value.cast<gtsam::Point3>();
                    xp = l.x();
                    yp = l.y();
                }
                legend = "Landmarks Ground Truth";
                ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 5.0, colors::ORANGE);
            }
            else
            {
                std::cerr << "Received key " << gtsam::Symbol(p.key) << " which could not be parsed, skipping\n";
                std::cerr << "Chr: \"" << gtsam::symbolChr(p.key) << "\"\nIndex: " << gtsam::symbolIndex(p.key) << "\n";
                continue; // Should never happen??
            }

            if (!value_keys.exists(p.key))
            { // Found value with no factor attached to it
                ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 10.0, ImVec4(1.0, 0.0, 0.0, 1.0));
            }

            ImPlot::PlotScatter(legend.c_str(), &xp, &yp, 1);
            ss << chr << idx;
            ImPlot::PlotText(ss.str().c_str(), xp, yp, false, ImVec2(15, 15));
            ss.str("");
        }

    }




    void draw_covar_ell(const Eigen::Vector2d &l, const Eigen::Matrix2d &S, const double s, const char *covariance_label, const int n)
    {
        Eigen::MatrixXd ell = ellipse(l, S, s, n);
        ImPlot::PlotLine(covariance_label, &ell(0, 0), &ell(1, 0), n, 0, 2 * sizeof(double));
        ImPlot::SetNextMarkerStyle(ImPlotMarker_Diamond, 5.0, ImVec4(119.0 / 255.0, 100.0 / 255.0, 182.0 / 255.0, 1.0));
        ImPlot::PlotScatter("lmk", &l(0), &l(1), 1);
    }

    void draw_circle(const Eigen::Vector2d &center, const double r, const int n)
    {
        Eigen::MatrixXd circ = circle(center, r, n);
        ImPlot::SetNextLineStyle(colors::UGLY_YELLOW, 20.0);
        ImPlot::PlotLine("##circle", &circ(0, 0), &circ(1, 0), n, 0, 2 * sizeof(double));
    }

} // namespace visualization