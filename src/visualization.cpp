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
#include "data_association/DataAssociation.h"
#include "slam/slam.h"

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

Eigen::MatrixXd ellipse2d(const Eigen::Vector2d &mu, const Eigen::Matrix2d &P, const double s = 1.0, const int n = 200)
{
    Eigen::RowVectorXd thetas = Eigen::RowVectorXd::LinSpaced(n, 0, 2.0 * M_PI);
    Eigen::MatrixXd points(2, n);
    points << thetas.array().cos(),
        thetas.array().sin();
    Eigen::LLT<Eigen::Matrix2d> chol(P);
    Eigen::MatrixXd ell = (s * chol.matrixL().toDenseMatrix() * points).colwise() + mu;
    return ell;
}

std::vector<Eigen::MatrixXd> ellipse3d(const Eigen::Vector3d &mu, const Eigen::Matrix3d &P, const double s = 1.0, const int num_level_curves = 5, const int n = 200)
{
    std::vector<Eigen::MatrixXd> circs;
    Eigen::LLT<Eigen::Matrix3d> chol(P);
    Eigen::MatrixXd L = chol.matrixL().toDenseMatrix();

    Eigen::RowVectorXd thetas = Eigen::RowVectorXd::LinSpaced(n, 0, 2.0 * M_PI);
    double phi_start = 0.0;
    double phi_stop = M_PI / 2.0;
    double phi_step = (phi_stop - phi_start) / (num_level_curves - 1);
    for (int i = 0; i < num_level_curves; i++)
    {
        Eigen::MatrixXd circ(3, n);
        double phi = phi_start + phi_step * i;
        circ.topRows(2) << thetas.array().cos() * sin(phi),
            thetas.array().sin() * sin(phi);
        circ.row(2).array() = cos(phi);
        Eigen::MatrixXd ell = (s * L * circ).colwise() + mu;
        circs.push_back(ell.topRows(2));
    }
    return circs;
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

    // void config_table(const config::Config &conf)
    // {
    //     // std::stringstream ss;
    //     // std::string buffer;
    //     // if (ImGui::BeginTable("config table", 2))
    //     // {
    //     //     ImGui::TableNextRow();
    //     //     ImGui::TableNextColumn();

    //     //     ImGui::TextWrapped("Association method");
    //     //     ImGui::TableNextColumn();
    //     //     ss.str("");
    //     //     ss << conf.association_method;
    //     //     buffer = ss.str();
    //     //     ImGui::TextWrapped(buffer.c_str());

    //     //     ImGui::TableNextRow();
    //     //     ImGui::TableNextColumn();

    //     //     ImGui::TextWrapped("Optimization method");
    //     //     ImGui::TableNextColumn();
    //     //     ss.str("");
    //     //     ss << conf.optimization_method;
    //     //     buffer = ss.str();
    //     //     ImGui::TextWrapped(buffer.c_str());

    //     //     ImGui::TableNextRow();
    //     //     ImGui::TableNextColumn();

    //     //     ImGui::TextWrapped("Marginals factorization");
    //     //     ImGui::TableNextColumn();
    //     //     ImGui::TextWrapped((conf.marginals_factorization == gtsam::Marginals::CHOLESKY ? "Cholesky" : "QR"));

    //     //     ImGui::EndTable();
    //     // }
    // }

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
                if (gtsam::symbolIndex(x_key) < latest_time_step)
                {
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
                if (gtsam::symbolIndex(x_from_key) < latest_time_step)
                {
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
                if (gtsam::symbolIndex(x_key) < latest_time_step)
                {
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
                if (gtsam::symbolIndex(x_from_key) < latest_time_step)
                {
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
                if (idx < latest_time_step)
                {
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
                if (!lmks_to_draw.exists(p.key))
                {
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
                if (gtsam::symbolIndex(x_key) < latest_time_step)
                {
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
                if (gtsam::symbolIndex(x_from_key) < latest_time_step)
                {
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
                if (gtsam::symbolIndex(x_key) < latest_time_step)
                {
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
                if (gtsam::symbolIndex(x_from_key) < latest_time_step)
                {
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
                if (idx < latest_time_step)
                {
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
                if (!lmks_to_draw.exists(p.key))
                {
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
        Eigen::MatrixXd ell = ellipse2d(l, S, s, n);
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

    void draw_hypothesis(const da::hypothesis::Hypothesis &hypothesis,
                         const slam::Measurements<gtsam::Point2> &measurements,
                         const gtsam::NonlinearFactorGraph &graph,
                         const gtsam::Values &estimates,
                         const gtsam::Key x_key,
                         const double sigmas,
                         const double ic_prob,
                         const std::map<gtsam::Key, bool> &lmk_cov_to_draw)
    {
        gtsam::KeyVector keys = hypothesis.associated_landmarks();
        const gtsam::Pose2 &x_pose = estimates.at<gtsam::Pose2>(x_key);

        // Draw landmarks
        char chr;
        uint64_t idx;
        double xp, yp;
        std::string legend;
        std::stringstream ss;

        chr = 'l';
        legend = "Landmarks";
        for (const gtsam::Key l : keys)
        {
            idx = gtsam::symbolIndex(l);
            gtsam::Point2 lmk = estimates.at<gtsam::Point2>(l);
            lmk = x_pose.transformTo(lmk);
            xp = lmk.x();
            yp = lmk.y();
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 5.0, ImVec4(119.0 / 255.0, 100.0 / 255.0, 182.0 / 255.0, 1.0));
            ImPlot::PlotScatter(legend.c_str(), &xp, &yp, 1);
            ss << chr << idx;
            ImPlot::PlotText(ss.str().c_str(), xp, yp, false, ImVec2(15, 15));
            ss.str("");
        }

        // Draw current pose
        chr = 'x';
        idx = gtsam::symbolIndex(x_key);
        xp = 0;
        yp = 0;
        legend = "Pose";
        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 5.0, ImVec4(19.0 / 255.0, 160.0 / 255.0, 17.0 / 255.0, 1.0));
        ImPlot::PlotScatter(legend.c_str(), &xp, &yp, 1);
        ss << chr << idx;
        ImPlot::PlotText(ss.str().c_str(), xp, yp, false, ImVec2(15, 15));
        ss.str("");

        keys.push_back(x_key);

        gtsam::Marginals marginals = gtsam::Marginals(graph, estimates);
        gtsam::JointMarginal joint_marginal = marginals.jointMarginalCovariance(keys);

        Eigen::Matrix2d S;

        double line[4];
        std::string meas_label, lmk_label;
        std::string table_title = "Mahalanobis threshold = " + std::to_string(sigmas * sigmas) + ", sigma = " + std::to_string(sigmas) + ", probability (chi2 test) = " + std::to_string(ic_prob);

        const int num_cols_table = 5;
        ImGui::Begin("MLE Costs");
        ImGui::TextWrapped("%s", table_title.c_str());
        if (ImGui::BeginTable("table", num_cols_table))
        {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextWrapped("Measurement");
            ImGui::TableNextColumn();
            ImGui::TextWrapped("Landmark");
            ImGui::TableNextColumn();
            ImGui::TextWrapped("Mahalanobis distance");
            ImGui::TableNextColumn();
            ImGui::TextWrapped("Sigma distance");
            ImGui::TableNextColumn();
            ImGui::TextWrapped("MLE cost");
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("---");
            ImGui::TableNextColumn();
            ImGui::Text("---");
            ImGui::TableNextColumn();
            ImGui::Text("---");
            ImGui::TableNextColumn();
            ImGui::Text("---");
            ImGui::TableNextColumn();
            ImGui::Text("---");
            ImGui::EndTable();
        }
        ImGui::End();

        // Loop over all measurements that were made in this timestep and
        for (const auto &association : hypothesis.associations())
        {
            uint64_t meas_idx = association->measurement;
            const gtsam::Point2 &meas = measurements[meas_idx].measurement;

            // gtsam::Point2 meas_world = x_pose.transformFrom(meas, Gx, Gz);

            line[0] = 0;
            line[1] = meas.x();

            line[2] = 0;
            line[3] = meas.y();

            ImPlot::PlotLine("##measurement", line, line + 2, 2);
            ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 15.0f);
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Cross, 15.0f);
            ImPlot::PlotScatter("Measurement", &meas.x(), &meas.y(), 1);
            ImPlot::PopStyleVar();

            // Interpolate point between for measurement text
            chr = 'z';
            idx = measurements[meas_idx].idx;
            ss << chr << idx;

            meas_label = ss.str();
            ss.str("");

            ImPlot::PlotText(meas_label.c_str(), meas.x(), meas.y(), false, ImVec2(15, 15));

            // std::map<gtsam::Key, double> lmk_mle_cost;
            if (association->associated())
            {
                gtsam::Key lmk_key = *association->landmark;
                gtsam::Point2 associated_lmk = estimates.at<gtsam::Point2>(lmk_key);
                gtsam::Point2 associated_lmk_body = x_pose.transformTo(associated_lmk);
                ss << gtsam::Symbol(lmk_key);
                lmk_label = ss.str();
                ss.str("");

                double mh, log_norm_factor;
                mh = da::individual_compatability(*association, x_key, joint_marginal, measurements, log_norm_factor, S);
                ImGui::Begin("MLE Costs");
                if (ImGui::BeginTable("table", num_cols_table))
                {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("%s", meas_label.c_str());
                    ImGui::TableNextColumn();
                    ImGui::Text("%s", lmk_label.c_str());
                    ImGui::TableNextColumn();
                    ImGui::Text("%f", mh);
                    ImGui::TableNextColumn();
                    ImGui::Text("%f", std::sqrt(mh));
                    ImGui::TableNextColumn();
                    ImGui::Text("%f", mh + log_norm_factor);
                    ImGui::EndTable();
                }
                ImGui::End();
                if (lmk_cov_to_draw.at(lmk_key))
                {
                    // S = Gz * S * Gz.transpose() + Gx * Pxx * Gx.transpose(); // Express S in world frame. Correct to do this way??
                    Eigen::MatrixXd ell = ellipse2d(associated_lmk_body, S, sigmas);
                    int count = ell.cols();
                    int stride = 2 * sizeof(double);
                    ImPlot::PlotLine("Covariance ellipse", &ell(0, 0), &ell(1, 0), count, 0, stride);
                }
                // Draw line between measurement and lmk, and cross for measurement


                line[0] = associated_lmk_body.x();
                line[1] = meas.x();

                line[2] = associated_lmk_body.y();
                line[3] = meas.y();

                ImPlot::PlotLine("Association", line, line + 2, 2);
            }
            else
            {
                ImGui::Begin("MLE Costs");
                if (ImGui::BeginTable("table", num_cols_table))
                {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("%s", meas_label.c_str());
                    ImGui::TableNextColumn();
                    ImGui::Text("N/A");
                    ImGui::TableNextColumn();
                    ImGui::Text("N/A");
                    ImGui::TableNextColumn();
                    ImGui::Text("N/A");
                    ImGui::TableNextColumn();
                    ImGui::Text("N/A");
                    ImGui::EndTable();
                }
                ImGui::End();
            }
        }
    }

    void draw_hypothesis(
        const da::hypothesis::Hypothesis &hypothesis,
        const slam::Measurements<gtsam::Point3> &measurements,
        const gtsam::NonlinearFactorGraph &graph,
        const gtsam::Values &estimates,
        const gtsam::Key x_key,
        const double sigmas,
        const double ic_prob,
        const std::map<gtsam::Key, bool> &lmk_cov_to_draw)
    {
        gtsam::KeyVector keys = hypothesis.associated_landmarks();
        const gtsam::Pose3 x_pose = estimates.at<gtsam::Pose3>(x_key);

        // Draw landmarks
        char chr;
        uint64_t idx;
        double xp, yp;
        std::string legend;
        std::stringstream ss;

        chr = 'l';
        legend = "Landmarks";
        for (const gtsam::Key l : keys)
        {
            idx = gtsam::symbolIndex(l);
            gtsam::Point3 lmk = estimates.at<gtsam::Point3>(l);
            lmk = x_pose.transformTo(lmk);
            xp = lmk.x();
            yp = lmk.y();
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 5.0, ImVec4(119.0 / 255.0, 100.0 / 255.0, 182.0 / 255.0, 1.0));
            ImPlot::PlotScatter(legend.c_str(), &xp, &yp, 1);
            ss << chr << idx;
            ImPlot::PlotText(ss.str().c_str(), xp, yp, false, ImVec2(15, 15));
            ss.str("");
        }

        // Draw current pose
        chr = 'x';
        idx = gtsam::symbolIndex(x_key);
        xp = 0;
        yp = 0;
        legend = "Pose";
        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 5.0, ImVec4(19.0 / 255.0, 160.0 / 255.0, 17.0 / 255.0, 1.0));
        ImPlot::PlotScatter(legend.c_str(), &xp, &yp, 1);
        ss << chr << idx;
        ImPlot::PlotText(ss.str().c_str(), xp, yp, false, ImVec2(15, 15));
        ss.str("");

        keys.push_back(x_key);

        gtsam::Marginals marginals = gtsam::Marginals(graph, estimates);
        gtsam::JointMarginal joint_marginal = marginals.jointMarginalCovariance(keys);

        Eigen::Matrix3d S;

        double line[4];
        std::string meas_label, lmk_label;
        std::string table_title = "Mahalanobis threshold = " + std::to_string(sigmas * sigmas) + ", sigma = " + std::to_string(sigmas) + ", probability (chi2 test) = " + std::to_string(ic_prob);

        ImGui::Begin("MLE Costs");
        ImGui::TextWrapped("%s", table_title.c_str());
        if (ImGui::BeginTable("table", 4))
        {
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::TextWrapped("Measurement");
            ImGui::TableNextColumn();
            ImGui::TextWrapped("Landmark");
            ImGui::TableNextColumn();
            ImGui::TextWrapped("Mahalanobis distance");
            ImGui::TableNextColumn();
            ImGui::TextWrapped("Sigma distance");
            ImGui::TableNextColumn();
            ImGui::TextWrapped("MLE cost");
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            ImGui::Text("---");
            ImGui::TableNextColumn();
            ImGui::Text("---");
            ImGui::TableNextColumn();
            ImGui::Text("---");
            ImGui::TableNextColumn();
            ImGui::Text("---");
            ImGui::TableNextColumn();
            ImGui::Text("---");
            ImGui::EndTable();
        }
        ImGui::End();

        // Loop over all measurements that were made in this timestep and
        for (const auto &association : hypothesis.associations())
        {
            uint64_t meas_idx = association->measurement;
            const auto &meas = measurements[meas_idx].measurement;
            // const auto &noise = measurements[meas_idx].noise;

            // gtsam::Point3 meas_world = x_pose.transformFrom(meas, Gx, Gz);

            line[0] = 0;
            line[1] = meas.x();

            line[2] = 0;
            line[3] = meas.y();

            ImPlot::PlotLine("##measurement", line, line + 2, 2);
            ImPlot::PushStyleVar(ImPlotStyleVar_LineWeight, 15.0f);
            ImPlot::SetNextMarkerStyle(ImPlotMarker_Cross, 15.0f);
            ImPlot::PlotScatter("Measurement", &meas.x(), &meas.y(), 1);
            ImPlot::PopStyleVar();

            // Interpolate point between for measurement text
            chr = 'z';
            idx = measurements[meas_idx].idx;
            ss << chr << idx;

            meas_label = ss.str();
            ss.str("");

            ImPlot::PlotText(meas_label.c_str(), meas.x(), meas.y(), false, ImVec2(15, 15));

            // std::map<gtsam::Key, double> lmk_mle_cost;
            if (association->associated())
            {
                gtsam::Key lmk_key = *association->landmark;
                gtsam::Point3 associated_lmk = estimates.at<gtsam::Point3>(lmk_key);
                gtsam::Point3 associated_lmk_body = x_pose.transformTo(associated_lmk);
                ss << gtsam::Symbol(lmk_key);
                lmk_label = ss.str();
                ss.str("");

                double mh, log_norm_factor;
                mh = da::individual_compatability(*association, x_key, joint_marginal, measurements, log_norm_factor, S);
                ImGui::Begin("MLE Costs");
                if (ImGui::BeginTable("table", 4))
                {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("%s", meas_label.c_str());
                    ImGui::TableNextColumn();
                    ImGui::Text("%s", lmk_label.c_str());
                    ImGui::TableNextColumn();
                    ImGui::Text("%f", mh);
                    ImGui::TableNextColumn();
                    ImGui::Text("%f", std::sqrt(mh));
                    ImGui::TableNextColumn();
                    ImGui::Text("%f", mh + log_norm_factor);
                    ImGui::EndTable();
                }
                ImGui::End();
                if (lmk_cov_to_draw.at(lmk_key))
                {
                    // S = Gz * S * Gz.transpose() + Gx * Pxx * Gx.transpose(); // Transform S into world frame
                    std::vector<Eigen::MatrixXd> ell_level_curves = ellipse3d(associated_lmk_body, S, sigmas);
                    for (const Eigen::MatrixXd &ell_level_curve : ell_level_curves)
                    {
                        int count = ell_level_curve.cols();
                        int stride = 2 * sizeof(double);
                        ImPlot::PlotLine("Covariance ellipse", &ell_level_curve(0, 0), &ell_level_curve(1, 0), count, 0, stride);
                    }
                }
                // Draw line between measurement and lmk, and cross for measurement

                line[0] = associated_lmk_body.x();
                line[2] = associated_lmk_body.y();

                ImPlot::PlotLine("Association", line, line + 2, 2);
            }
            else
            {
                ImGui::Begin("MLE Costs");
                if (ImGui::BeginTable("table", 4))
                {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("%s", meas_label.c_str());
                    ImGui::TableNextColumn();
                    ImGui::Text("N/A");
                    ImGui::TableNextColumn();
                    ImGui::Text("N/A");
                    ImGui::TableNextColumn();
                    ImGui::Text("N/A");
                    ImGui::TableNextColumn();
                    ImGui::Text("N/A");
                    ImGui::EndTable();
                }
                ImGui::End();
            }
        }
    }
} // namespace visualization