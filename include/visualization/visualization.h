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

    void new_frame();
    void render();
    void progress_bar(int curr_timestep, int tot_timesteps);
    bool running();
    bool init();
    void shutdown();

    // Assumes that factors are PoseToPoint 2D, robot is pose and landmarks are points
    void draw_factor_graph(const gtsam::NonlinearFactorGraph &graph, const gtsam::Values &estimates);
    void draw_covar_ell(const Eigen::Vector2d& l, const Eigen::Matrix2d& S, const double s = 1.0, const char* covariance_label = "Covariance", const int n = 200);
    void draw_circle(const Eigen::Vector2d& center, const double r = 1.0, const int n = 200);
    // Eigen::MatrixXd ellipse(const Eigen::Vector2d &mu, const Eigen::Matrix2d &P, const double s = 1.0, const int n = 200);
    // Eigen::MatrixXd ellipse(const Eigen::Vector3d &mu, const Eigen::Matrix3d &P, const double s = 1.0, const int n = 200)

    // template<class POSE, class POINT>
    // void draw_marginal_covar_ells(
    //     const gtsam::Values& estimates,
    //     const gtsam::Marginals& marginals,
    //     const double s,
    //     const char* covariance_label = "Covariance", const int n = 200)
    // {
    //     for (gtsam::Key k : estimates.keys()) {
    //         Eigen::MatrixXd ell;
    //         if (gtsam::symbolChr(k) == 'l') {
    //             const POINT& l = estimates.at<POINT>(k);
    //         }
    //         else if (gtsam::symbolChr(k) == 'x') {


    //     } else {
    //         continue; // Should not happen
    //     }
    //     Eigen::MatrixXd ell = ellipse(l, S, s, n);
    //     ImPlot::PlotLine(covariance_label, &ell(0, 0), &ell(1, 0), n, 0, 2 * sizeof(double));
    //     ImPlot::SetNextMarkerStyle(ImPlotMarker_Diamond, 5.0, ImVec4(119.0 / 255.0, 100.0 / 255.0, 182.0 / 255.0, 1.0));
    //     ImPlot::PlotScatter("lmk", &l(0), &l(1), 1);

} // namespace visualization