#pragma once

#include <stdio.h>

#include <memory>

#include "da_slam/types.hpp"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#if defined(IMGUI_IMPL_OPENGL_ES2)
#include <GLES2/gl2.h>
#endif
#include <GLFW/glfw3.h>  // Will drag system OpenGL headers

// [Win32] Our example includes a copy of glfw3.lib pre-compiled with VS2010 to maximize ease of testing and
// compatibility with old VS compilers. To link with VS2010-era libraries, VS2015+ requires linking with
// legacy_stdio_definitions.lib, which we do using this pragma. Your own project should not be affected, as you are
// likely to link with a newer binary of GLFW that is adequate for your version of Visual Studio.
#if defined(_MSC_VER) && (_MSC_VER >= 1900) && !defined(IMGUI_DISABLE_WIN32_FUNCTIONS)
#pragma comment(lib, "legacy_stdio_definitions")
#endif

#include <gtsam/nonlinear/NonlinearFactorGraph.h>

#include "da_slam/data_association/data_association_interface.hpp"
#include "da_slam/data_association/hypothesis.hpp"

namespace da_slam::visualization
{

void progress_bar(int curr_timestep, int tot_timesteps);

// Assumes that factors are PoseToPoint 2D, robot is pose and landmarks are points
void draw_factor_graph(const gtsam::NonlinearFactorGraph& graph, const gtsam::Values& estimates,
                       int latest_time_step = 0);
void draw_hypothesis(const data_association::hypothesis::Hypothesis& hypothesis,
                     const types::Measurements<gtsam::Point2>& measurements, const gtsam::NonlinearFactorGraph& graph,
                     const gtsam::Values& estimates, const gtsam::Key x_key, const double sigmas, const double ic_prob,
                     const std::map<gtsam::Key, bool>& lmk_cov_to_draw);

void draw_hypothesis(const data_association::hypothesis::Hypothesis& hypothesis,
                     const types::Measurements<gtsam::Point3>& measurements, const gtsam::NonlinearFactorGraph& graph,
                     const gtsam::Values& estimates, const gtsam::Key x_key, const double sigmas, const double ic_prob,
                     const std::map<gtsam::Key, bool>& lmk_cov_to_draw);

void draw_factor_graph_ground_truth(const gtsam::NonlinearFactorGraph& graph, const gtsam::Values& estimates,
                                    int latest_time_step = 0);
void draw_covar_ell(const Eigen::Vector2d& l, const Eigen::Matrix2d& S, const double s = 1.0,
                    const char* covariance_label = "Covariance", const int n = 200);

void draw_covar_ell(const Eigen::Vector3d& l, const Eigen::Matrix3d& S, const double z = 0.0, const double s = 1.0,
                    const char* covariance_label = "Covariance", const int n = 200);
void draw_circle(const Eigen::Vector2d& center, const double r = 1.0, const int n = 200);

}  // namespace da_slam::visualization