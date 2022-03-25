#include "visualization/visualization.h"
#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <cmath>
#include "imgui.h"
#include "imgui_internal.h"
#include "implot.h"
#include <glog/logging.h>
#include "slam/utils_g2o.h"
#include <utility>
#include <limits>
#include <string>
#include <sstream>

#include <gtsam/geometry/Pose2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/sam/BearingRangeFactor.h>
#include <gtsam/slam/dataset.h>
#include <gtsam_unstable/slam/PoseToPointFactor.h>
// #include "slam/utils.h"
#include <cmath>
#include <fstream>
#include <tuple>
#include <algorithm>
#include <iostream>

// #include "slam/slam_g2o_file.h"
#include "slam/utils_g2o.h"
#include "slam/slam.h"
#include "slam/types.h"
#include "data_association/ml/MaximumLikelihood.h"
#include "data_association/Hypothesis.h"

using gtsam::symbol_shorthand::L; // gtsam/slam/dataset.cpp
using gtsam::symbol_shorthand::X; // gtsam/slam/dataset.cpp

using namespace std;
using namespace gtsam;
using namespace Eigen;

namespace gtsam
{
    using PoseToPointFactor2 = PoseToPointFactor<Pose2, Point2>;
}

int main(int argc, char **argv)
{
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();

    // Setup visualization
    if (!viz::init())
    {
        cout << "Failed to initialize visualization, aborting!\n";
        return -1;
    }

    // 1. Build initial estimates of map

    gtsam::Values estimates;
    gtsam::NonlinearFactorGraph graph;
    gtsam::KeyVector graph_keys;
    gtsam::FastVector<slam::Measurement<gtsam::Point2>> measurements;

    gtsam::ISAM2Params isam_params;
    isam_params.relinearizeThreshold = 0.01;
    isam_params.relinearizeSkip = 1;
    isam_params.setOptimizationParams(gtsam::ISAM2DoglegParams());
    gtsam::ISAM2 isam(isam_params);

    // Initialize in origin
    gtsam::Pose2 x0 = gtsam::Pose2();
    uint64_t pose_idx = 0;
    auto priorModel = //
        noiseModel::Diagonal::Variances(Vector3(1e-6, 1e-6, 1e-8));

    auto odom_noise = //
        noiseModel::Diagonal::Sigmas(Vector3(0.05, 0.05, 2.0 * M_PI / 180.0));

    graph.addPrior(X(pose_idx), x0, priorModel);
    estimates.insert(X(pose_idx), x0);
    graph_keys.push_back(X(pose_idx));

    const double meas_sigma = 0.2; // 0.2 m std in x and y for measurements

    gtsam::noiseModel::Isotropic::shared_ptr meas_noise =
        gtsam::noiseModel::Isotropic::Sigma(2, meas_sigma);

    double prob = 0.99;
    double sigmas = sqrt(da::chi2inv(prob, 2));

    cout << "sigmas: " << sigmas << "\n";

    // Measurements

    gtsam::Point2 z1(1.0, 0.8);
    gtsam::Point2 z2(1.0, -0.8);
    slam::Measurement<gtsam::Point2> m1, m2;
    m1.measurement = z1;
    m1.noise = meas_noise;

    m2.measurement = z2;
    m2.noise = meas_noise;

    measurements.push_back(m1);
    measurements.push_back(m2);

    uint64_t lmk_idx = 0;

    graph.add(gtsam::PoseToPointFactor2(X(pose_idx), L(lmk_idx), z1, meas_noise));
    gtsam::Point2 l0 = x0 * z1;
    estimates.insert(L(lmk_idx), l0);
    graph_keys.push_back(L(lmk_idx));

    lmk_idx++;
    graph.add(gtsam::PoseToPointFactor2(X(pose_idx), L(lmk_idx), z2, meas_noise));
    gtsam::Point2 l1 = x0 * z2;
    estimates.insert(L(lmk_idx), l1);
    graph_keys.push_back(L(lmk_idx));

    isam.update(graph, estimates);

    graph.resize(0);
    estimates.clear();

    bool next_timestep = false;

    auto isam_graph = isam.getFactorsUnsafe();
    auto curr_estimates = isam.calculateEstimate();
    gtsam::Marginals marginals = gtsam::Marginals(isam_graph, curr_estimates);
    gtsam::JointMarginal joint_marginals = marginals.jointMarginalCovariance(graph_keys);
    gtsam::Matrix Hx, Hl;

    std::stringstream covariance_label_ss;
    covariance_label_ss << "Covariance ellipsis, " << sigmas << " sigmas or " << prob << " confidence interval\n";
    const std::string covariance_label = covariance_label_ss.str();

    da::ml::MaximumLikelihood2D ml(sigmas);

    while (viz::running() && !next_timestep)
    {
        viz::new_frame();

        gtsam::PoseToPointFactor2 f1(X(pose_idx), L(0), z1, meas_noise);
        gtsam::Vector error = f1.evaluateError(x0, l0, Hx, Hl);
        da::hypothesis::Association a0(0, L(0), Hx, Hl, error);
        double log_norm_factor;
        Eigen::Matrix2d S0, S1;
        double mh_dist = da::individual_compatability(a0, X(0), joint_marginals, measurements, log_norm_factor, S0);

        gtsam::PoseToPointFactor2 f2(X(pose_idx), L(1), z2, meas_noise);
        error = f2.evaluateError(x0, l1, Hx, Hl);
        da::hypothesis::Association a1(1, L(1), Hx, Hl, error);
        mh_dist = da::individual_compatability(a1, X(0), joint_marginals, measurements, log_norm_factor, S1);

        ImGui::Begin("Factor graph");
        if (ImPlot::BeginPlot("##factor graph", ImVec2(-1, -1)))
        {
            viz::draw_covar_ell(l0, S0, sigmas, covariance_label.c_str());
            viz::draw_covar_ell(l1, S1, sigmas, covariance_label.c_str());
            viz::draw_factor_graph(isam_graph, curr_estimates);

            ImPlot::EndPlot();
        }
        ImGui::End();

        ImGui::Begin("Menu");
        next_timestep = ImGui::Button("Next timestep");
        ImGui::End();

        viz::render();
    }

    next_timestep = false;

    gtsam::Pose2 odom(1.0, -0.03, 0.0);

    gtsam::Pose2 x1 = x0 * odom;
    graph.add(gtsam::BetweenFactor<gtsam::Pose2>(X(0), X(1), odom, odom_noise));
    estimates.insert(X(1), x1);

    isam.update(graph, estimates);
    graph.resize(0);
    estimates.clear();

    graph_keys.push_back(X(1));

    isam_graph = isam.getFactorsUnsafe();
    curr_estimates = isam.calculateEstimate();
    marginals = gtsam::Marginals(isam_graph, curr_estimates);
    joint_marginals = marginals.jointMarginalCovariance(graph_keys);

    

    while (viz::running() && !next_timestep)
    {
        viz::new_frame();

        gtsam::PoseToPointFactor2 f1(X(pose_idx), L(0), z1, meas_noise);
        gtsam::Vector error = f1.evaluateError(x0, l0, Hx, Hl);
        da::hypothesis::Association a0(0, L(0), Hx, Hl, error);
        double log_norm_factor;
        Eigen::Matrix2d S0, S1;
        double mh_dist = da::individual_compatability(a0, X(0), joint_marginals, measurements, log_norm_factor, S0);

        gtsam::PoseToPointFactor2 f2(X(pose_idx), L(1), z2, meas_noise);
        error = f2.evaluateError(x0, l1, Hx, Hl);
        da::hypothesis::Association a1(1, L(1), Hx, Hl, error);
        mh_dist = da::individual_compatability(a1, X(0), joint_marginals, measurements, log_norm_factor, S1);

        ImGui::Begin("Factor graph");
        if (ImPlot::BeginPlot("##factor graph", ImVec2(-1, -1)))
        {
            viz::draw_covar_ell(l0, S0, sigmas, covariance_label.c_str());
            viz::draw_covar_ell(l1, S1, sigmas, covariance_label.c_str());
            viz::draw_factor_graph(isam_graph, curr_estimates);

            ImPlot::EndPlot();
        }
        ImGui::End();

        ImGui::Begin("Menu");
        next_timestep = ImGui::Button("Next timestep");
        ImGui::End();

        viz::render();
    }
}