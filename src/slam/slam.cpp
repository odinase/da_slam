#include "da_slam/slam/slam.hpp"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>
#include <spdlog/spdlog.h>

#include <iostream>
#include <memory>
#include <vector>

#include "da_slam/data_association/data_association_interface.hpp"
#include "da_slam/data_association/hypothesis.hpp"
#include "da_slam/slam/slam_interface.hpp"
#include "da_slam/types.hpp"
#include "da_slam/utils.hpp"
#include "da_slam/fmt.hpp"

using gtsam::symbol_shorthand::L;
using gtsam::symbol_shorthand::X;

namespace da_slam::slam
{
template <typename Pose, typename Point>
void Slam<Pose, Point>::process_timestep(const types::Timestep<Pose, Point>& timestep)
{
    namespace da = data_association;
    namespace hyp = data_association::hypothesis;

    if (timestep.step > 0) {
        add_odom(timestep.odom);
    }

    if (timestep.measurements.size() == 0) {
        m_latest_hypothesis = hyp::Hypothesis::empty_hypothesis();

        spdlog::info("No measurements to associate, so returning now...");

        return;
    }

    const auto& full_graph = get_graph();
    const auto& estimates = current_estimates();

    m_hypothesis_graph = full_graph;
    m_hypothesis_values = estimates;

    gtsam::Marginals marginals;
    try {
        marginals = gtsam::Marginals(full_graph, estimates, m_marginals_factorization);
    }
    catch (const gtsam::IndeterminantLinearSystemException& err) {
        throw types::IndeterminantLinearSystemExceptionWithGraphValues(err, m_graph, m_estimates,
                                                                       "Error when computing marginals!");
    }

    const auto hypo = m_data_association->associate(estimates, marginals, timestep.measurements);
    m_latest_hypothesis = hypo;
    const auto& assos = hypo.associations();

    spdlog::info("There are {} associations", assos.size());

    const auto transform_world_body = estimates.template at<Pose>(utils::pose_key(m_latest_pose_idx));
    int associated_measurements = 0;
    // bool new_loop_closure = false;
    for (auto&& asso : assos) {
        const auto meas_idx = static_cast<size_t>(asso->measurement);
        const auto meas = timestep.measurements.at(meas_idx).measurement;
        const auto& meas_noise = timestep.measurements.at(meas_idx).noise;
        const auto meas_world = transform_world_body * meas;
        if (asso->associated()) {
            spdlog::info("Measurement z{} associated with landmark {}", asso->measurement,
                         gtsam::Symbol(*asso->landmark));
            // new_loop_closure = true;
            m_graph.add(gtsam::PoseToPointFactor<Pose, Point>(utils::pose_key(m_latest_pose_idx), *asso->landmark, meas,
                                                              meas_noise));
            associated_measurements++;
        }
        else {
            spdlog::info("Measurement z{} unassociated, initialize landmark l{}", asso->measurement,
                         m_latest_landmark_idx);
            m_graph.add(gtsam::PoseToPointFactor<Pose, Point>(utils::pose_key(m_latest_pose_idx),
                                                              utils::lmk_key(m_latest_landmark_idx), meas, meas_noise));
            m_estimates.insert(utils::lmk_key(m_latest_landmark_idx), meas_world);
            increment_latest_landmark_idx();
        }
    }

    spdlog::info("Associated {} / {} in timestep {}", associated_measurements, timestep.measurements.size(),
                 timestep.step);

    optimize();
}

template <typename Pose, typename Point>
void Slam<Pose, Point>::initialize(
    const Eigen::Vector<double, Pose::dimension>& pose_prior_noise,
    std::unique_ptr<data_association::IDataAssociation<types::Measurement<Point>>> data_association,
    OptimizationMethod optimizaton_method, gtsam::Marginals::Factorization marginals_factorization)
{
    m_pose_prior_noise = gtsam::noiseModel::Diagonal::Sigmas(pose_prior_noise);
    m_data_association = std::move(data_association);
    m_optimization_method = optimizaton_method;
    m_marginals_factorization = marginals_factorization;

    // Add prior on first pose
    m_graph.add(gtsam::PriorFactor<Pose>(utils::pose_key(m_latest_pose_idx), Pose{}, m_pose_prior_noise));
    m_estimates.insert(utils::pose_key(m_latest_pose_idx), Pose{});
}
template <typename Pose, typename Point>
void Slam<Pose, Point>::add_odom(const types::Odometry<Pose>& odom)
{
    const auto prev_pose = latest_pose();
    m_graph.add(gtsam::BetweenFactor<Pose>(X(m_latest_pose_idx), X(m_latest_pose_idx + 1), odom.odom, odom.noise));
    Pose this_pose = prev_pose * odom.odom;
    m_estimates.insert(X(m_latest_pose_idx + 1), this_pose);

    optimize();

    increment_latest_pose_idx();
}

template <typename Pose, typename Point>
void Slam<Pose, Point>::optimize()
{
    try {
        switch (m_optimization_method) {
            case OptimizationMethod::GAUSS_NEWTON:
            {
                const gtsam::GaussNewtonParams params{};
                gtsam::GaussNewtonOptimizer optimizer{m_graph, m_estimates, params};
                m_estimates = optimizer.optimize();
                break;
            }
            case OptimizationMethod::LEVENBERG_MARQUARDT:
            {
                gtsam::LevenbergMarquardtParams params{};
                gtsam::LevenbergMarquardtOptimizer optimizer{m_graph, m_estimates, params};
                m_estimates = optimizer.optimize();
                break;
            }
        }
    }
    catch (const gtsam::IndeterminantLinearSystemException& err) {
        throw types::IndeterminantLinearSystemExceptionWithGraphValues(err, m_graph, m_estimates,
                                                                       "Error after adding odom!");
    }
}

template class Slam<gtsam::Pose3, gtsam::Point3>;
template class Slam<gtsam::Pose2, gtsam::Point2>;

}  // namespace da_slam::slam

std::ostream& operator<<(std::ostream& os, const da_slam::slam::OptimizationMethod& optimization_method)
{
    switch (optimization_method) {
        case da_slam::slam::OptimizationMethod::GAUSS_NEWTON:
        {
            os << "GaussNewton";
            break;
        }
        case da_slam::slam::OptimizationMethod::LEVENBERG_MARQUARDT:
        {
            os << "LevenbergMarquardt";
            break;
        }
    }
    return os;
}
