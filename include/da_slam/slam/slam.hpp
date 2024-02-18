#ifndef DA_SLAM_SLAM_SLAM_HPP
#define DA_SLAM_SLAM_SLAM_HPP

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

namespace da_slam::slam
{

enum class OptimizationMethod : uint8_t {
    GAUSS_NEWTON = 0,
    LEVENBERG_MARQUARDT = 1,
};

template <typename Pose, typename Point>
class Slam
{
   public:
    Slam() : m_latest_pose_idx{0}, m_latest_landmark_idx{0}
    {
    }

    const gtsam::Values& current_estimates() const
    {
        return m_estimates;
    }

    void process_timestep(const types::Timestep<Pose, Point>& timestep);

    void initialize(const Eigen::Vector<double, Pose::dimension>& pose_prior_noise,
                    std::unique_ptr<data_association::IDataAssociation<types::Measurement<Point>>> data_association,
                    OptimizationMethod optimizaton_method, gtsam::Marginals::Factorization marginals_factorization);

    const gtsam::NonlinearFactorGraph& get_graph() const
    {
        return m_graph;
    }

    double error() const
    {
        return get_graph().error(current_estimates());
    }

    const data_association::hypothesis::Hypothesis& latest_hypothesis() const
    {
        return m_latest_hypothesis;
    }

    gtsam::Key latest_pose_key() const
    {
        return utils::pose_key(m_latest_pose_idx);
    }

    Pose latest_pose() const
    {
        return m_estimates.at<Pose>(latest_pose_key());
    }

    const gtsam::NonlinearFactorGraph& hypothesis_graph() const
    {
        return m_hypothesis_graph;
    }

    const gtsam::Values& hypothesis_estimates() const
    {
        return m_hypothesis_values;
    }

   private:
    void increment_latest_pose_idx()
    {
        m_latest_pose_idx++;
    }

    void increment_latest_landmark_idx()
    {
        m_latest_landmark_idx++;
    }
    void add_odom(const types::Odometry<Pose>& odom);
    void optimize();

    gtsam::NonlinearFactorGraph m_graph{};
    gtsam::Values m_estimates{};

    gtsam::noiseModel::Diagonal::shared_ptr m_pose_prior_noise{};
    gtsam::noiseModel::Diagonal::shared_ptr m_lmk_prior_noise{};

    std::unique_ptr<data_association::IDataAssociation<types::Measurement<Point>>> m_data_association{};
    data_association::hypothesis::Hypothesis m_latest_hypothesis{};
    gtsam::Values m_hypothesis_values{};
    gtsam::NonlinearFactorGraph m_hypothesis_graph{};

    uint64_t m_latest_pose_idx{};
    uint64_t m_latest_landmark_idx{};

    OptimizationMethod m_optimization_method = OptimizationMethod::GAUSS_NEWTON;
    gtsam::Marginals::Factorization m_marginals_factorization = gtsam::Marginals::Factorization::CHOLESKY;
};

using Slam3D = Slam<gtsam::Pose3, gtsam::Point3>;
using Slam2D = Slam<gtsam::Pose2, gtsam::Point2>;

}  // namespace da_slam::slam

std::ostream& operator<<(std::ostream& os, const da_slam::slam::OptimizationMethod& optimization_method);

#endif  // DA_SLAM_SLAM_SLAM_HPP