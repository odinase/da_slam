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

    void process_timestep(const types::Timestep<Pose, Point>& timestep)
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
        bool new_loop_closure = false;
        for (int i = 0; i < assos.size(); i++) {
            const auto asso = assos[i];
            const auto meas = timestep.measurements[asso->measurement].measurement;
            const auto& meas_noise = timestep.measurements[asso->measurement].noise;
            const auto meas_world = transform_world_body * meas;
            if (asso->associated()) {
                spdlog::info("Measurement z{} associated with landmark {}", asso->measurement,
                             gtsam::Symbol(*asso->landmark));
                new_loop_closure = true;
                m_graph.add(gtsam::PoseToPointFactor<Pose, Point>(utils::pose_key(m_latest_pose_idx), *asso->landmark,
                                                                  meas, meas_noise));
                associated_measurements++;
            }
            else {
                spdlog::info("Measurement z{} unassociated, initialize landmark l{}", asso->measurement,
                             m_latest_landmark_idx);
                m_graph.add(gtsam::PoseToPointFactor<Pose, Point>(
                    utils::pose_key(m_latest_pose_idx), utils::lmk_key(m_latest_landmark_idx), meas, meas_noise));
                m_estimates.insert(utils::lmk_key(m_latest_landmark_idx), meas_world);
                increment_latest_landmark_idx();
            }
        }

        spdlog::info("Associated {} / {} in timestep {}", associated_measurements, timestep.measurements.size(),
                     timestep.step);

        optimize();
    }

    void initialize(const Eigen::Vector<double, Pose::dimension>& pose_prior_noise,
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

inline std::ostream& operator<<(std::ostream& os, const da_slam::slam::OptimizationMethod& optimization_method)
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

#endif  // DA_SLAM_SLAM_SLAM_HPP