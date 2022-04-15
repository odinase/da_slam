#ifndef SLAM_H
#define SLAM_H

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtParams.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/Pose3.h>
#include <vector>
#include <memory>
#include <iostream>

#include "slam/types.h"
#include "data_association/Hypothesis.h"
#include "data_association/DataAssociation.h"


namespace slam
{
    enum class OptimizationMethod {
        GaussNewton = 0,
        LevenbergMarquardt = 1,
    };
} // slam

inline std::ostream& operator<<(std::ostream& os, const slam::OptimizationMethod& optimization_method) {
      switch (optimization_method) {
    case slam::OptimizationMethod::GaussNewton: {
      os << "GaussNewton";
      break;
    }
    case slam::OptimizationMethod::LevenbergMarquardt: {
      os << "LevenbergMarquardt";
      break;
    }
  }
  return os;
}

namespace slam
{
    using gtsam::symbol_shorthand::L;
    using gtsam::symbol_shorthand::X;

    template <class POSE, class POINT>
    class SLAM
    {
    private:
        gtsam::NonlinearFactorGraph graph_;
        gtsam::Values estimates_;

        gtsam::noiseModel::Diagonal::shared_ptr pose_prior_noise_;
        gtsam::noiseModel::Diagonal::shared_ptr lmk_prior_noise_;

        std::shared_ptr<da::DataAssociation<Measurement<POINT>>> data_association_;
        da::hypothesis::Hypothesis latest_hypothesis_;
        gtsam::Values hypothesis_values_;
        gtsam::NonlinearFactorGraph hypothesis_graph_;

        unsigned long int latest_pose_key_;
        unsigned long int latest_landmark_key_;

        void incrementLatestPoseKey() { latest_pose_key_++; }
        void incrementLatestLandmarkKey() { latest_landmark_key_++; }

        void addOdom(const Odometry<POSE> &odom);
        gtsam::FastVector<POINT> predictLandmarks() const;
        void log_timestep(const Timestep<POSE, POINT>& timestep, const da::hypothesis::Hypothesis& h);

        OptimizationMethod optimization_method_;
        gtsam::Marginals::Factorization marginals_factorization_;
        void optimize();

    public:
        SLAM();

        // inline const gtsam::Values currentEstimates() const { return estimates_; }
        inline const gtsam::Values& currentEstimates() const { return estimates_; }
        void processTimestep(const Timestep<POSE, POINT>& timestep);
        void initialize(
            const gtsam::Vector &pose_prior_noise,
            std::shared_ptr<da::DataAssociation<Measurement<POINT>>> data_association,
            OptimizationMethod optimizaton_method = OptimizationMethod::GaussNewton,
            gtsam::Marginals::Factorization marginals_factorization = gtsam::Marginals::CHOLESKY
        );
        gtsam::FastVector<POSE> getTrajectory() const;
        gtsam::FastVector<POINT> getLandmarkPoints() const;
        inline const gtsam::NonlinearFactorGraph& getGraph() const { return graph_; }
        inline double error() const { return getGraph().error(currentEstimates()); }
        inline const da::hypothesis::Hypothesis& latestHypothesis() const { return latest_hypothesis_; }
        inline gtsam::Key latestPoseKey() const { return X(latest_pose_key_); }
        inline POSE latestPose() const { return estimates_.at<POSE>(latestPoseKey()); }

        inline const gtsam::NonlinearFactorGraph& hypothesisGraph() const { return hypothesis_graph_; }
        inline const gtsam::Values& hypothesisEstimates() const { return hypothesis_values_; }

    };

    using SLAM3D = SLAM<gtsam::Pose3, gtsam::Point3>;
    using SLAM2D = SLAM<gtsam::Pose2, gtsam::Point2>;

} // namespace slam

#include "slam/slam.hxx"

#endif // SLAM_H