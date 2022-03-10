#ifndef SLAM_H
#define SLAM_H

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/Pose3.h>
#include <vector>

#include "slam/types.h"
#include "data_association/Hypothesis.h"
#include "data_association/DataAssociation.h"

namespace slam
{

    using gtsam::symbol_shorthand::L;
    using gtsam::symbol_shorthand::X;

    template <class POSE, class POINT>
    class SLAM
    {
    private:
        gtsam::NonlinearFactorGraph graph_;
        gtsam::ISAM2 isam_;

        gtsam::Values initial_estimates_;
        gtsam::noiseModel::Diagonal::shared_ptr pose_prior_noise_;
        gtsam::noiseModel::Diagonal::shared_ptr lmk_prior_noise_;

        std::shared_ptr<da::DataAssociation<Measurement<POINT>>> data_association_;

        unsigned long int latest_pose_key_;
        POSE latest_pose_;
        unsigned long int latest_landmark_key_;

        void incrementLatestPoseKey() { latest_pose_key_++; }
        void incrementLatestLandmarkKey() { latest_landmark_key_++; }

        void addOdom(const Odometry<POSE> &odom);
        gtsam::FastVector<POINT> predictLandmarks() const;
        void log_timestep(const Timestep<POSE, POINT>& timestep, const da::hypothesis::Hypothesis& h);

    public:
        SLAM();

        inline const gtsam::Values currentEstimates() const { return isam_.calculateEstimate(); }
        void processTimestep(const Timestep<POSE, POINT>& timestep);
        void initialize(const gtsam::Vector &pose_prior_noise, std::shared_ptr<da::DataAssociation<Measurement<POINT>>> data_association); //, const gtsam::Vector &lmk_prior_noise);
        gtsam::FastVector<POSE> getTrajectory() const;
        gtsam::FastVector<POINT> getLandmarkPoints() const;
        inline const gtsam::NonlinearFactorGraph& getGraph() const { return isam_.getFactorsUnsafe(); }
        inline double error() const { return isam_.getFactorsUnsafe().error(currentEstimates()); }
        inline void update() { isam_.update(); }
        void update(gtsam::NonlinearFactorGraph& graph, gtsam::Values& initial_estimates);
    };

    using SLAM3D = SLAM<gtsam::Pose3, gtsam::Point3>;
    using SLAM2D = SLAM<gtsam::Pose2, gtsam::Point2>;

} // namespace slam

#include "slam/slam.hxx"

#endif // SLAM_H