#include "slam/slam.h"
#include "slam/types.h"
#include "data_association/Hypothesis.h"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>

#include <iostream>
#include <chrono>

namespace slam
{

  template <class POSE, class POINT>
  SLAM<POSE, POINT>::SLAM()
      : latest_pose_key_(0),
        latest_landmark_key_(0)
  {
  }

  template <class POSE, class POINT>
  void SLAM<POSE, POINT>::initialize(const gtsam::Vector &pose_prior_noise, std::shared_ptr<da::DataAssociation<Measurement<POINT>>> data_association)
  {
    pose_prior_noise_ = gtsam::noiseModel::Diagonal::Sigmas(pose_prior_noise);
    data_association_ = data_association;

    // Run with Gauss Newton (should be default)
    gtsam::ISAM2Params params;
    params.setOptimizationParams(gtsam::ISAM2GaussNewtonParams());
    // params.setOptimizationParams(gtsam::ISAM2DoglegParams());
    params.setRelinearizeThreshold(0.05);
    // double smoother_lag = 0.0;

    isam_ = gtsam::ISAM2(params);
    // isam_ = gtsam::ISAM2(params);

    // Add prior on first pose
    graph_.add(gtsam::PriorFactor<POSE>(X(latest_pose_key_), POSE(), pose_prior_noise_));
    initial_estimates_.insert(X(latest_pose_key_), POSE());

    isam_.update(graph_, initial_estimates_);

    estimates_ = isam_.calculateEstimate();
    graph_.resize(0);
    initial_estimates_.clear();
  }

  template <class POSE, class POINT>
  gtsam::FastVector<POSE> SLAM<POSE, POINT>::getTrajectory() const
  {
    gtsam::FastVector<POSE> trajectory;
    for (int i = 0; i < latest_pose_key_; i++)
    {
      trajectory.push_back(estimates_.at<POSE>(X(i)));
    }
    return trajectory;
  }

  template <class POSE, class POINT>
  gtsam::FastVector<POINT> SLAM<POSE, POINT>::getLandmarkPoints() const
  {
    gtsam::FastVector<POINT> landmarks;
    for (int i = 0; i < latest_landmark_key_; i++)
    {
      landmarks.push_back(estimates_.at<POINT>(L(i)));
    }
    return landmarks;
  }

  template <class POSE, class POINT>
  void SLAM<POSE, POINT>::processTimestep(const Timestep<POSE, POINT> &timestep)
  {
    if (timestep.step > 0)
    {
      addOdom(timestep.odom);
    }

    gtsam::NonlinearFactorGraph full_graph = isam_.getFactorsUnsafe();
    gtsam::Values estimates = isam_.calculateEstimate();

    gtsam::Marginals marginals = gtsam::Marginals(full_graph, estimates);

    da::hypothesis::Hypothesis h = data_association_->associate(estimates, marginals, timestep.measurements);

    const auto &assos = h.associations();
    POSE T_wb = estimates.at<POSE>(X(latest_pose_key_));
    int associated_measurements = 0;
    bool new_loop_closure = false;
    for (int i = 0; i < assos.size(); i++)
    {
      da::hypothesis::Association::shared_ptr a = assos[i];
      POINT meas = timestep.measurements[a->measurement].measurement;
      const auto &meas_noise = timestep.measurements[a->measurement].noise;
      POINT meas_world = T_wb * meas;
      if (a->associated())
      {
        new_loop_closure = true;
        graph_.add(gtsam::PoseToPointFactor<POSE, POINT>(X(latest_pose_key_), *a->landmark, meas, meas_noise));
        associated_measurements++;
      }
      else
      {
        graph_.add(gtsam::PoseToPointFactor<POSE, POINT>(X(latest_pose_key_), L(latest_landmark_key_), meas, meas_noise));
        initial_estimates_.insert(L(latest_landmark_key_), meas_world);
        incrementLatestLandmarkKey();
      }
    }
    std::cout << "Associated " << associated_measurements << " / " << timestep.measurements.size() << " measurements in timestep " << timestep.step << "\n";

    isam_.update(graph_, initial_estimates_);
    if (new_loop_closure) {
      for (int i = 0; i < 5; i++) {
        isam_.update();
      }
    }
    estimates_ = isam_.calculateEstimate();

    graph_.resize(0);
    initial_estimates_.clear();
  }

  template <class POSE, class POINT>
  void SLAM<POSE, POINT>::addOdom(const Odometry<POSE> &odom)
  {
    graph_.add(gtsam::BetweenFactor<POSE>(X(latest_pose_key_), X(latest_pose_key_ + 1), odom.odom, odom.noise));
    POSE this_pose = latest_pose_ * odom.odom;
    initial_estimates_.insert(X(latest_pose_key_ + 1), this_pose);
    latest_pose_ = this_pose;
    isam_.update(graph_, initial_estimates_);

    graph_.resize(0);
    initial_estimates_.clear();
    
    incrementLatestPoseKey();
  }

  template <class POSE, class POINT>
  gtsam::FastVector<POINT> SLAM<POSE, POINT>::predictLandmarks() const
  {
    gtsam::KeyList landmark_keys = estimates_.filter(gtsam::Symbol::ChrTest('l')).keys();
    if (landmark_keys.size() == 0)
    {
      return {};
    }
    gtsam::FastVector<POINT> predicted_measurements;
    for (const auto &lmk : landmark_keys)
    {
      predicted_measurements.push_back(estimates_.at<POINT>(lmk));
    }

    return predicted_measurements;
  }
} // namespace slam