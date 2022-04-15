#include "slam/slam.h"
#include "slam/types.h"
#include "data_association/Hypothesis.h"
#include "data_association/DataAssociation.h"

#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>

#include <gtsam/base/FastVector.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam_unstable/slam/PoseToPointFactor.h>

#include <iostream>
#include <chrono>
#include <fstream>
#include <set>

namespace slam
{

  template <class POSE, class POINT>
  SLAM<POSE, POINT>::SLAM()
      : latest_pose_key_(0),
        latest_landmark_key_(0)
  {
  }

  template <class POSE, class POINT>
  void SLAM<POSE, POINT>::initialize(
      const gtsam::Vector &pose_prior_noise,
      std::shared_ptr<da::DataAssociation<Measurement<POINT>>> data_association,
      OptimizationMethod optimizaton_method,
      gtsam::Marginals::Factorization marginals_factorization)
  {
    pose_prior_noise_ = gtsam::noiseModel::Diagonal::Sigmas(pose_prior_noise);
    data_association_ = data_association;

    optimization_method_ = optimizaton_method;
    marginals_factorization_ = marginals_factorization;

    // Add prior on first pose
    graph_.add(gtsam::PriorFactor<POSE>(X(latest_pose_key_), POSE(), pose_prior_noise_));
    estimates_.insert(X(latest_pose_key_), POSE());
  }

  template <class POSE, class POINT>
  gtsam::FastVector<POSE> SLAM<POSE, POINT>::getTrajectory() const
  {
    gtsam::Values estimates = currentEstimates();
    gtsam::FastVector<POSE> trajectory;
    for (int i = 0; i < latest_pose_key_; i++)
    {
      trajectory.push_back(estimates.at<POSE>(X(i)));
    }
    return trajectory;
  }

  template <class POSE, class POINT>
  gtsam::FastVector<POINT> SLAM<POSE, POINT>::getLandmarkPoints() const
  {
    gtsam::Values estimates = currentEstimates();
    gtsam::FastVector<POINT> landmarks;
    for (int i = 0; i < latest_landmark_key_; i++)
    {
      landmarks.push_back(estimates.at<POINT>(L(i)));
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

    da::hypothesis::Hypothesis h = da::hypothesis::Hypothesis::empty_hypothesis();

    // We have no measurements to associate, so terminate early
    if (timestep.measurements.size() == 0)
    {
      latest_hypothesis_ = h;

#ifdef LOGGING
      std::cout << "No measurements to associate, so returning now...\n";
#endif

      return;
    }

    const gtsam::NonlinearFactorGraph &full_graph = getGraph();
    const gtsam::Values &estimates = currentEstimates();

    hypothesis_graph_ = full_graph;
    hypothesis_values_ = estimates;

    gtsam::Marginals marginals;
    try
    {
      marginals = gtsam::Marginals(full_graph, estimates, marginals_factorization_);
    }
    catch (gtsam::IndeterminantLinearSystemException &indetErr)
    {
      throw IndeterminantLinearSystemExceptionWithGraphValues(indetErr, graph_, estimates_, "Error when computing marginals!");
    }

    h = data_association_->associate(estimates, marginals, timestep.measurements);
    latest_hypothesis_ = h;

    const auto &assos = h.associations();

#ifdef LOGGING
    std::cout << "There are " << assos.size() << " associations\n";
#endif

    POSE T_wb = estimates.at<POSE>(X(latest_pose_key_));
    int associated_measurements = 0;
    bool new_loop_closure = false;
    for (int i = 0; i < assos.size(); i++)
    {
      da::hypothesis::Association::shared_ptr a = assos[i];
      uint64_t meas_idx = timestep.measurements[a->measurement].idx;
      POINT meas = timestep.measurements[a->measurement].measurement;
      const auto &meas_noise = timestep.measurements[a->measurement].noise;
      POINT meas_world = T_wb * meas;
      if (a->associated())
      {
#ifdef LOGGING
        std::cout << "Measurement z" << a->measurement << " associated with landmark " << gtsam::Symbol(*a->landmark) << "\n";
#endif
        new_loop_closure = true;
        graph_.add(gtsam::PoseToPointFactor<POSE, POINT>(X(latest_pose_key_), *a->landmark, meas, meas_noise));
        associated_measurements++;
      }
      else
      {
#ifdef LOGGING
        std::cout << "Measurement z" << a->measurement << " unassociated, initialize landmark l" << latest_landmark_key_ << "\n";
#endif
        graph_.add(gtsam::PoseToPointFactor<POSE, POINT>(X(latest_pose_key_), L(latest_landmark_key_), meas, meas_noise));
        estimates_.insert(L(latest_landmark_key_), meas_world);
        incrementLatestLandmarkKey();
      }
    }

#ifdef LOGGING
    std::cout << "Associated " << associated_measurements << " / " << timestep.measurements.size() << " measurements in timestep " << timestep.step << "\n";
#endif

    optimize();
  }

  template <class POSE, class POINT>
  void SLAM<POSE, POINT>::addOdom(const Odometry<POSE> &odom)
  {
    graph_.add(gtsam::BetweenFactor<POSE>(X(latest_pose_key_), X(latest_pose_key_ + 1), odom.odom, odom.noise));
    POSE this_pose = latest_pose_ * odom.odom;
    estimates_.insert(X(latest_pose_key_ + 1), this_pose);
    latest_pose_ = this_pose;

    optimize();

    incrementLatestPoseKey();
  }

  template <class POSE, class POINT>
  gtsam::FastVector<POINT> SLAM<POSE, POINT>::predictLandmarks() const
  {
    const gtsam::Values estimates = currentEstimates();
    gtsam::KeyList landmark_keys = estimates.filter(gtsam::Symbol::ChrTest('l')).keys();
    if (landmark_keys.size() == 0)
    {
      return {};
    }
    gtsam::FastVector<POINT> predicted_measurements;
    for (const auto &lmk : landmark_keys)
    {
      predicted_measurements.push_back(estimates.at<POINT>(lmk));
    }

    return predicted_measurements;
  }

  template <class POSE, class POINT>
  void SLAM<POSE, POINT>::optimize()
  {
    try
    {
      switch (optimization_method_)
      {
      case OptimizationMethod::GaussNewton:
      {
        gtsam::GaussNewtonParams params;
        gtsam::GaussNewtonOptimizer optimizer(graph_, estimates_, params);
        estimates_ = optimizer.optimize();
        break;
      }
      case OptimizationMethod::LevenbergMarquardt:
      {
        gtsam::LevenbergMarquardtParams params;
        gtsam::LevenbergMarquardtOptimizer optimizer(graph_, estimates_, params);
        estimates_ = optimizer.optimize();
        break;
      }
      }
    }
    catch (gtsam::IndeterminantLinearSystemException &indetErr)
    {
      throw IndeterminantLinearSystemExceptionWithGraphValues(indetErr, graph_, estimates_, "Error after adding odom!");
    }
  }

} // namespace slam