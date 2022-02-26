#include "slam/slam.h"
#include "slam/types.h"
#include "data_association/Hypothesis.h"

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
  void SLAM<POSE, POINT>::initialize(const gtsam::Vector &pose_prior_noise, std::shared_ptr<da::DataAssociation<Measurement<POINT>>> data_association)
  {
    pose_prior_noise_ = gtsam::noiseModel::Diagonal::Sigmas(pose_prior_noise);
    data_association_ = data_association;

    // Run with Gauss Newton (should be default)
    gtsam::ISAM2Params params;
    params.setOptimizationParams(gtsam::ISAM2GaussNewtonParams());
    // params.setOptimizationParams(gtsam::ISAM2DoglegParams());
    params.setRelinearizeThreshold(0.001);
    // double smoother_lag = 0.0;

    isam_ = gtsam::ISAM2(params);
    // isam_ = gtsam::ISAM2(params);

    // Add prior on first pose
    graph_.add(gtsam::PriorFactor<POSE>(X(latest_pose_key_), POSE(), pose_prior_noise_));
    initial_estimates_.insert(X(latest_pose_key_), POSE());

    isam_.update(graph_, initial_estimates_);

    graph_.resize(0);
    initial_estimates_.clear();
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

    // We have no measurements to associate, so terminate early
    if (timestep.measurements.size() == 0)
    {
      return;
    }

    gtsam::NonlinearFactorGraph full_graph = isam_.getFactorsUnsafe();
    gtsam::Values estimates = isam_.calculateEstimate();

    gtsam::Marginals marginals = gtsam::Marginals(full_graph, estimates);

    da::hypothesis::Hypothesis h = da::hypothesis::Hypothesis::empty_hypothesis();

    // We have landmarks to associate
    if (latest_landmark_key_ > 0)
    {
      h = data_association_->associate(estimates, marginals, timestep.measurements);
    }
    // No landmarks, so no measurements can be associated
    else
    {
      h.fill_with_unassociated_measurements(timestep.measurements.size());
    }

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

    // if (associated_measurements == 0 && timestep.measurements.size() > 0)
    // {
    //   log_timestep(timestep, h);
    // }

    isam_.update(graph_, initial_estimates_);
    if (new_loop_closure)
    {
      for (int i = 0; i < 5; i++)
      {
        isam_.update();
      }
    }

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

  // template <class POSE, class POINT>
  // void SLAM<POSE, POINT>::log_timestep(const Timestep<POSE, POINT> &timestep, const da::hypothesis::Hypothesis &h)
  // {
  //   std::ofstream o;                                                                                  // ofstream is the class for fstream package
  //   o.open("/home/odinase/prog/C++/da-slam/logs/timestep_" + std::to_string(timestep.step) + ".txt"); // open is the method of ofstream

  //   gtsam::NonlinearFactorGraph full_graph = isam_.getFactorsUnsafe();
  //   const gtsam::Values estimates = isam_.calculateEstimate();

  //   gtsam::Marginals marginals = gtsam::Marginals(full_graph, estimates);
  //   gtsam::KeyList landmark_keys = estimates.filter(gtsam::Symbol::ChrTest('l')).keys();

  //   size_t num_measurements = timestep.measurements.size();
  //   size_t num_landmarks = landmark_keys.size();

  //   gtsam::Matrix cost_matrix = gtsam::Matrix::Constant(num_measurements + num_landmarks, num_landmarks, -std::numeric_limits<double>::infinity());

  //   // Fill bottom diagonal with "dummy measurements" meaning they are unassigned.
  //   cost_matrix.bottomRows(num_landmarks).diagonal() << gtsam::Vector::Constant(num_landmarks, -10'000);

  //   // We want to find all landmarks that were deemed close enough to at least measurement and draw their marginal expectation and covariance
  //   gtsam::Matrix Hx, Hl;

  //   for (int meas_idx = 0; meas_idx < num_measurements; meas_idx++)
  //   {
  //     o << "z" << meas_idx << " ";
  //     const auto &meas = timestep.measurements[meas_idx].measurement;
  //     o << meas.transpose() << "\n";
  //     POINT meas_world = latest_pose_ * meas;
  //     const auto &noise = timestep.measurements[meas_idx].noise;

  //     for (int lmk_idx = 0; lmk_idx < num_landmarks; lmk_idx++)
  //     {
  //       gtsam::Key l = L(lmk_idx);
  //       POINT lmk = estimates.at<POINT>(l);
  //       if ((meas_world - lmk).norm() > 3)
  //       {
  //         continue; // Landmark too far away to be relevant.
  //       }
  //       o << "l" << gtsam::symbolIndex(l) << " ";

  //       gtsam::PoseToPointFactor<POSE, POINT> factor(latest_pose_key_, l, meas, noise);
  //       gtsam::Vector error = factor.evaluateError(latest_pose_, lmk, Hx, Hl);
  //       da::hypothesis::Association a(meas_idx, l, Hx, Hl, error);

  //       Eigen::VectorXd innov = a.error;
  //       // std::cout << "in timestep " << timestep.step << " and pose key is " << gtsam::symbolIndex(latest_pose_key_) << "\n";
  //       // std::cout << "checking landmark " << gtsam::symbolIndex(l) << " with " << landmark_keys.size() << " lmks in total\n";
  //       Eigen::MatrixXd P = marginals.jointMarginalCovariance(gtsam::KeyVector{{X(latest_pose_key_), l}}).fullMatrix();

  //       int rows = a.Hx.rows();
  //       int cols = a.Hx.cols() + a.Hl.cols();
  //       Eigen::MatrixXd H(rows, cols);
  //       H << a.Hx, a.Hl;

  //       Eigen::MatrixXd R = noise->sigmas().array().square().matrix().asDiagonal();
  //       Eigen::MatrixXd S = H * P * H.transpose() + R;

  //       double nis = innov.transpose() * S.llt().solve(innov);

  //       if (nis < 3.1 * 3.1)
  //       {
  //         // We use negative NIS to force the highest reward to be the one with the lowest nis
  //         cost_matrix(meas_idx, lmk_idx) = -nis;
  //       }

  //       // Express lmk in same frame as meas
  //       POINT lmk_body = latest_pose_.inverse() * lmk;
  //       o << lmk_body.transpose() << " ";
  //       for (int i = 0; i < S.rows(); i++)
  //       {
  //         for (int j = 0; j < S.cols(); j++)
  //         {
  //           o << S(i, j) << " ";
  //         }
  //       }
  //       o << "\n";
  //     }
  //     o << "\n";
  //   }

  //   o << "c " << cost_matrix.rows() << " " << cost_matrix.cols() << " ";

  //   for (int i = 0; i < cost_matrix.rows(); i++)
  //   {
  //     for (int j = 0; j < cost_matrix.cols(); j++)
  //     {
  //       o << cost_matrix(i, j) << " ";
  //     }
  //   }

  //   o << "\n";

  //   o.close();
  // }

} // namespace slam