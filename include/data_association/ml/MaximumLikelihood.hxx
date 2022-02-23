#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam_unstable/slam/PoseToPointFactor.h>
#include <iostream>
#include <utility>
#include <algorithm>
#include <memory>
#include <slam/types.h>
#include <limits>

#include "data_association/Hungarian.h"

namespace da
{

  namespace ml
  {
    using gtsam::symbol_shorthand::L;
    using gtsam::symbol_shorthand::X;

    template <class POSE, class POINT>
    MaximumLikelihood<POSE, POINT>::MaximumLikelihood(double sigmas, double range_threshold)
        : mh_threshold_(sigmas * sigmas),
          range_threshold_(range_threshold)
    {
    }

    template <class POSE, class POINT>
    Hypothesis MaximumLikelihood<POSE, POINT>::associate(
        const gtsam::Values &estimates,
        const gtsam::Marginals &marginals,
        const gtsam::FastVector<slam::Measurement<POINT>> &measurements)
    {
      gtsam::KeyList landmark_keys = estimates.filter(gtsam::Symbol::ChrTest('l')).keys();
      auto poses = estimates.filter(gtsam::Symbol::ChrTest('x'));
      int last_pose = poses.size() - 1; // Assuming first pose is 0
      gtsam::Key x_key = X(last_pose);
      POSE x_pose = estimates.at<POSE>(x_key);
      size_t num_measurements = measurements.size();
      size_t num_landmarks = landmark_keys.size();

      if (num_landmarks == 0 || num_measurements == 0)
      {
        hypothesis::Hypothesis h = hypothesis::Hypothesis::empty_hypothesis();
        h.fill_with_unassociated_measurements(num_measurements);
        return h;
      }

      gtsam::Matrix Hx, Hl;
      gtsam::Matrix cost_matrix = gtsam::Matrix::Constant(num_measurements + num_landmarks, num_landmarks, -std::numeric_limits<double>::infinity());

      // Fill bottom diagonal with "dummy measurements" meaning they are unassigned.
      cost_matrix.bottomRows(num_landmarks).diagonal() << gtsam::Vector::Constant(num_landmarks, -10'000);

      for (int meas_idx = 0; meas_idx < num_measurements; meas_idx++)
      {
        const auto &meas = measurements[meas_idx].measurement;
        POINT meas_world = x_pose * meas;
        const auto &noise = measurements[meas_idx].noise;

        for (int lmk_idx = 0; lmk_idx < num_landmarks; lmk_idx++)
        {
          gtsam::Key l = L(lmk_idx);
          POINT lmk = estimates.at<POINT>(l);
          if ((meas_world - lmk).norm() > range_threshold_)
          {
            // std::cout << "landmark " << lmk_idx << " not associated, too far away at " << (meas_world - lmk).norm() << "? Skipping\n";
            continue; // Landmark too far away to be relevant.
          }

          gtsam::PoseToPointFactor<POSE, POINT> factor(x_key, l, meas, noise);
          gtsam::Vector error = factor.evaluateError(x_pose, lmk, Hx, Hl);
          hypothesis::Association a(meas_idx, l, Hx, Hl, error);
          double nis = individual_compatability(a, x_key, marginals, measurements);

          // Individually compatible?
          if (nis < mh_threshold_)
          {
            // We use negative NIS to force the highest reward to be the one with the lowest nis
            cost_matrix(meas_idx, lmk_idx) = -nis;
          }
        }
      }

      std::vector<int> associated_landmarks = auction(cost_matrix);
      double tot_reward = 0.0;
      double asso_reward = 0.0;
      // Loop over all landmarks with measurement associated with it and pick put the best one by pruning out all measurements except the best one in terms of Mahalanobis distance.
      hypothesis::Hypothesis h = hypothesis::Hypothesis::empty_hypothesis();
      for (int lmk_idx = 0; lmk_idx < associated_landmarks.size(); lmk_idx++)
      {
        int meas_idx = associated_landmarks[lmk_idx];
        tot_reward += cost_matrix(meas_idx, lmk_idx);
        if (meas_idx == -1 || meas_idx >= measurements.size())
        {
          // std::cout << "landmark " << lmk_idx << " not associated\n";//, too far away at "<< (meas_world - lmk).norm() << "? Skipping\n";
          continue; // Landmark associated with dummy measurement, so skip
        }
        gtsam::Key l = L(lmk_idx);
        POINT lmk = estimates.at<POINT>(l);

        asso_reward += cost_matrix(meas_idx, lmk_idx);

        const auto &meas = measurements[meas_idx].measurement;
        const auto &noise = measurements[meas_idx].noise;

        gtsam::PoseToPointFactor<POSE, POINT> factor(x_key, l, meas, noise);
        gtsam::Vector error = factor.evaluateError(x_pose, lmk, Hx, Hl);
        Association::shared_ptr a = std::make_shared<Association>(meas_idx, l, Hx, Hl, error);
        double nis = individual_compatability(*a, x_key, marginals, measurements);
        // std::cout << "landmark " << lmk_idx << " associated to measurement " << meas_idx << " with nis " << nis << " which is within threshold " << mh_threshold_ << "\n";
        h.extend(a);
      }
      h.fill_with_unassociated_measurements(measurements.size());

      std::cout << "total reward: " << tot_reward << "\nasso reward: " << asso_reward << "\n";

// This computation is not needed right now
#ifdef HYPOTHESIS_QUALITY
      std::cout << "Computing joint NIS\n";
      double nis = joint_compatability<POSE::dimension, POINT::RowsAtCompileTime, POINT::RowsAtCompileTime>(h, x_key, marginals, measurements);
      h.set_nis(nis);
#endif

      return h;
    }
  } // namespace ml
} // namespace da
