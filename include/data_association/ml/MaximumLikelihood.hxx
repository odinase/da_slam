#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam_unstable/slam/PoseToPointFactor.h>
#include <gtsam/base/FastMap.h>
#include <iostream>
#include <utility>
#include <algorithm>
#include <memory>
#include <slam/types.h>
#include <limits>

#include <chrono>


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
      std::cout << "\n--------------- ASSOCIATE START -------------\n";
      std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

      gtsam::KeyList landmark_keys = estimates.filter(gtsam::Symbol::ChrTest('l')).keys();
      auto poses = estimates.filter(gtsam::Symbol::ChrTest('x'));
      int last_pose = poses.size() - 1; // Assuming first pose is 0
      gtsam::Key x_key = X(last_pose);
      POSE x_pose = estimates.at<POSE>(x_key);
      size_t num_measurements = measurements.size();
      size_t num_landmarks = landmark_keys.size();

      std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
      std::cout << "Initialization of div variables took " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;

    begin = std::chrono::steady_clock::now();

      gtsam::KeyVector keys;
      keys.push_back(x_key);
      for (const auto& lmk_key : landmark_keys) {
        keys.push_back(lmk_key);
      }

end = std::chrono::steady_clock::now();
      std::cout << "Building key vector took " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;

    begin = std::chrono::steady_clock::now();

      gtsam::JointMarginal joint_marginals = marginals.jointMarginalCovariance(keys);

      end = std::chrono::steady_clock::now();
      std::cout << "Making joint marginals took " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;

      begin = std::chrono::steady_clock::now();

      gtsam::Matrix Hx, Hl;

      // Map of landmarks that are individually compatible with at least one measurement, with NIS
      gtsam::FastMap<gtsam::Key, std::vector<std::pair<int, double>>> lmk_meas_asso_candidates;

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
            continue; // Landmark too far away to be relevant.
          }

          gtsam::PoseToPointFactor<POSE, POINT> factor(x_key, l, meas, noise);
          gtsam::Vector error = factor.evaluateError(x_pose, lmk, Hx, Hl);
          hypothesis::Association a(meas_idx, l, Hx, Hl, error);
          double nis = individual_compatability(a, x_key, joint_marginals, measurements);

          // Individually compatible?
          if (nis < mh_threshold_)
          {
            lmk_meas_asso_candidates[l].push_back({meas_idx, nis});
          }
        }
      }

      end = std::chrono::steady_clock::now();
      std::cout << "Looping over all measurements and landmarks took " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;

      begin = std::chrono::steady_clock::now();

      hypothesis::Hypothesis h = hypothesis::Hypothesis::empty_hypothesis();

      size_t num_assoed_lmks = lmk_meas_asso_candidates.size();

      // We found landmarks that can be associated, set up for auction algorithm
      if (num_assoed_lmks > 0)
      {
        // Build cost matrix
        gtsam::Matrix cost_matrix = gtsam::Matrix::Constant(
            num_measurements + num_assoed_lmks,
            num_assoed_lmks,
            -std::numeric_limits<double>::infinity());

        // Fill bottom diagonal with "dummy measurements" meaning they are unassigned.
        cost_matrix.bottomRows(num_assoed_lmks).diagonal() << gtsam::Vector::Constant(num_assoed_lmks, -10'000);

        // To keep track of what column in the cost matrix corresponds to what actual landmark
        std::vector<gtsam::Key> cost_mat_col_to_lmk;

        // Fill cost matrix based on valid associations
        int lmk_idx = 0;
        for (const auto &[lmk, meas_candidates] : lmk_meas_asso_candidates)
        {
          cost_mat_col_to_lmk.push_back(lmk);
          for (const auto &[meas_idx, nis] : meas_candidates)
          {
            // We use negative NIS to force the highest reward to be the one with the lowest nis
            cost_matrix(meas_idx, lmk_idx) = -nis;
          }
          lmk_idx++;
        }


        end = std::chrono::steady_clock::now();
        std::cout << "Building cost matrix took " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
      

        begin = std::chrono::steady_clock::now();
      
        std::vector<int> associated_landmarks = auction(cost_matrix);


        end = std::chrono::steady_clock::now();
        std::cout << "Auction algorithm took " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
      

        begin = std::chrono::steady_clock::now();

        double tot_reward = 0.0;
        double asso_reward = 0.0;

        for (int lmk_idx = 0; lmk_idx < num_assoed_lmks; lmk_idx++)
        {
          int meas_idx = associated_landmarks[lmk_idx];
          tot_reward += cost_matrix(meas_idx, lmk_idx);
          if (meas_idx == -1 || meas_idx >= num_measurements)
          {
            if (meas_idx == -1) {
              std::cout << "Landmark l" << gtsam::symbolIndex(cost_mat_col_to_lmk[lmk_idx]) << " associated to -1?\n";
            }
            continue; // Landmark associated with dummy measurement, so skip
          }
          gtsam::Key l = cost_mat_col_to_lmk[lmk_idx];
          POINT lmk = estimates.at<POINT>(l);

          asso_reward += cost_matrix(meas_idx, lmk_idx);

          const auto &meas = measurements[meas_idx].measurement;
          const auto &noise = measurements[meas_idx].noise;

          gtsam::PoseToPointFactor<POSE, POINT> factor(x_key, l, meas, noise);
          gtsam::Vector error = factor.evaluateError(x_pose, lmk, Hx, Hl);
          Association::shared_ptr a = std::make_shared<Association>(meas_idx, l, Hx, Hl, error);
          // double nis = individual_compatability(*a, x_key, marginals, measurements);
          h.extend(a);
        }

        end = std::chrono::steady_clock::now();
        std::cout << "Building hypothesis from auction solution took " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
      }

      // Regardless of if no or only some measurements were made, fill hypothesis with remaining unassociated measurements and return
      h.fill_with_unassociated_measurements(num_measurements);


// This computation is not needed right now
#ifdef HYPOTHESIS_QUALITY
      std::cout << "Computing joint NIS\n";
      double nis = joint_compatability<POSE::dimension, POINT::RowsAtCompileTime, POINT::RowsAtCompileTime>(h, x_key, marginals, measurements);
      h.set_nis(nis);
#endif
      std::cout << "--------------- ASSOCIATE end -------------\n\n";
      return h;
    }
  } // namespace ml
} // namespace da
