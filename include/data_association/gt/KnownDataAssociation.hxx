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
#include "data_association/DataAssociation.h"


namespace da
{

  namespace gt
  {
    using gtsam::symbol_shorthand::L;
    using gtsam::symbol_shorthand::X;

    template <class POSE, class POINT>
    KnownDataAssociation<POSE, POINT>::KnownDataAssociation(const std::map<uint64_t, gtsam::Key> &meas_lmk_assos) 
    : meas_lmk_assos_(meas_lmk_assos),
    curr_landmark_count_(0)
    {
    }

    template <class POSE, class POINT>
    Hypothesis KnownDataAssociation<POSE, POINT>::associate(
        const gtsam::Values &estimates,
        const gtsam::Marginals &marginals,
        const gtsam::FastVector<slam::Measurement<POINT>> &measurements)
    {
      // gtsam::KeyList landmark_keys = estimates.filter(gtsam::Symbol::ChrTest('l')).keys();
      auto poses = estimates.filter(gtsam::Symbol::ChrTest('x'));
      int last_pose = poses.size() - 1; // Assuming first pose is 0
      gtsam::Key x_key = X(last_pose);
      POSE x_pose = estimates.at<POSE>(x_key);
      // size_t num_measurements = measurements.size();
      // size_t num_landmarks = landmark_keys.size();

      // Make hypothesis to return later
      hypothesis::Hypothesis h = hypothesis::Hypothesis::empty_hypothesis();

      /*
      * Look through all measurements, and find landmark they should be associated with.
      * If this landmark key does not exist in the mapping between ground truth map and our map, leave measurement unassociated.
      * Otherwise, associate.
      */

      gtsam::Matrix Hx, Hl;

      size_t tot_measurements = measurements.size();
      for (int meas_idx = 0; meas_idx < tot_measurements; meas_idx++)
      {
        const auto &measurement = measurements[meas_idx];
        gtsam::Key lmk_gt = meas_lmk_assos_[measurement.idx];
        const auto lmk_mapping_it = gt_lmk2map_lmk_.find(lmk_gt);

        // If we find the mapping, associate to it
        if (lmk_mapping_it != gt_lmk2map_lmk_.end())
        {
// #ifdef LOGGING
//           std::cout << "Found ground truth landmark " << gtsam::Symbol(lmk_gt);
// #endif // LOGGING
          gtsam::Key l = lmk_mapping_it->second;
          POINT lmk = estimates.at<POINT>(l);
          const auto &meas = measurement.measurement;
          const auto &noise = measurement.noise;

          gtsam::PoseToPointFactor<POSE, POINT> factor(x_key, l, meas, noise);
          gtsam::Vector error = factor.evaluateError(x_pose, lmk, Hx, Hl);
          Association::shared_ptr a = std::make_shared<Association>(meas_idx, l, Hx, Hl, error);
          h.extend(a);
        }
        // We have not seen this landmark before - add to mapping 
        else {
          // TODO: Not sure if keeping track of landmark count internally is a good way of doing this, but ohwell
          gt_lmk2map_lmk_[lmk_gt] = L(curr_landmark_count_);
          curr_landmark_count_++;
          Association::shared_ptr a = std::make_shared<Association>(meas_idx);
          h.extend(a);
        }
      }

#ifdef LOGGING
      std::cout << "\n\n";
      std::cout << "KnownDataAssociations made associations:\n";
      for (const auto& asso : h.associations()) {
        if (asso->associated()) {
        std::cout << "Measurement z" << measurements[asso->measurement].idx << " associated with landmark " << gtsam::Symbol(*asso->landmark) << "\n";
        } else {
          std::cout << "Measurement z" << measurements[asso->measurement].idx << " unassociated, initialized landmark " << gtsam::Symbol(gt_lmk2map_lmk_.find(meas_lmk_assos_[measurements[asso->measurement].idx])->second) << "\n";
        }
      }
      std::cout << "\n";
#endif // LOGGING

      return h;
    }

  } // namespace gt
} // namespace da
