#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam_unstable/slam/PoseToPointFactor.h>
#include <iostream>
#include <utility>
#include <algorithm>
#include <memory>
#include <slam/types.h>

namespace da
{

  namespace ml
  {
    using gtsam::symbol_shorthand::L;
    using gtsam::symbol_shorthand::X;

    template <class POSE, class POINT>
    MaximumLikelihood<POSE, POINT>::MaximumLikelihood(double sigmas, double range_threshold)
        : mh_threshold_(sigmas*sigmas),
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

      // First loop over all measurements, and find the lowest Mahalanobis distance
      gtsam::FastMap<gtsam::Key, gtsam::FastVector<std::pair<int, double>>> lmk_measurement_assos;
      gtsam::Matrix Hx, Hl;

      for (int i = 0; i < measurements.size(); i++)
      {
        const auto &meas = measurements[i].measurement;
        POINT meas_world = x_pose * meas;
        const auto &noise = measurements[i].noise;

        double lowest_nis = std::numeric_limits<double>::infinity();

        std::pair<int, double> smallest_innovation(-1, 0.0);
        for (const auto &l : landmark_keys)
        {
          POINT lmk = estimates.at<POINT>(l);
          if ((meas_world - lmk).norm() > range_threshold_)
          {
            continue; // Landmark too far away to be relevant.
          }
          gtsam::PoseToPointFactor<POSE, POINT> factor(x_key, l, meas, noise);
          gtsam::Vector error = factor.evaluateError(x_pose, lmk, Hx, Hl);
          hypothesis::Association a(i, l, Hx, Hl, error);
          double nis = individual_compatability(a, x_key, marginals, measurements);

          // Individually compatible?
          if (nis < mh_threshold_)
          {
            // Better association than already found?
            if (nis < lowest_nis)
            {
              lowest_nis = nis;
              smallest_innovation = {gtsam::symbolIndex(l), nis};
            }
          }
        }
        // We found a valid association
        if (smallest_innovation.first != -1)
        {
          lmk_measurement_assos[L(smallest_innovation.first)].push_back({i, smallest_innovation.second});
        }
      }

      // Loop over all landmarks with measurement associated with it and pick put the best one by pruning out all measurements except the best one in terms of Mahalanobis distance.
      hypothesis::Hypothesis h = hypothesis::Hypothesis::empty_hypothesis();
      for (const auto &[l, ms] : lmk_measurement_assos)
      {
        auto p = std::min_element(ms.begin(), ms.end(), [](const auto &p1, const auto &p2)
                                  { return p1.second < p2.second; });
        // Pretty redundant to do full recomputation here, but oh well
        POINT lmk = estimates.at<POINT>(l);
        const auto &meas = measurements[p->first].measurement;
        const auto &noise = measurements[p->first].noise;
        gtsam::PoseToPointFactor<POSE, POINT> factor(x_key, l, meas, noise);
        gtsam::Vector error = factor.evaluateError(x_pose, lmk, Hx, Hl);
        Association::shared_ptr a = std::make_shared<Association>(p->first, l, Hx, Hl, error);
        h.extend(a);
      }
      h.fill_with_unassociated_measurements(measurements.size());

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
