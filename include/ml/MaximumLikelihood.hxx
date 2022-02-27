#include "ml/MaximumLikelihood.h"
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam_unstable/slam/PoseToPointFactor.h>
#include <iostream>
#include "jcbb/jcbb.h"
#include <utility>
#include <algorithm>
#include <memory>
#include <slam/types.h>

namespace ml
{
  using gtsam::symbol_shorthand::L;
  using gtsam::symbol_shorthand::X;

  template<class POSE, class POINT>
  MaximumLikelihood<POSE, POINT>::MaximumLikelihood(const gtsam::Values &estimates, const gtsam::Marginals &marginals, const gtsam::FastVector<slam::Measurement<POINT>> &measurements, double ic_prob, double range_threshold)
      : estimates_(estimates),
        marginals_(marginals),
        measurements_(measurements),
        ic_prob_(ic_prob),
        range_threshold_(range_threshold)
  {
    landmark_keys_ = estimates_.filter(gtsam::Symbol::ChrTest('l')).keys();
    auto poses = estimates_.filter(gtsam::Symbol::ChrTest('x'));
    int last_pose = poses.size() - 1; // Assuming first pose is 0
    x_key_ = X(last_pose);
    x_pose_ = estimates.at<POSE>(x_key_);
  }

  template<class POSE, class POINT>
  double MaximumLikelihood<POSE, POINT>::joint_compatability(const Hypothesis &h) const
  {
    int N = h.num_associations();
    int n = POSE::dimension; // Pose dimension
    int m = POINT::RowsAtCompileTime; // Landmark dimension
    int d = POINT::RowsAtCompileTime; // Measurement dimension

    gtsam::KeyVector joint_states;
    joint_states.push_back(x_key_);
    int num_associated_meas_to_lmk = 0;
    for (const auto &asso : h.associations())
    {
      if (asso->associated())
      {
        num_associated_meas_to_lmk++;
        joint_states.push_back(*asso->landmark);
      }
    }

    Eigen::MatrixXd Pjoint = marginals_.jointMarginalCovariance(joint_states).fullMatrix();

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(num_associated_meas_to_lmk * d, n + num_associated_meas_to_lmk * m);
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(num_associated_meas_to_lmk * d, num_associated_meas_to_lmk * d);

    Eigen::VectorXd innov(N * d);
    int k = 0, j = 0;

    for (const auto &a : h.associations())
    {
      if (a->associated())
      {
        innov.segment(k, d) = a->error;
        H.block(k, 0, d, n) = a->Hx;
        H.block(k, n + j, d, m) = a->Hl;

        // Adding R might be done more cleverly
        const auto& meas_noise = measurements_[a->measurement].noise;
        R.block(k, k, d, d) = meas_noise->sigmas().array().square().matrix().asDiagonal();

        k += d;
        j += m;
      }
    }

    Eigen::MatrixXd Sjoint = H * Pjoint * H.transpose() + R;

    double nis = innov.transpose() * Sjoint.llt().solve(innov);
    return nis;
  }

  template<class POSE, class POINT>
  std::pair<double, double> MaximumLikelihood<POSE, POINT>::individual_compatability(const Association &a) const
  {
    Eigen::VectorXd innov = a.error;
    Eigen::MatrixXd P = marginals_.jointMarginalCovariance(gtsam::KeyVector{{x_key_, *a.landmark}}).fullMatrix();
    // TODO: Fix here later
    int rows = a.Hx.rows();
    int cols = a.Hx.cols() + a.Hl.cols();
    Eigen::MatrixXd H(rows, cols);
    H << a.Hx, a.Hl;

    const auto& meas_noise = measurements_[a.measurement].noise;
    Eigen::MatrixXd R = meas_noise->sigmas().array().square().matrix().asDiagonal();
    Eigen::MatrixXd S = H * P * H.transpose() + R;

    Eigen::LLT<Eigen::MatrixXd> chol = S.llt();
    auto& L = chol.matrixL();
    double log_norm_factor = 2.0*L.toDenseMatrix().diagonal().array().log().sum();

    double mh_dist = innov.transpose() * chol.solve(innov); 

    return {mh_dist, log_norm_factor};
  }

  template<class POSE, class POINT>
  Hypothesis MaximumLikelihood<POSE, POINT>::associate() const
  {
    // First loop over all measurements, and find the lowest Mahalanobis distance
    gtsam::FastMap<gtsam::Key, gtsam::FastVector<std::pair<int, double>>> lmk_measurement_assos;
    gtsam::Matrix Hx, Hl;

    for (int i = 0; i < measurements_.size(); i++)
    {
      const auto &meas = measurements_[i].measurement;
      POINT meas_world = x_pose_ * meas;
      const auto& noise = measurements_[i].noise;

      double lowest_mle_cost = std::numeric_limits<double>::infinity();

      std::pair<int, double> smallest_innovation(-1, 0.0);
      for (const auto &l : landmark_keys_)
      {
        POINT lmk = estimates_.at<POINT>(l);
        if ((meas_world - lmk).norm() > range_threshold_) {
          continue; // Landmark too far away to be relevant.
        }
        gtsam::PoseToPointFactor<POSE, POINT> factor(x_key_, l, meas, noise);
        gtsam::Vector error = factor.evaluateError(x_pose_, lmk, Hx, Hl);
        Association a(i, l, Hx, Hl, error);
        auto [mh_dist, log_norm_factor] = individual_compatability(a);
        double mle_cost = mh_dist + log_norm_factor;

        // TODO: Refactor out things not JCBB from jcbb
        double mh_threshold = jcbb::chi2inv(1 - ic_prob_, POINT::RowsAtCompileTime);
        // Individually compatible?
        if (mh_dist < mh_threshold)
        {
          // Better association than already found?
          if (mle_cost < lowest_mle_cost)
          {
            lowest_mle_cost = mle_cost;
            smallest_innovation = {gtsam::symbolIndex(l), mle_cost};
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
    Hypothesis h = Hypothesis::empty_hypothesis();
    for (const auto &[l, meas_candidates] : lmk_measurement_assos)
    {
      auto best_candidate = std::min_element(meas_candidates.begin(), meas_candidates.end(), [](const auto &lhs, const auto &rhs)
                                { return lhs.second < rhs.second; });
      // Pretty redundant to do full recomputation here, but oh well
      int best_meas = best_candidate->first;
      POINT lmk = estimates_.at<POINT>(l);
      const auto &meas = measurements_[best_meas].measurement;
      const auto &noise = measurements_[best_meas].noise;
      gtsam::PoseToPointFactor<POSE, POINT> factor(x_key_, l, meas, noise);
      gtsam::Vector error = factor.evaluateError(x_pose_, lmk, Hx, Hl);
      Association::shared_ptr a = std::make_shared<Association>(best_meas, l, Hx, Hl, error);
      h.extend(a);
    }
    h.fill_with_unassociated_measurements(measurements_.size());

    double nis = joint_compatability(h);
    h.set_nis(nis);

    return h;
  }
} // namespace ml