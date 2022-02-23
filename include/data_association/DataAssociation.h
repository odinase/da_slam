#pragma once

#include "data_association/Hypothesis.h"
#include <limits>
#include <memory>
#include <boost/math/distributions.hpp>
#include <deque>
namespace da
{

  template <class MEASUREMENT>
  class DataAssociation
  {
  public:
    // using shared_ptr = std::shared_ptr<DataAssociation<MEASUREMENT>>;
    // using unique_ptr = std::unique_ptr<DataAssociation<MEASUREMENT>>;

    virtual hypothesis::Hypothesis associate(
        const gtsam::Values &estimates,
        const gtsam::Marginals &marginals,
        const gtsam::FastVector<MEASUREMENT> &measurements) = 0;
    virtual ~DataAssociation() {}
  };

  enum class AssociationMethod
  {
    JCBB,
    ML,
    KnownDataAssociation
  };

  template <class MEASUREMENT>
  double individual_compatability(
      const hypothesis::Association &a,
      gtsam::Key x_key,
      const gtsam::Marginals &marginals,
      const gtsam::FastVector<MEASUREMENT> &measurements)
  {
    // Should never happen...
    if (!a.associated())
    {
      return std::numeric_limits<double>::infinity();
    }
    Eigen::VectorXd innov = a.error;
    Eigen::MatrixXd P = marginals.jointMarginalCovariance(gtsam::KeyVector{{x_key, *a.landmark}}).fullMatrix();
    // TODO: Fix here later
    int rows = a.Hx.rows();
    int cols = a.Hx.cols() + a.Hl.cols();
    Eigen::MatrixXd H(rows, cols);
    H << a.Hx, a.Hl;

    const auto &meas_noise = measurements[a.measurement].noise;
    Eigen::MatrixXd R = meas_noise->sigmas().array().square().matrix().asDiagonal();
    Eigen::MatrixXd S = H * P * H.transpose() + R;

    return innov.transpose() * S.llt().solve(innov);
  }

  template <class MEASUREMENT>
  double joint_compatability(
      const hypothesis::Hypothesis &h,
      gtsam::Key x_key,
      const gtsam::Marginals &marginals,
      const gtsam::FastVector<MEASUREMENT> &measurements,
      int state_dim,
      int lmk_dim,
      int meas_dim)
  {
    gtsam::KeyVector joint_states;
    joint_states.push_back(x_key);
    int num_associated_meas_to_lmk = 0;
    for (const auto &asso : h.associations())
    {
      if (asso->associated())
      {
        num_associated_meas_to_lmk++;
        joint_states.push_back(*asso->landmark);
      }
    }

    if (num_associated_meas_to_lmk == 0)
    {
      return std::numeric_limits<double>::infinity();
    }

    Eigen::MatrixXd Pjoint = marginals.jointMarginalCovariance(joint_states).fullMatrix();

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(num_associated_meas_to_lmk * meas_dim, state_dim + num_associated_meas_to_lmk * lmk_dim);
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(num_associated_meas_to_lmk * meas_dim, num_associated_meas_to_lmk * meas_dim);

    Eigen::VectorXd innov(num_associated_meas_to_lmk * meas_dim);
    int meas_idx = 0, lmk_idx = 0;

    for (const auto &a : h.associations())
    {
      if (a->associated())
      {
        innov.segment(meas_idx, meas_dim) = a->error;
        H.block(meas_idx, 0, meas_dim, state_dim) = a->Hx;
        H.block(meas_idx, state_dim + lmk_idx, meas_dim, lmk_dim) = a->Hl;

        const auto &meas_noise = measurements[a->measurement].noise;

        // Adding R might be done more cleverly
        R.block(meas_idx, meas_idx, meas_dim, meas_dim) = meas_noise->sigmas().array().square().matrix().asDiagonal();

        meas_idx += meas_dim;
        lmk_idx += lmk_dim;
      }
    }

    Eigen::MatrixXd Sjoint = H * Pjoint * H.transpose() + R;

    double nis = innov.transpose() * Sjoint.llt().solve(innov);
    return nis;
  }

  template <const unsigned int STATE_DIM,
            const unsigned int LANDMARK_DIM,
            const unsigned int MEASUREMENT_DIM,
            class MEASUREMENT>
  double joint_compatability(
      const hypothesis::Hypothesis &h,
      gtsam::Key x_key,
      const gtsam::Marginals &marginals,
      const gtsam::FastVector<MEASUREMENT> &measurements)
  {
    gtsam::KeyVector joint_states;
    joint_states.push_back(x_key);
    int num_associated_meas_to_lmk = 0;
    for (const auto &asso : h.associations())
    {
      if (asso->associated())
      {
        num_associated_meas_to_lmk++;
        joint_states.push_back(*asso->landmark);
      }
    }

    if (num_associated_meas_to_lmk == 0)
    {
      return std::numeric_limits<double>::infinity();
    }

    Eigen::MatrixXd Pjoint = marginals.jointMarginalCovariance(joint_states).fullMatrix();

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(num_associated_meas_to_lmk * MEASUREMENT_DIM, STATE_DIM + num_associated_meas_to_lmk * LANDMARK_DIM);
    Eigen::MatrixXd R = Eigen::MatrixXd::Zero(num_associated_meas_to_lmk * MEASUREMENT_DIM, num_associated_meas_to_lmk * MEASUREMENT_DIM);

    Eigen::VectorXd innov(num_associated_meas_to_lmk * MEASUREMENT_DIM);
    int meas_idx = 0, lmk_idx = 0;

    for (const auto &a : h.associations())
    {
      if (a->associated())
      {
        innov.segment(meas_idx, MEASUREMENT_DIM) = a->error;
        H.block(meas_idx, 0, MEASUREMENT_DIM, STATE_DIM) = a->Hx;
        H.block(meas_idx, STATE_DIM + lmk_idx, MEASUREMENT_DIM, LANDMARK_DIM) = a->Hl;

        const auto &meas_noise = measurements[a->measurement].noise;
        // Adding R might be done more cleverly
        R.block(meas_idx, meas_idx, MEASUREMENT_DIM, MEASUREMENT_DIM) = meas_noise->sigmas().array().square().matrix().asDiagonal();

        meas_idx += MEASUREMENT_DIM;
        lmk_idx += LANDMARK_DIM;
      }
    }

    Eigen::MatrixXd Sjoint = H * Pjoint * H.transpose() + R;

    double nis = innov.transpose() * Sjoint.llt().solve(innov);
    return nis;
  }

  double chi2inv(double p, unsigned int dim)
  {
    boost::math::chi_squared dist(dim);
    return quantile(dist, p);
  }

  std::vector<int> auction(const Eigen::MatrixXd& problem, double eps = 1e-3, uint64_t max_iterations = 10'000) {
    int m = problem.rows();
    int n = problem.cols();

    std::deque<int> unassigned_queue;
    std::vector<int> assigned_landmarks;

    // Initilize
    for (int i = 0; i < n; i++) {
      unassigned_queue.push_back(i);
      assigned_landmarks.push_back(-1);
    }

    // Use Eigen vector for convenience below
    Eigen::VectorXd prices(m);
    for (int i = 0; i < m; i++) {
      prices(i) = 0;
    }

    uint64_t curr_iter = 0;

    while (!unassigned_queue.empty()) {
      int l_star = unassigned_queue.front();
      unassigned_queue.pop_front();

      if (curr_iter > max_iterations) {
        break;
      }
      Eigen::MatrixXd::Index i_star;
      double val_max = (problem.col(l_star) - prices).maxCoeff(&i_star);

      auto prev_owner = std::find(assigned_landmarks.begin(), assigned_landmarks.end(), i_star);
      assigned_landmarks[l_star] = i_star;

      if (prev_owner != assigned_landmarks.end()) {
        // The item has a previous owner
        *prev_owner = -1;
        int pos = std::distance(assigned_landmarks.begin(), prev_owner);
        unassigned_queue.push_back(pos);
      }

      double y = problem(i_star, l_star) - val_max;
      prices(i_star) += y + eps;
      curr_iter++;
    }

    return assigned_landmarks;
  }

} // namespace data_association