#ifndef DATA_ASSOCIATION_H
#define DATA_ASSOCIATION_H

#include "data_association/Hypothesis.h"
#include <limits>
#include <memory>
#include <boost/math/distributions.hpp>
#include <deque>
#include <iostream>
#include <cmath>
#include <utility>
#include "visualization/visualization.h"
#include "imgui.h"
#include "implot.h"
#include <functional>
#include <memory>
#include <optional>
#include "slam/types.h"


namespace da
{
  enum class AssociationMethod : int {
    MaximumLikelihood = 0,
    KnownDataAssociation = 1,
  };
}

std::ostream& operator<<(std::ostream& os, const da::AssociationMethod& asso_method);

namespace da {

  template <class MEASUREMENT>
  class DataAssociation
  {
  public:
    virtual hypothesis::Hypothesis associate(
        const gtsam::Values &estimates,
        const gtsam::Marginals &marginals,
        const gtsam::FastVector<MEASUREMENT> &measurements) = 0;
    virtual ~DataAssociation() {}
  };

  template <class MEASUREMENT>
  double individual_compatability(
      const hypothesis::Association &a,
      gtsam::Key x_key,
      const gtsam::JointMarginal &joint_marginals,
      const gtsam::FastVector<MEASUREMENT> &measurements,
      std::optional<std::reference_wrapper<double>> log_norm_factor = {},
      std::optional<Eigen::Ref<Eigen::MatrixXd>> S_ = {})
  {
    int rows = a.Hx.rows();
    int cols = a.Hx.cols() + a.Hl.cols();
    Eigen::MatrixXd H(rows, cols);
    H << a.Hx, a.Hl;

    Eigen::MatrixXd Pxx = joint_marginals(x_key, x_key);
    Eigen::MatrixXd Pll = joint_marginals(*a.landmark, *a.landmark);
    Eigen::MatrixXd Pxl = joint_marginals(x_key, *a.landmark);
    const auto &Plx = Pxl.transpose();

    rows = Pxx.rows() + Pll.rows();
    cols = rows;

    Eigen::MatrixXd P(rows, cols);
    P << Pxx, Pxl,
        Plx, Pll;

    Eigen::MatrixXd R = measurements[a.measurement].noise->sigmas().array().square().matrix().asDiagonal();

    Eigen::MatrixXd S = H * P * H.transpose() + R;

    if (S_)
    {
      S_ = S;
    }

    const Eigen::VectorXd &innov = a.error;

    Eigen::LLT<Eigen::MatrixXd> chol = S.llt();

    if (log_norm_factor)
    {
      auto &L = chol.matrixL();
      log_norm_factor->get() = 2.0 * L.toDenseMatrix().diagonal().array().log().sum();
    }

    return innov.transpose() * chol.solve(innov);
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

  template <class POSE, class POINT>
  std::pair<gtsam::Vector, gtsam::Matrix> innovation(
      gtsam::Key x_key,
      gtsam::Key lmk_key,
      const POSE& x,
      const POINT& l,
      const gtsam::JointMarginal &joint_marginal,
      const slam::Measurement<POINT> &measurement)
  {
    gtsam::Matrix Hx, Hl;
    gtsam::PoseToPointFactor<POSE, POINT> factor(x_key, lmk_key, measurement.measurement, measurement.noise);
    gtsam::Vector innovation = factor.evaluateError(x, l, Hx, Hl);

    int rows = Hx.rows();
    int cols = Hx.cols() + Hl.cols();
    Eigen::MatrixXd H(rows, cols);
    H << Hx, Hl;

    Eigen::MatrixXd Pxx = joint_marginal(x_key, x_key);
    Eigen::MatrixXd Pll = joint_marginal(lmk_key, lmk_key);
    Eigen::MatrixXd Pxl = joint_marginal(x_key, lmk_key);
    const auto &Plx = Pxl.transpose();

    rows = Pxx.rows() + Pll.rows();
    cols = rows;

    Eigen::MatrixXd P(rows, cols);
    P << Pxx, Pxl,
        Plx, Pll;

    Eigen::MatrixXd R = measurement.noise->sigmas().array().square().matrix().asDiagonal();

    Eigen::MatrixXd S = H * P * H.transpose() + R;

    return {innovation, S};
  }

  double chi2inv(double p, unsigned int dim);
  std::vector<int> auction(const Eigen::MatrixXd &problem, double eps = 1e-3, uint64_t max_iterations = 10'000);
  std::vector<int> hungarian(const Eigen::MatrixXd &cost_matrix);

} // namespace da

#endif // DATA_ASSOCIATION_H