#include "data_association/DataAssociation.h"
#include <Eigen/Core>
#include <gtsam/base/Matrix.h>
#include <gtsam/inference/Key.h>
#include <gtsam/nonlinear/Marginals.h>

namespace data_association
{

  double joint_compatability(const hypothesis::Hypothesis &h) const
  {
    int N = h.num_associations();
    int n = POSE::dimension; // Pose dimension
    int m = POINT::RowsAtCompileTime; // Landmark dimension
    int d = POINT::RowsAtCompileTime; // Measurement dimension
    // int n = gtsam::Pose3::dimension;
    // int m = gtsam::Pose3::dimension;
    // int d = gtsam::Pose3::dimension;

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
        R.block(k, k, d, d) = meas_noise_->sigmas().array().square().matrix().asDiagonal();

        k += d;
        j += m;
      }
    }

    Eigen::MatrixXd Sjoint = H * Pjoint * H.transpose() + R;

    double nis = innov.transpose() * Sjoint.llt().solve(innov);
    return nis;
  }

} // namespace data_association