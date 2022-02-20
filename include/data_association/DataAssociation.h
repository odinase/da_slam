#pragma once

#include "data_association/Hypothesis.h"
#include <memory>

namespace da
{

    template <class MEASUREMENT>
    class DataAssociation
    {
    public:
        typedef std::shared_ptr<DataAssociation> shared_ptr;
        typedef std::unique_ptr<DataAssociation> unique_ptr;

        virtual hypothesis::Hypothesis associate(
            const gtsam::Values &estimates,
            const gtsam::Marginals &marginals,
            const gtsam::FastVector<MEASUREMENT> &measurements) = 0;
        virtual ~DataAssociation();
    };

    enum class AssociationMethod
    {
        JCBB,
        ML,
        KnownDataAssociation
    };


    template<class MEASUREMENT>
    double individual_compatability(
            const hypothesis::Association &a,
            gtsam::Key x_key,
            const gtsam::Marginals& marginals,
            const gtsam::FastVector<MEASUREMENT>& measurements
        )
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



  double joint_compatability(
      const Hypothesis &h
    );
  {
    int N = h.num_associations();
    int n = gtsam::Pose3::dimension;
    int m = gtsam::Pose3::dimension;
    int d = gtsam::Pose3::dimension;

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