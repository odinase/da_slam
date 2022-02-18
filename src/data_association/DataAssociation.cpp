#include "data_association/DataAssociation.h"
#include <Eigen/Core>
#include <gtsam/base/Matrix.h>
#include <gtsam/inference/Key.h>
#include <gtsam/nonlinear/Marginals.h>

namespace data_association
{
    double DataAssociation::individual_compatability(const Association &a) const
    {
        // Should never happen...
        if (!a.associated())
        {
            return std::numeric_limits<double>::infinity();
        }
        Eigen::VectorXd innov = a.error;
        Eigen::MatrixXd P = marginals_.jointMarginalCovariance(gtsam::KeyVector{{x_key_, *a.landmark}}).fullMatrix();
        // TODO: Fix here later
        int rows = a.Hx.rows();
        int cols = a.Hx.cols() + a.Hl.cols();
        Eigen::MatrixXd H(rows, cols);
        H << a.Hx, a.Hl;

        const auto &meas_noise = measurements_[a.measurement].noise;
        Eigen::MatrixXd R = meas_noise->sigmas().array().square().matrix().asDiagonal();
        Eigen::MatrixXd S = H * P * H.transpose() + R;

        return innov.transpose() * S.llt().solve(innov);
    }
} // namespace data_association