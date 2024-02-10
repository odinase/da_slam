#ifndef DA_SLAM_DATA_ASSOCIATION_INNOVATION_HPP
#define DA_SLAM_DATA_ASSOCIATION_INNOVATION_HPP

#include <boost/math/distributions.hpp>
#include <cmath>
#include <deque>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <utility>

#include "da_slam/data_association/hypothesis.hpp"
#include "da_slam/types.hpp"
#include "gtsam/nonlinear/Marginals.h"

namespace da_slam::data_association::innovation
{

template <typename Pose, typename Point>
std::pair<gtsam::Vector, gtsam::Matrix> innovation(gtsam::Key x_key, gtsam::Key lmk_key, const Pose& x, const Point& l,
                                                   const gtsam::JointMarginal& joint_marginal,
                                                   const types::Measurement<Point>& measurement)
{
    gtsam::PoseToPointFactor<Pose, Point> factor(x_key, lmk_key, measurement.measurement, measurement.noise);
    gtsam::Matrix Hx{};
    gtsam::Matrix Hl{};
    gtsam::Vector innovation = factor.evaluateError(x, l, Hx, Hl);

    const auto h_rows = Hx.rows();
    const auto h_cols = Hx.cols() + Hl.cols();
    Eigen::MatrixXd H(h_rows, h_cols);
    H << Hx, Hl;

    Eigen::MatrixXd Pxx = joint_marginal(x_key, x_key);
    Eigen::MatrixXd Pll = joint_marginal(lmk_key, lmk_key);
    Eigen::MatrixXd Pxl = joint_marginal(x_key, lmk_key);
    const auto& Plx = Pxl.transpose();

    const auto p_rows = Pxx.rows() + Pll.rows();
    const auto p_cols = p_rows;

    Eigen::MatrixXd P(p_rows, p_cols);
    P << Pxx, Pxl,  //
        Plx, Pll;

    Eigen::MatrixXd R = measurement.noise->sigmas().array().square().matrix().asDiagonal();

    Eigen::MatrixXd S = H * P * H.transpose() + R;

    return {innovation, S};
}

template <typename Point>
auto innovation_covariance(const Eigen::MatrixXd& P, const Eigen::MatrixXd& H, const types::Measurement<Point>& meas)
    -> Eigen::Matrix<double, Point::RowsAtCompileTime, Point::RowsAtCompileTime>
{
    const Eigen::MatrixXd R = meas.noise->sigmas().array().square().matrix().asDiagonal();
    return H * P * H.transpose() + R;
}

template <typename Point>
auto innovation_covariance(gtsam::Key x_key, const gtsam::JointMarginal& joint_marginal,
                           const hypothesis::Association& asso, const types::Measurement<Point>& meas)
    -> Eigen::Matrix<double, Point::RowsAtCompileTime, Point::RowsAtCompileTime>
{
    // Get measurement Jacobian
    const auto h_rows = asso.Hx.rows();
    const auto h_cols = asso.Hx.cols() + asso.Hl.cols();
    gtsam::Matrix H(h_rows, h_cols);
    H << asso.Hx, asso.Hl;

    // Get joint state covariance
    gtsam::Key lmk_key = *asso.landmark;
    Eigen::MatrixXd Pxx = joint_marginal(x_key, x_key);
    Eigen::MatrixXd Pll = joint_marginal(lmk_key, lmk_key);
    Eigen::MatrixXd Pxl = joint_marginal(x_key, lmk_key);
    const auto& Plx = Pxl.transpose();

    const auto p_rows = Pxx.rows() + Pll.rows();
    const auto p_cols = p_rows;

    Eigen::MatrixXd P(p_rows, p_cols);
    P << Pxx, Pxl,  //
        Plx, Pll;

    const Eigen::MatrixXd R = meas.noise->sigmas().array().square().matrix().asDiagonal();
    return H * P * H.transpose() + R;
}

}  // namespace da_slam::data_association::innovation

#endif  // DA_SLAM_DATA_ASSOCIATION_INNOVATION_HPP