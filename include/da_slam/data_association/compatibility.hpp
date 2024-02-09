#ifndef DA_SLAM_DATA_ASSOCIATION_COMPATIBILITY_HPP
#define DA_SLAM_DATA_ASSOCIATION_COMPATIBILITY_HPP

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

namespace da_slam::data_association::compatibility {

template <typename Measurement>
double individual_compatibility(const hypothesis::Association& a, gtsam::Key x_key,
                                const gtsam::JointMarginal& joint_marginals,
                                const gtsam::FastVector<Measurement>& measurements,
                                std::optional<std::reference_wrapper<double>> log_norm_factor = {},
                                std::optional<Eigen::Ref<Eigen::MatrixXd>> S_ = {})
{
    const auto h_rows = a.Hx.rows();
    const auto h_cols = a.Hx.cols() + a.Hl.cols();
    Eigen::MatrixXd H(h_rows, h_cols);
    H << a.Hx, a.Hl;

    Eigen::MatrixXd Pxx = joint_marginals(x_key, x_key);
    Eigen::MatrixXd Pll = joint_marginals(*a.landmark, *a.landmark);
    Eigen::MatrixXd Pxl = joint_marginals(x_key, *a.landmark);
    const auto& Plx = Pxl.transpose();

    const auto p_rows = Pxx.rows() + Pll.rows();
    const auto p_cols = p_rows;

    Eigen::MatrixXd P(p_rows, p_cols);
    P << Pxx, Pxl,  //
        Plx, Pll;

    Eigen::MatrixXd R = measurements[a.measurement].noise->sigmas().array().square().matrix().asDiagonal();

    Eigen::MatrixXd S = H * P * H.transpose() + R;

    if (S_) {
        S_ = S;
    }

    const Eigen::VectorXd& innov = a.error;

    Eigen::LLT<Eigen::MatrixXd> chol = S.llt();

    if (log_norm_factor) {
        auto& L = chol.matrixL();
        log_norm_factor->get() = 2.0 * L.toDenseMatrix().diagonal().array().log().sum();
    }

    return innov.transpose() * chol.solve(innov);
}

template <class MEASUREMENT>
double joint_compatibility(const hypothesis::Hypothesis& h, gtsam::Key x_key, const gtsam::Marginals& marginals,
                           const gtsam::FastVector<MEASUREMENT>& measurements, int state_dim, int lmk_dim, int meas_dim)
{
    gtsam::KeyVector joint_states{};
    joint_states.push_back(x_key);
    int num_associated_meas_to_lmk = 0;
    for (const auto& asso : h.associations()) {
        if (asso->associated()) {
            num_associated_meas_to_lmk++;
            joint_states.push_back(*asso->landmark);
        }
    }

    if (num_associated_meas_to_lmk == 0) {
        return std::numeric_limits<double>::infinity();
    }

    Eigen::MatrixXd Pjoint = marginals.jointMarginalCovariance(joint_states).fullMatrix();

    Eigen::MatrixXd H =
        Eigen::MatrixXd::Zero(num_associated_meas_to_lmk * meas_dim, state_dim + num_associated_meas_to_lmk * lmk_dim);
    Eigen::MatrixXd R =
        Eigen::MatrixXd::Zero(num_associated_meas_to_lmk * meas_dim, num_associated_meas_to_lmk * meas_dim);

    Eigen::VectorXd innov(num_associated_meas_to_lmk * meas_dim);
    int meas_idx = 0, lmk_idx = 0;

    for (const auto& a : h.associations()) {
        if (a->associated()) {
            innov.segment(meas_idx, meas_dim) = a->error;
            H.block(meas_idx, 0, meas_dim, state_dim) = a->Hx;
            H.block(meas_idx, state_dim + lmk_idx, meas_dim, lmk_dim) = a->Hl;

            const auto& meas_noise = measurements[a->measurement].noise;

            // Adding R might be done more cleverly
            R.block(meas_idx, meas_idx, meas_dim, meas_dim) =
                meas_noise->sigmas().array().square().matrix().asDiagonal();

            meas_idx += meas_dim;
            lmk_idx += lmk_dim;
        }
    }

    Eigen::MatrixXd Sjoint = H * Pjoint * H.transpose() + R;

    double nis = innov.transpose() * Sjoint.llt().solve(innov);
    return nis;
}

template <int StateDim, int LandmarkDim, int MeasurementDim, typename Measurement>
double joint_compatibility(const hypothesis::Hypothesis& h, gtsam::Key x_key, const gtsam::Marginals& marginals,
                           const gtsam::FastVector<Measurement>& measurements)
{
    gtsam::KeyVector joint_states;
    joint_states.push_back(x_key);
    int num_associated_meas_to_lmk = 0;
    for (const auto& asso : h.associations()) {
        if (asso->associated()) {
            num_associated_meas_to_lmk++;
            joint_states.push_back(*asso->landmark);
        }
    }

    if (num_associated_meas_to_lmk == 0) {
        return std::numeric_limits<double>::infinity();
    }

    Eigen::MatrixXd Pjoint = marginals.jointMarginalCovariance(joint_states).fullMatrix();

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(num_associated_meas_to_lmk * MeasurementDim,
                                              StateDim + num_associated_meas_to_lmk * LandmarkDim);
    Eigen::MatrixXd R =
        Eigen::MatrixXd::Zero(num_associated_meas_to_lmk * MeasurementDim, num_associated_meas_to_lmk * MeasurementDim);

    Eigen::VectorXd innov(num_associated_meas_to_lmk * MeasurementDim);
    int meas_idx = 0, lmk_idx = 0;

    for (auto&& a : h.associations()) {
        if (a->associated()) {
            innov.segment(meas_idx, MeasurementDim) = a->error;
            H.block(meas_idx, 0, MeasurementDim, StateDim) = a->Hx;
            H.block(meas_idx, StateDim + lmk_idx, MeasurementDim, LandmarkDim) = a->Hl;

            const auto& meas_noise = measurements[a->measurement].noise;
            // Adding R might be done more cleverly
            R.block(meas_idx, meas_idx, MeasurementDim, MeasurementDim) =
                meas_noise->sigmas().array().square().matrix().asDiagonal();

            meas_idx += MeasurementDim;
            lmk_idx += LandmarkDim;
        }
    }

    Eigen::MatrixXd Sjoint = H * Pjoint * H.transpose() + R;

    double nis = innov.transpose() * Sjoint.llt().solve(innov);
    return nis;
}

} // namespace da_slam::data_association::compatibility

#endif // DA_SLAM_DATA_ASSOCIATION_COMPATIBILITY_HPP