#include "da_slam/data_association/maximum_likelihood.hpp"

// #include <gtsam/base/FastVector.h>
// #include <gtsam/geometry/Pose3.h>

// #include <Eigen/Core>
// #include <da_slam/fmt.hpp>
// #include <iostream>
// #include <limits>
// #include <numeric>
// #include <vector>


// #include "da_slam/data_association/assignment_solvers/assignment_solver_interface.hpp"
// #include "da_slam/data_association/data_association_interface.hpp"
// #include "da_slam/data_association/hypothesis.hpp"
#include "da_slam/slam/utils_g2o.hpp"
#include <spdlog/spdlog.h>
#include "da_slam/data_association/compatibility.hpp"
// #include "da_slam/types.hpp"

using gtsam::symbol_shorthand::X;

namespace da_slam::data_association::maximum_likelihood
{

template <typename Pose, typename Point>
hypothesis::Hypothesis MaximumLikelihood<Pose, Point>::associate(
    const gtsam::Values& estimates, const gtsam::Marginals& marginals,
    const gtsam::FastVector<types::Measurement<Point>>& measurements) const
{
    const auto landmark_keys = gtsam::findLmKeys(estimates);
    const auto num_poses = gtsam::num_poses(estimates);
    const auto last_pose = num_poses - 1;  // Assuming first pose is 0
    gtsam::Key x_key = X(last_pose);
    const auto x_pose = estimates.at<Pose>(x_key);
    size_t num_measurements = measurements.size();
    size_t num_landmarks = landmark_keys.size();

    // Make hypothesis to return later
    hypothesis::Hypothesis h = hypothesis::Hypothesis::empty_hypothesis();

    // If no landmarks, return immediately
    if (num_landmarks == 0) {
        h.fill_with_unassociated_measurements(num_measurements);
        return h;
    }

#ifdef PROFILING
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Initialization of div variables took "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;

    begin = std::chrono::steady_clock::now();
#endif

    gtsam::Matrix Hx, Hl;
    gtsam::KeyVector keys;
    keys.push_back(x_key);

    for (int meas_idx = 0; meas_idx < num_measurements; meas_idx++) {
        const auto& meas = measurements[meas_idx].measurement;
        const auto meas_world = x_pose * meas;
        const auto& noise = measurements[meas_idx].noise;

        for (int lmk_idx = 0; lmk_idx < num_landmarks; lmk_idx++) {
            const auto l = L(lmk_idx);
            const auto lmk = estimates.at<Point>(l);
            if ((meas_world - lmk).norm() <= m_range_threshold) {
                // Key not already in vector
                if (std::find(keys.begin(), keys.end(), l) == keys.end()) {
                    keys.push_back(l);
                }
            }
        }
    }

#ifdef PROFILING
    end = std::chrono::steady_clock::now();
    std::cout << "Building key vector took "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
#endif

    // If no landmarks are close enough, terminate
    if (keys.size() == 1) {
        spdlog::info("No landmarks close enough to measurements, terminating!");
#ifdef LOGGING
        std::cout << "No landmarks close enough to measurements, terminating!\n";
#endif
        h.fill_with_unassociated_measurements(num_measurements);
        return h;
    }

    gtsam::JointMarginal joint_marginals = marginals.jointMarginalCovariance(keys);

    // Map of landmarks that are individually compatible with at least one measurement, with NIS
    gtsam::FastMap<gtsam::Key, std::vector<std::pair<int, double>>> lmk_meas_asso_candidates;

    for (int meas_idx = 0; meas_idx < num_measurements; meas_idx++) {
        const auto& meas = measurements[meas_idx].measurement;
        const auto meas_world = x_pose * meas;
        const auto& noise = measurements[meas_idx].noise;

        // Start iteration at second element as the first one is state
        for (size_t i = 1; i < keys.size(); i++) {
            gtsam::Key l = keys.at(i);
            const auto lmk = estimates.at<Point>(l);
            gtsam::PoseToPointFactor<Pose, Point> factor(x_key, l, meas, noise);
            gtsam::Vector error = factor.evaluateError(x_pose, lmk, Hx, Hl);
            hypothesis::Association a(meas_idx, l, Hx, Hl, error);
            double log_norm_factor;
            double mh_dist = compatibility::individual_compatibility(a, x_key, joint_marginals, measurements, log_norm_factor);

            double mle_cost = mh_dist + log_norm_factor;

            // Individually compatible?
            if (mh_dist < m_mh_threshold) {
                lmk_meas_asso_candidates[l].push_back({meas_idx, mle_cost});
            }
        }
    }

#ifdef PROFILING
    end = std::chrono::steady_clock::now();
    std::cout << "Computing individual compatibility took "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;

    begin = std::chrono::steady_clock::now();
#endif

    size_t num_assoed_lmks = lmk_meas_asso_candidates.size();

    // We found landmarks that can be associated, set up for auction algorithm
    if (num_assoed_lmks > 0) {
        // Build cost matrix
        gtsam::Matrix cost_matrix = gtsam::Matrix::Constant(num_measurements, num_assoed_lmks + num_measurements,
                                                            std::numeric_limits<double>::infinity());

        // Fill bottom diagonal with "dummy measurements" meaning they are unassigned.
        cost_matrix.rightCols(num_measurements).diagonal().array() = 10'000;

        // To keep track of what column in the cost matrix corresponds to what actual landmark
        std::vector<gtsam::Key> cost_mat_col_to_lmk;

        // Fill cost matrix based on valid associations
        int lmk_idx = 0;
        double lowest_mle_cost = std::numeric_limits<double>::infinity();

        for (const auto& [lmk, meas_candidates] : lmk_meas_asso_candidates) {
            cost_mat_col_to_lmk.push_back(lmk);
            for (const auto& [meas_idx, mle_cost] : meas_candidates) {
                cost_matrix(meas_idx, lmk_idx) = mle_cost;
                if (mle_cost < lowest_mle_cost) {
                    lowest_mle_cost = mle_cost;
                }
            }
            lmk_idx++;
        }

#ifdef PROFILING
        end = std::chrono::steady_clock::now();
        std::cout << "Building cost matrix took "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
#endif

        cost_matrix.array() -= lowest_mle_cost;  // We subtract lowest to ensure all costs are nonnegative

#ifdef PROFILING
        begin = std::chrono::steady_clock::now();
#endif

        std::vector<int> associated_measurements = m_assignment_solver->solve(cost_matrix);

#ifdef PROFILING
        end = std::chrono::steady_clock::now();
        std::cout << "Hungarian algorithm took "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
#endif

#ifdef LOGGING
        for (int m = 0; m < associated_measurements.size(); m++) {
            std::cout << "Measurement " << m << " associated with ";
            if (associated_measurements[m] < num_assoed_lmks) {
                std::cout << " landmark " << associated_measurements[m];
            }
            else {
                std::cout << "no landmark";
            }
            std::cout << "\n";
        }
#endif

#ifdef PROFILING
        begin = std::chrono::steady_clock::now();
#endif

        for (int meas_idx = 0; meas_idx < num_measurements; meas_idx++) {
            int lmk_idx = associated_measurements[meas_idx];
            if (lmk_idx == -1 || lmk_idx >= num_assoed_lmks) {
                continue;  // Measurement associated with dummy landmark, so skip
            }
            gtsam::Key l = cost_mat_col_to_lmk[lmk_idx];
            const auto lmk = estimates.at<Point>(l);

            const auto& meas = measurements[meas_idx].measurement;
            const auto& noise = measurements[meas_idx].noise;

            gtsam::PoseToPointFactor<Pose, Point> factor(x_key, l, meas, noise);
            gtsam::Vector error = factor.evaluateError(x_pose, lmk, Hx, Hl);
            auto a = std::make_shared<hypothesis::Association>(meas_idx, l, Hx, Hl, error);
            // auto [nis, log_norm_factor] = individual_compatability(*a, x_key, joint_marginals, measurements);

            h.extend(a);
        }

#ifdef PROFILING
        end = std::chrono::steady_clock::now();
        std::cout << "Building hypothesis from auction solution took "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
#endif
    }

    // Regardless of if no or only some measurements were made, fill hypothesis with remaining unassociated
    // measurements and return
    h.fill_with_unassociated_measurements(num_measurements);

#ifdef LOGGING
    std::cout << "\n\nMaximum likelihood made associations:\n";
    for (const auto& asso : h.associations()) {
        if (asso->associated()) {
            std::cout << "Measurement z" << measurements[asso->measurement].idx << " associated with landmark "
                      << gtsam::Symbol(*asso->landmark) << "\n";
        }
        else {
            std::cout << "Measurement z" << measurements[asso->measurement].idx << " unassociated\n";
        }
    }
    std::cout << "\n";
#endif  // LOGGING

#ifdef HYPOTHESIS_QUALITY
    std::cout << "Computing joint NIS\n";
    double nis = joint_compatability<Pose::dimension, Point::RowsAtCompileTime, Point::RowsAtCompileTime>(
        h, x_key, marginals, measurements);
    h.set_nis(nis);
#endif
    return h;
}

template class MaximumLikelihood<gtsam::Pose2, gtsam::Point2>;
template class MaximumLikelihood<gtsam::Pose3, gtsam::Point3>;

}  // namespace da_slam::data_association::maximum_likelihood
