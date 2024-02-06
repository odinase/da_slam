#include <gtsam/base/FastMap.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam_unstable/slam/PoseToPointFactor.h>
#include <slam/types.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <memory>
#include <utility>

#include "data_association/DataAssociation.h"

namespace da
{

namespace ml
{
using gtsam::symbol_shorthand::L;
using gtsam::symbol_shorthand::X;

template <class POSE, class POINT>
MaximumLikelihood<POSE, POINT>::MaximumLikelihood(double sigmas, double range_threshold)
: mh_threshold_(sigmas * sigmas), range_threshold_(range_threshold), sigmas_(sigmas)
{
}

template <class POSE, class POINT>
Hypothesis MaximumLikelihood<POSE, POINT>::associate(const gtsam::Values& estimates, const gtsam::Marginals& marginals,
                                                     const gtsam::FastVector<slam::Measurement<POINT>>& measurements)
{
#ifdef PROFILING
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
#endif

    const auto landmark_keys = estimates.extract<POINT>(gtsam::Symbol::ChrTest('l'))  //
                               | ranges::views::keys                                  //
                               | ranges::to<std::vector<gtsam::Key>>();
    const auto num_poses = estimates.extract<POSE>(gtsam::Symbol::ChrTest('x')).size();
    int last_pose = num_poses - 1;  // Assuming first pose is 0
    gtsam::Key x_key = X(last_pose);
    POSE x_pose = estimates.at<POSE>(x_key);
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
        POINT meas_world = x_pose * meas;
        const auto& noise = measurements[meas_idx].noise;

        for (int lmk_idx = 0; lmk_idx < num_landmarks; lmk_idx++) {
            gtsam::Key l = L(lmk_idx);
            POINT lmk = estimates.at<POINT>(l);
            if ((meas_world - lmk).norm() <= range_threshold_) {
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
#ifdef LOGGING
        std::cout << "No landmarks close enough to measurements, terminating!\n";
#endif
        h.fill_with_unassociated_measurements(num_measurements);
        return h;
    }

#ifdef PROFILING
    begin = std::chrono::steady_clock::now();
#endif

    gtsam::JointMarginal joint_marginals = marginals.jointMarginalCovariance(keys);

#ifdef PROFILING
    end = std::chrono::steady_clock::now();
    std::cout << "Making joint marginals took "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;

    begin = std::chrono::steady_clock::now();
#endif

    // Map of landmarks that are individually compatible with at least one measurement, with NIS
    gtsam::FastMap<gtsam::Key, std::vector<std::pair<int, double>>> lmk_meas_asso_candidates;

    for (int meas_idx = 0; meas_idx < num_measurements; meas_idx++) {
        const auto& meas = measurements[meas_idx].measurement;
        POINT meas_world = x_pose * meas;
        const auto& noise = measurements[meas_idx].noise;

        // Start iteration at second element as the first one is state
        for (int i = 1; i < keys.size(); i++) {
            gtsam::Key l = keys[i];
            POINT lmk = estimates.at<POINT>(l);
            gtsam::PoseToPointFactor<POSE, POINT> factor(x_key, l, meas, noise);
            gtsam::Vector error = factor.evaluateError(x_pose, lmk, Hx, Hl);
            hypothesis::Association a(meas_idx, l, Hx, Hl, error);
            double log_norm_factor;
            double mh_dist = individual_compatability(a, x_key, joint_marginals, measurements, log_norm_factor);

            double mle_cost = mh_dist + log_norm_factor;

            // Individually compatible?
            if (mh_dist < mh_threshold_) {
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

        std::vector<int> associated_measurements = hungarian(cost_matrix);

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
            POINT lmk = estimates.at<POINT>(l);

            const auto& meas = measurements[meas_idx].measurement;
            const auto& noise = measurements[meas_idx].noise;

            gtsam::PoseToPointFactor<POSE, POINT> factor(x_key, l, meas, noise);
            gtsam::Vector error = factor.evaluateError(x_pose, lmk, Hx, Hl);
            Association::shared_ptr a = std::make_shared<Association>(meas_idx, l, Hx, Hl, error);
            // auto [nis, log_norm_factor] = individual_compatability(*a, x_key, joint_marginals, measurements);

            h.extend(a);
        }

#ifdef PROFILING
        end = std::chrono::steady_clock::now();
        std::cout << "Building hypothesis from auction solution took "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
#endif
    }

    // Regardless of if no or only some measurements were made, fill hypothesis with remaining unassociated measurements
    // and return
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
    double nis = joint_compatability<POSE::dimension, POINT::RowsAtCompileTime, POINT::RowsAtCompileTime>(
        h, x_key, marginals, measurements);
    h.set_nis(nis);
#endif
    return h;
}

template <class POSE, class POINT>
Hypothesis MaximumLikelihood<POSE, POINT>::associate_bad(
    const gtsam::Values& estimates, const gtsam::Marginals& marginals,
    const gtsam::FastVector<slam::Measurement<POINT>>& measurements)
{
#ifdef PROFILING
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
#endif

    const auto landmark_keys = estimates.extract<POINT>(gtsam::Symbol::ChrTest('l'))  //
                               | ranges::views::keys                                  //
                               | ranges::to<gtsam::KeyList>();
    auto poses = estimates.extract<POSE>(gtsam::Symbol::ChrTest('x'))                                   //
                 | ranges::actions::sort([](auto&& lhs, auto&& rhs) { return lhs.first < rhs.first; })  //
                 | ranges::views::values                                                                //
                 | ranges::to<std::vector<POSE>>();
    int last_pose = poses.size() - 1;  // Assuming first pose is 0
    gtsam::Key x_key = X(last_pose);
    POSE x_pose = estimates.at<POSE>(x_key);
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
        POINT meas_world = x_pose * meas;
        const auto& noise = measurements[meas_idx].noise;

        for (int lmk_idx = 0; lmk_idx < num_landmarks; lmk_idx++) {
            gtsam::Key l = L(lmk_idx);
            POINT lmk = estimates.at<POINT>(l);
            if ((meas_world - lmk).norm() <= range_threshold_) {
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
#ifdef LOGGING
        std::cout << "No landmarks close enough to measurements, terminating!\n";
#endif
        return h;
    }

#ifdef PROFILING
    begin = std::chrono::steady_clock::now();
#endif

    gtsam::JointMarginal joint_marginals = marginals.jointMarginalCovariance(keys);

#ifdef PROFILING
    end = std::chrono::steady_clock::now();
    std::cout << "Making joint marginals took "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;

    begin = std::chrono::steady_clock::now();
#endif

    // Map of landmarks that are individually compatible with at least one measurement, with NIS
    gtsam::FastMap<gtsam::Key, std::vector<std::pair<int, double>>> lmk_meas_asso_candidates;

    for (int meas_idx = 0; meas_idx < num_measurements; meas_idx++) {
        const auto& meas = measurements[meas_idx].measurement;
        POINT meas_world = x_pose * meas;
        const auto& noise = measurements[meas_idx].noise;

        double lowest_mle_cost = std::numeric_limits<double>::infinity();

        std::pair<int, double> smallest_innovation(-1, 0.0);

        // Start iteration at second element as the first one is state
        for (int i = 1; i < keys.size(); i++) {
            gtsam::Key l = keys[i];
            POINT lmk = estimates.at<POINT>(l);
            gtsam::PoseToPointFactor<POSE, POINT> factor(x_key, l, meas, noise);
            gtsam::Vector error = factor.evaluateError(x_pose, lmk, Hx, Hl);
            hypothesis::Association a(meas_idx, l, Hx, Hl, error);
            double log_norm_factor;
            Eigen::Matrix2d S;
            double mh_dist = individual_compatability(a, x_key, joint_marginals, measurements, log_norm_factor, S);

            double mle_cost = mh_dist + log_norm_factor;

            // Individually compatible?
            if (mh_dist < mh_threshold_) {
                if (mle_cost < lowest_mle_cost) {
                    lowest_mle_cost = mle_cost;
                    smallest_innovation = {gtsam::symbolIndex(l), mle_cost};
                }
            }
        }
        if (smallest_innovation.first != -1) {
            lmk_meas_asso_candidates[L(smallest_innovation.first)].push_back({meas_idx, smallest_innovation.second});
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
        // Loop over all landmarks with measurement associated with it and pick put the best one by pruning out all
        // measurements except the best one in terms of Mahalanobis distance.
        for (const auto& [l, ms] : lmk_meas_asso_candidates) {
            auto p = std::min_element(ms.begin(), ms.end(),
                                      [](const auto& p1, const auto& p2) { return p1.second < p2.second; });
            // Pretty redundant to do full recomputation here, but oh well
            int meas_idx = p->first;
            POINT lmk = estimates.at<POINT>(l);

            const auto& meas = measurements[meas_idx].measurement;
            const auto& noise = measurements[meas_idx].noise;

            gtsam::PoseToPointFactor<POSE, POINT> factor(x_key, l, meas, noise);
            gtsam::Vector error = factor.evaluateError(x_pose, lmk, Hx, Hl);
            Association::shared_ptr a = std::make_shared<Association>(p->first, l, Hx, Hl, error);
            h.extend(a);
        }
    }

    // Regardless of if no or only some measurements were made, fill hypothesis with remaining unassociated measurements
    // and return
    h.fill_with_unassociated_measurements(num_measurements);

#ifdef HYPOTHESIS_QUALITY
    std::cout << "Computing joint NIS\n";
    double nis = joint_compatability<POSE::dimension, POINT::RowsAtCompileTime, POINT::RowsAtCompileTime>(
        h, x_key, marginals, measurements);
    h.set_nis(nis);
#endif

    return h;
}

}  // namespace ml
}  // namespace da
