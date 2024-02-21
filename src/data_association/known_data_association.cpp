#include "da_slam/data_association/known_data_association.hpp"

#include <gtsam/base/FastMap.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam_unstable/slam/PoseToPointFactor.h>

#include <algorithm>
#include <chrono>
#include <da_slam/types.hpp>
#include <iostream>
#include <limits>
#include <memory>
#include <range/v3/all.hpp>
#include <utility>
#include "da_slam/fmt.hpp"

using gtsam::symbol_shorthand::L;
using gtsam::symbol_shorthand::X;

namespace rngv = ranges::views;

namespace da_slam::data_association::ground_truth
{

template <typename Pose, typename Point>
hypothesis::Hypothesis KnownDataAssociation<Pose, Point>::associate(
    const gtsam::Values& estimates, const gtsam::Marginals& /*marginals*/,
    const gtsam::FastVector<types::Measurement<Point>>& measurements) const
{
    // gtsam::KeyList landmark_keys = estimates.extract(gtsam::Symbol::ChrTest('l')).keys();
    const auto pose_num = estimates.template extract<Pose>(gtsam::Symbol::ChrTest('x')).size();
    const auto last_pose = pose_num - 1;  // Assuming first pose is 0
    const auto x_key = X(last_pose);
    const auto x_pose = estimates.template at<Pose>(x_key);
    // size_t num_measurements = measurements.size();
    // size_t num_landmarks = landmark_keys.size();

    // Make hypothesis to return later
    hypothesis::Hypothesis h = hypothesis::Hypothesis::empty_hypothesis();

    /*
     * Look through all measurements, and find landmark they should be associated with.
     * If this landmark key does not exist in the mapping between ground truth map and our map, leave measurement
     * unassociated. Otherwise, associate.
     */

    gtsam::Matrix Hx, Hl;

    for (auto&& [meas_idx, measurement] : measurements | rngv::enumerate) {
        gtsam::Key lmk_gt = m_meas_lmk_assos.at(measurement.idx);
        const auto lmk_mapping_it = m_gt_lmk2map_lmk.find(lmk_gt);

        hypothesis::Association::shared_ptr a{};

        // If we find the mapping, associate to it
        if (lmk_mapping_it != m_gt_lmk2map_lmk.end()) {
            spdlog::debug("Found ground truth {}", gtsam::Symbol(lmk_gt));
            const auto l = lmk_mapping_it->second;
            const auto lmk = estimates.template at<Point>(l);
            const auto& meas = measurement.measurement;
            const auto& noise = measurement.noise;

            const gtsam::PoseToPointFactor<Pose, Point> factor{x_key, l, meas, noise};
            const auto error = factor.evaluateError(x_pose, lmk, Hx, Hl);
            a = std::make_shared<hypothesis::Association>(meas_idx, l, Hx, Hl, error);
        }
        // We have not seen this landmark before - add to mapping
        else {
            // TODO: Not sure if keeping track of landmark count internally is a good way of doing this, but ohwell
            m_gt_lmk2map_lmk[lmk_gt] = L(m_curr_landmark_count);
            m_curr_landmark_count++;
            a = std::make_shared<hypothesis::Association>(meas_idx);
        }
        h.extend(a);
    }

#ifdef LOGGING
    std::cout << "\n\n";
    std::cout << "KnownDataAssociations made associations:\n";
    for (const auto& asso : h.associations()) {
        if (asso->associated()) {
            std::cout << "Measurement z" << measurements[asso->measurement].idx << " associated with landmark "
                      << gtsam::Symbol(*asso->landmark) << "\n";
        }
        else {
            std::cout << "Measurement z" << measurements[asso->measurement].idx
                      << " unassociated, initialized landmark "
                      << gtsam::Symbol(
                             m_gt_lmk2map_lmk.find(m_meas_lmk_assos[measurements[asso->measurement].idx])->second)
                      << "\n";
        }
    }
    std::cout << "\n";
#endif  // LOGGING

    return h;
}

template class KnownDataAssociation<gtsam::Pose2, gtsam::Point2>;
template class KnownDataAssociation<gtsam::Pose3, gtsam::Point3>;

}  // namespace da_slam::data_association::ground_truth