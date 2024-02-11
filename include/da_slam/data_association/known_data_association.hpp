#ifndef KNOWN_DATA_ASSOCIATION_H
#define KNOWN_DATA_ASSOCIATION_H

#include <gtsam/base/FastVector.h>
#include <gtsam/geometry/Pose3.h>
#include <spdlog/spdlog.h>

#include <Eigen/Core>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <range/v3/all.hpp>
#include <vector>

#include "da_slam/data_association/data_association_interface.hpp"
#include "da_slam/data_association/hypothesis.hpp"
#include "da_slam/types.hpp"
#include "da_slam/utils.hpp"

namespace da_slam::data_association
{
using hypothesis::Association;
using hypothesis::Hypothesis;

template <typename Pose, typename Point>
class KnownDataAssociation : public IDataAssociation<types::Measurement<Point>>
{
   public:
    explicit KnownDataAssociation(const std::map<uint64_t, gtsam::Key>& meas_lmk_assos)
    : m_meas_lmk_assos{meas_lmk_assos}, m_curr_landmark_count{0}
    {
    }

    hypothesis::Hypothesis associate(const gtsam::Values& estimates, const gtsam::Marginals& marginals,
                                     const gtsam::FastVector<types::Measurement<Point>>& measurements) override
    {
        const auto num_poses = estimates.extract<Pose>(gtsam::Symbol::ChrTest('x')).size();
        const auto last_pose = num_poses - 1;  // Assuming first pose is 0
        const auto x_key = utils::pose_key(last_pose);
        const auto x_pose = estimates.at<Pose>(x_key);

        // Make hypothesis to return later
        hypothesis::Hypothesis h = hypothesis::Hypothesis::empty_hypothesis();

        /*
         * Look through all measurements, and find landmark they should be associated with.
         * If this landmark key does not exist in the mapping between ground truth map and our map, leave measurement
         * unassociated. Otherwise, associate.
         */

        gtsam::Matrix Hx, Hl;

        namespace rngv = ranges::views;
        for (auto&& [meas_idx, measurement] : measurements | rngv::enumerate) {
            const auto lmk_gt = m_meas_lmk_assos[measurement.idx];
            const auto lmk_mapping_it = m_gt_lmk2map_lmk.find(lmk_gt);

            // If we find the mapping, associate to it
            if (lmk_mapping_it != m_gt_lmk2map_lmk.end()) {
                spdlog::info("Found ground truth landmark {}", gtsam::Symbol(lmk_gt));
                const auto l = lmk_mapping_it->second;
                const auto lmk = estimates.at<Point>(l);
                const auto& meas = measurement.measurement;
                const auto& noise = measurement.noise;

                const gtsam::PoseToPointFactor<Pose, Point> factor(x_key, l, meas, noise);
                const auto error = factor.evaluateError(x_pose, lmk, Hx, Hl);
                Association::shared_ptr a = std::make_shared<Association>(meas_idx, l, Hx, Hl, error);
                h.extend(a);
            }
            // We have not seen this landmark before - add to mapping
            else {
                // TODO: Not sure if keeping track of landmark count internally is a good way of doing this, but ohwell
                m_gt_lmk2map_lmk[lmk_gt] = utils::lmk_key(m_curr_landmark_count);
                m_curr_landmark_count++;
                Association::shared_ptr a = std::make_shared<Association>(meas_idx);
                h.extend(a);
            }
        }

        // #ifdef LOGGING
        //         std::cout << "\n\n";
        //         std::cout << "KnownDataAssociations made associations:\n";
        //         for (const auto& asso : h.associations()) {
        //             if (asso->associated()) {
        //                 std::cout << "Measurement z" << measurements[asso->measurement].idx << " associated with
        //                 landmark "
        //                           << gtsam::Symbol(*asso->landmark) << "\n";
        //             }
        //             else {
        //                 std::cout << "Measurement z" << measurements[asso->measurement].idx
        //                           << " unassociated, initialized landmark "
        //                           << gtsam::Symbol(
        //                                  m_gt_lmk2map_lmk.find(meas_lmk_assos_[measurements[asso->measurement].idx])->second)
        //                           << "\n";
        //             }
        //         }
        //         std::cout << "\n";
        // #endif  // LOGGING

        return h;
    }

   private:
    std::map<uint64_t, gtsam::Key> m_meas_lmk_assos{};
    std::map<gtsam::Key, gtsam::Key> m_gt_lmk2map_lmk{};
    uint64_t m_curr_landmark_count{};
};

using KnownDataAssociation2D = KnownDataAssociation<gtsam::Pose2, gtsam::Point2>;
using KnownDataAssociation3D = KnownDataAssociation<gtsam::Pose3, gtsam::Point3>;

}  // namespace da_slam::data_association

#endif  // KNOWN_DATA_ASSOCIATION_H