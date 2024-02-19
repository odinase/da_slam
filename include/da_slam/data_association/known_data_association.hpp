#ifndef DA_SLAM_DATA_ASSOCIATION_KNOWN_DATA_ASSOCIATION_HPP
#define DA_SLAM_DATA_ASSOCIATION_KNOWN_DATA_ASSOCIATION_HPP

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

namespace da_slam::data_association::ground_truth
{

template <typename Pose, typename Point>
class KnownDataAssociation : public IDataAssociation<types::Measurement<Point>>
{
   public:
    explicit KnownDataAssociation(const std::map<uint64_t, gtsam::Key>& meas_lmk_assos)
    : m_meas_lmk_assos{meas_lmk_assos}, m_curr_landmark_count{0}
    {
    }

    hypothesis::Hypothesis associate(const gtsam::Values& estimates, const gtsam::Marginals& marginals,
                                     const gtsam::FastVector<types::Measurement<Point>>& measurements) const override;

   private:
    std::map<uint64_t, gtsam::Key> m_meas_lmk_assos{};
    mutable std::map<gtsam::Key, gtsam::Key> m_gt_lmk2map_lmk{};
    mutable uint64_t m_curr_landmark_count{};
};

using KnownDataAssociation2D = KnownDataAssociation<gtsam::Pose2, gtsam::Point2>;
using KnownDataAssociation3D = KnownDataAssociation<gtsam::Pose3, gtsam::Point3>;

}  // namespace da_slam::data_association::ground_truth

#endif  // DA_SLAM_DATA_ASSOCIATION_KNOWN_DATA_ASSOCIATION_HPP