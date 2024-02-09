#ifndef KNOWN_DATA_ASSOCIATION_H
#define KNOWN_DATA_ASSOCIATION_H

#include <gtsam/base/FastVector.h>
#include <gtsam/geometry/Pose3.h>

#include <Eigen/Core>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <vector>

#include "data_association/DataAssociation.h"
#include "data_association/Hypothesis.h"
#include "slam/types.h"

namespace da
{
namespace gt
{
using hypothesis::Association;
using hypothesis::Hypothesis;

template <class POSE, class POINT>
class KnownDataAssociation : public DataAssociation<slam::Measurement<POINT>>
{
   private:
    std::map<uint64_t, gtsam::Key> meas_lmk_assos_;
    std::map<gtsam::Key, gtsam::Key> gt_lmk2map_lmk_;
    uint64_t curr_landmark_count_;

   public:
    KnownDataAssociation(const std::map<uint64_t, gtsam::Key>& meas_lmk_assos);
    virtual hypothesis::Hypothesis associate(const gtsam::Values& estimates, const gtsam::Marginals& marginals,
                                             const gtsam::FastVector<slam::Measurement<POINT>>& measurements) override;
};

using KnownDataAssociation2D = KnownDataAssociation<gtsam::Pose2, gtsam::Point2>;
using KnownDataAssociation3D = KnownDataAssociation<gtsam::Pose3, gtsam::Point3>;

}  // namespace gt
}  // namespace da

#include "data_association/gt/KnownDataAssociation.hxx"

#endif  // KNOWN_DATA_ASSOCIATION_H