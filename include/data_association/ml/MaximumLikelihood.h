#ifndef MAXIMUM_LIKELIHOOD_H
#define MAXIMUM_LIKELIHOOD_H

#include <gtsam/base/FastVector.h>
#include <gtsam/geometry/Pose3.h>

#include <Eigen/Core>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

#include "data_association/DataAssociation.h"
#include "data_association/Hypothesis.h"
#include "slam/types.h"

namespace da
{
namespace ml
{
using hypothesis::Association;
using hypothesis::Hypothesis;

template <class POSE, class POINT>
class MaximumLikelihood : public DataAssociation<slam::Measurement<POINT>>
{
   private:
    double mh_threshold_;
    double sigmas_;
    double range_threshold_;

   public:
    MaximumLikelihood(double sigmas, double range_threshold = std::numeric_limits<double>::infinity());
    virtual hypothesis::Hypothesis associate(const gtsam::Values& estimates, const gtsam::Marginals& marginals,
                                             const gtsam::FastVector<slam::Measurement<POINT>>& measurements) override;

    hypothesis::Hypothesis associate_bad(const gtsam::Values& estimates, const gtsam::Marginals& marginals,
                                         const gtsam::FastVector<slam::Measurement<POINT>>& measurements);
};

using MaximumLikelihood2D = MaximumLikelihood<gtsam::Pose2, gtsam::Point2>;
using MaximumLikelihood3D = MaximumLikelihood<gtsam::Pose3, gtsam::Point3>;

}  // namespace ml
}  // namespace da

#include "data_association/ml/MaximumLikelihood.hxx"

#endif  // MAXIMUM_LIKELIHOOD_H