#ifndef MAXIMUM_LIKELIHOOD_H
#define MAXIMUM_LIKELIHOOD_H

#include <vector>
#include <queue>
#include <limits>
#include <eigen3/Eigen/Core>
#include <numeric>
#include <unordered_set>
#include <unordered_map>
#include <iostream>
#include <fstream>

#include "slam/types.h"

#include <gtsam/base/FastVector.h>
#include <gtsam/geometry/Pose3.h>

#include "data_association/Hypothesis.h"
#include "data_association/DataAssociation.h"

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
      double range_threshold_;
      std::ofstream asso_eff_file_;
      std::ofstream nis_logger_;

    public:
      // typedef std::shared_ptr<MaximumLikelihood> shared_ptr;
      MaximumLikelihood(double sigmas, double range_threshold = 1e9);
      virtual hypothesis::Hypothesis associate(
          const gtsam::Values &estimates,
          const gtsam::Marginals &marginals,
          const gtsam::FastVector<slam::Measurement<POINT>> &measurements) override;
    };

    using MaximumLikelihood2D = MaximumLikelihood<gtsam::Pose2, gtsam::Point2>;
    using MaximumLikelihood3D = MaximumLikelihood<gtsam::Pose3, gtsam::Point3>;

  } // namespace ml
} // namespace da

#include "data_association/ml/MaximumLikelihood.hxx"

#endif // MAXIMUM_LIKELIHOOD_H