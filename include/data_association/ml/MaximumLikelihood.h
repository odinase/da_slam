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
#include "slam/types.h"

#include <gtsam/base/FastVector.h>
#include <gtsam/geometry/Pose3.h>

#include "data_association/Hypothesis.h"
#include "data_association/DataAssociation.h"

namespace data_association
{
        namespace ml
        {

                using jcbb::Association;
                using jcbb::Hypothesis;

                template <class POSE, class POINT>
                class MaximumLikelihood : public DataAssociation
                {

                private:
                        const gtsam::Values &estimates_;
                        const gtsam::Marginals &marginals_;
                        const gtsam::FastVector<slam::Measurement<POINT>> &measurements_;
                        POSE x_pose_;
                        gtsam::Key x_key_;
                        gtsam::KeyList landmark_keys_;
                        double ic_prob_;
                        double range_threshold_;

                public:
                        MaximumLikelihood(const gtsam::Values &estimates, const gtsam::Marginals &marginals_, const gtsam::FastVector<slam::Measurement<POINT>> &measurements, double ic_prob, double range_threshold = 1e9);
                        virtual hypothesis::Hypothesis associate() const override;
                };

        } // namespace ml
} // namespace data_association

#include "ml/MaximumLikelihood.hxx"

#endif // MAXIMUM_LIKELIHOOD_H