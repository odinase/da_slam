#ifndef DA_SLAM_DATA_ASSOCIATION_MAXIMUM_LIKELIHOOD_HPP
#define DA_SLAM_DATA_ASSOCIATION_MAXIMUM_LIKELIHOOD_HPP

#include <gtsam/base/FastVector.h>
#include <gtsam/geometry/Pose3.h>

#include <Eigen/Core>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

#include "da_slam/data_association/assignment_solvers/assignment_solver_interface.hpp"
#include "da_slam/data_association/data_association_interface.hpp"
#include "da_slam/data_association/hypothesis.hpp"
#include "da_slam/slam/utils_g2o.hpp"
#include "da_slam/types.hpp"

namespace da_slam::data_association::maximum_likelihood
{

struct MaximumLikelihoodParameters
{
    double sigmas{};
    double range_threshold{};
};

template <typename Pose, typename Point>
class MaximumLikelihood final : public IDataAssociation<types::Measurement<Point>>
{
   public:
    MaximumLikelihood(const MaximumLikelihoodParameters& params,
                      std::unique_ptr<assignment_solvers::IAssignmentSolver> assignment_solver)
    : m_assignment_solver(std::move(assignment_solver))
    , m_mh_threshold(params.sigmas * params.sigmas)
    , m_sigmas(params.sigmas)
    , m_range_threshold(params.range_threshold)
    {
    }

    hypothesis::Hypothesis associate(const gtsam::Values& estimates, const gtsam::Marginals& marginals,
                                     const gtsam::FastVector<types::Measurement<Point>>& measurements) const override;

   private:
    std::unique_ptr<assignment_solvers::IAssignmentSolver> m_assignment_solver{};
    double m_mh_threshold;
    double m_sigmas;
    double m_range_threshold;
};

using MaximumLikelihood2D = MaximumLikelihood<gtsam::Pose2, gtsam::Point2>;
using MaximumLikelihood3D = MaximumLikelihood<gtsam::Pose3, gtsam::Point3>;

}  // namespace da_slam::data_association::maximum_likelihood

#endif // DA_SLAM_DATA_ASSOCIATION_MAXIMUM_LIKELIHOOD_HPP