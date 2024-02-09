#ifndef DA_SLAM_TYPES_HPP
#define DA_SLAM_TYPES_HPP

#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/sam/BearingRangeFactor.h>
#include <gtsam/slam/dataset.h>
#include <gtsam_unstable/slam/PoseToPointFactor.h>

#include <memory>

namespace da_slam::types
{

template <typename Pose>
struct Odometry
{
    Pose odom{};
    gtsam::SharedNoiseModel noise{};
};

template <typename Point>
struct Measurement
{
    Point measurement{};
    uint64_t idx = 0;  // For ground truth
    gtsam::SharedNoiseModel noise{};
};

using Measurement2D = Measurement<gtsam::Point2>;
using Measurement3D = Measurement<gtsam::Point3>;

template <typename Point>
using Measurements = gtsam::FastVector<Measurement<Point>>;

template <typename Pose, typename Point>
struct Timestep
{
    int step;
    Odometry<Pose> odom;
    Measurements<Point> measurements;
};

using Timestep2D = Timestep<gtsam::Pose2, gtsam::Point2>;
using Timestep3D = Timestep<gtsam::Pose3, gtsam::Point3>;

struct IndeterminantLinearSystemExceptionWithGraphValues : public gtsam::IndeterminantLinearSystemException
{
    gtsam::NonlinearFactorGraph graph{};
    gtsam::Values values{};
    std::string when{};
    IndeterminantLinearSystemExceptionWithGraphValues(const gtsam::IndeterminantLinearSystemException& err,
                                                      const gtsam::NonlinearFactorGraph& graph_input,
                                                      const gtsam::Values& values_input,
                                                      const char* when_input) noexcept
    : gtsam::IndeterminantLinearSystemException(err.nearbyVariable())
    , graph(graph_input)
    , values(values_input)
    , when(when_input)
    {
    }
};

}  // namespace da_slam::types

#endif  // DA_SLAM_TYPES_HPP