#ifndef TYPES_H
#define TYPES_H

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/sam/BearingRangeFactor.h>
#include <gtsam/slam/dataset.h>
#include <gtsam_unstable/slam/PoseToPointFactor.h>
#include <gtsam/nonlinear/ISAM2.h>

#include <memory>


namespace slam {

template <class POSE>
struct Odometry {
    POSE odom;
    gtsam::SharedNoiseModel noise;
};

template <class POINT>
struct Measurement {
    POINT measurement;
    uint64_t idx = 0; // For ground truth
    gtsam::SharedNoiseModel noise;
};

using Measurement2D = Measurement<gtsam::Point2>;
using Measurement3D = Measurement<gtsam::Point3>;

template<class POINT>
using Measurements = gtsam::FastVector<Measurement<POINT>>;


template <class POSE, class POINT>
struct Timestep
{
    int step;
    Odometry<POSE> odom;
    Measurements<POINT> measurements;
};

using Timestep2D = Timestep<gtsam::Pose2, gtsam::Point2>;
using Timestep3D = Timestep<gtsam::Pose3, gtsam::Point3>;


struct IndeterminantLinearSystemExceptionWithISAM : public gtsam::IndeterminantLinearSystemException {
    std::unique_ptr<gtsam::ISAM2> isam;
    std::string when;
    IndeterminantLinearSystemExceptionWithISAM(const gtsam::IndeterminantLinearSystemException& err, std::unique_ptr<gtsam::ISAM2> isam_, const char* when_) noexcept :
    gtsam::IndeterminantLinearSystemException(err.nearbyVariable()),
     isam(std::move(isam_)),
     when(when_)
      {}
};


struct IndeterminantLinearSystemExceptionWithGraphValues : public gtsam::IndeterminantLinearSystemException {
    gtsam::NonlinearFactorGraph graph;
    gtsam::Values values;
    std::string when;
    IndeterminantLinearSystemExceptionWithGraphValues(const gtsam::IndeterminantLinearSystemException& err, const gtsam::NonlinearFactorGraph& graph_,
    const gtsam::Values& values_, const char* when_) noexcept :
    gtsam::IndeterminantLinearSystemException(err.nearbyVariable()),
    graph(graph_),
    values(values_),
     when(when_)
      {}
};


} // namespace slam

#endif // TYPES_H