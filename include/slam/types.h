#ifndef TYPES_H
#define TYPES_H

#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/sam/BearingRangeFactor.h>
#include <gtsam/slam/dataset.h>
#include <gtsam_unstable/slam/PoseToPointFactor.h>
namespace slam {

template <class POSE>
struct Odometry {
    POSE odom;
    gtsam::SharedNoiseModel noise;
};

template <class POINT>
struct Measurement {
    POINT measurement;
    gtsam::SharedNoiseModel noise;
};

using Measurement2D = Measurement<gtsam::Point2>;
using Measurement3D = Measurement<gtsam::Point3>;


template <class POSE, class POINT>
struct Timestep
{
    int step;
    Odometry<POSE> odom;
    gtsam::FastVector<Measurement<POINT>> measurements;
};

using Timestep2D = Timestep<gtsam::Pose2, gtsam::Point2>;
using Timestep3D = Timestep<gtsam::Pose3, gtsam::Point3>;


struct Landmark {
    unsigned long int id_gt;
    gtsam::Key key;
    gtsam::Pose3 pose;
};

} // namespace slam

#endif // TYPES_H