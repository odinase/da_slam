// NOLINTBEGIN
#ifndef DA_SLAM_SLAM_UTILS_G2O_HPP
#define DA_SLAM_SLAM_UTILS_G2O_HPP
/* ----------------------------------------------------------------------------

 * GTSAM Copyright 2010, Georgia Tech Research Corporation,
 * Atlanta, Georgia 30332-0415
 * All Rights Reserved
 * Authors: Frank Dellaert, et al. (see THANKS for the full author list)

 * See LICENSE for the license information

 * -------------------------------------------------------------------------- */

/**
 * @file landmarkSLAM_g2o.cpp
 * @brief A 2D/3D landmark SLAM pipeline that reads input from g2o, converts it
 * to a factor graph and does the optimization. Output is written on a file, in
 * g2o format Syntax for the script is ./landmarkSLAM_g2o input.g2o is3D
 * output.g2o
 * @date December 15, 2021
 * @author Luca Carlone, Yihao Zhang
 */

#include <gtsam/base/Value.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/sam/BearingRangeFactor.h>
#include <gtsam/slam/dataset.h>
#include <gtsam_unstable/slam/PoseToPointFactor.h>

#include <boost/filesystem/path.hpp>
#include <boost/pointer_cast.hpp>
#include <boost/shared_ptr.hpp>
#include <cmath>
#include <ctime>
#include <fstream>
#include <range/v3/all.hpp>

#include "da_slam/types.hpp"

namespace gtsam
{

// from gtsam4.0.3/slam/dataset.cpp
Rot3 NormalizedRot3(double w, double x, double y, double z);

void saveException(const std::string& g2ofilepath, const std::string& algName, const char* err_msg,
                   const std::string& other_msg);
// save graph errors in the same directory as the g2o file
void saveGraphErrors(const std::string& g2ofilepath, const std::string& algName, const std::vector<double>& errors);

void saveVector(const std::string& g2ofilepath, const std::string& saveName, const std::vector<double>& errors);

std::vector<size_t> findPoseToPointFactors(std::vector<std::shared_ptr<PoseToPointFactor<Pose2, Point2>>>& lmFactors2d,
                                           std::vector<std::shared_ptr<PoseToPointFactor<Pose3, Point3>>>& lmFactors3d,
                                           const NonlinearFactorGraph::shared_ptr& graph);

std::pair<std::vector<size_t>, std::vector<size_t>> findFactors(
    std::vector<std::shared_ptr<BetweenFactor<Pose2>>>& odomFactors2d,
    std::vector<std::shared_ptr<BetweenFactor<Pose3>>>& odomFactors3d,
    std::vector<std::shared_ptr<PoseToPointFactor<Pose2, Point2>>>& measFactors2d,
    std::vector<std::shared_ptr<PoseToPointFactor<Pose3, Point3>>>& measFactors3d,
    const NonlinearFactorGraph::shared_ptr& graph);

size_t num_poses(const Values& vals);

KeyVector findLmKeys(const Values& initial);

KeyVector findLmKeys(const Values::shared_ptr& initial);

KeyVector findPoseKeys(const Values::shared_ptr& initial, const KeyVector& ldmk_keys);

// may read landmark measurements if tagged by 'BR' or 'LANDMARK' (see dataset.cpp)
// TODO(LC): what about potential loop closures?
std::pair<NonlinearFactorGraph::shared_ptr, Values::shared_ptr> readG2oOdomOnly(const std::string& g2oFile,
                                                                                const bool is3D,
                                                                                const std::string& kernelType);

std::pair<NonlinearFactorGraph::shared_ptr, Values::shared_ptr> readG2owithLmks(const std::string& g2oFile,
                                                                                const bool is3D,
                                                                                const std::string& kernelType);

template <class POINT>
bool isPutativeAssociation(const Key key1, const Key key2, const Vector diff, const gtsam::Marginals& marginals)
{
    // Current joint marginals of the 2 landmarks
    gtsam::KeyVector keys;
    keys.push_back(key1);
    keys.push_back(key2);

    // Compute covariance of difference
    Matrix covJoint = marginals.jointMarginalCovariance(keys).fullMatrix();
    size_t varSize = covJoint.rows() / 2;
    Matrix diffMat =
        (Matrix(varSize, 2 * varSize) << Matrix::Identity(varSize, varSize), -Matrix::Identity(varSize, varSize))
            .finished();
    Matrix covDiff = diffMat * covJoint * diffMat.transpose();
    Matrix mahDist = diff.transpose() * covDiff.inverse() * diff;
    return mahDist(0, 0) < 9.0;  // ~chi2inv(0.99,2)
}

void writeG2oLdmkEdges(const NonlinearFactorGraph& graph, const Values& estimate, const std::string& filename,
                       const std::string& g2ofilepath);

}  // namespace gtsam

template <typename Pose, typename Point>
std::vector<da_slam::types::Timestep<Pose, Point>> convert_into_timesteps(
    std::vector<std::shared_ptr<gtsam::BetweenFactor<Pose>>>& odomFactors,
    std::vector<std::shared_ptr<gtsam::PoseToPointFactor<Pose, Point>>>& measFactors)
{
    // Sort factors based on robot pose key, so that we can simply check when in time they should appear
    std::sort(odomFactors.begin(), odomFactors.end(), [](const auto& lhs, const auto& rhs) {
        return gtsam::symbolIndex(lhs->key1()) < gtsam::symbolIndex(rhs->key1());
    });
    std::sort(measFactors.begin(), measFactors.end(), [](const auto& lhs, const auto& rhs) {
        return gtsam::symbolIndex(lhs->key1()) < gtsam::symbolIndex(rhs->key1());
    });

    size_t odoms = odomFactors.size();
    uint64_t num_timesteps =
        odoms + 1;  // There will always be one more robot pose than odometry factors since they're all between
    std::vector<da_slam::types::Timestep<Pose, Point>> timesteps{};
    uint64_t curr_measurement = 0;
    size_t tot_num_measurements = measFactors.size();
    timesteps.reserve(num_timesteps);
    for (uint64_t t = 0; t < num_timesteps; t++) {
        da_slam::types::Timestep<Pose, Point> timestep{};
        timestep.step = t;
        // Initialize first odom as identity, as we haven't moved yet
        if (t > 0) {
            timestep.odom.odom = odomFactors[t - 1]->measured();
            timestep.odom.noise = odomFactors[t - 1]->noiseModel();
        }
        else {
            timestep.odom.odom = Pose{};
        }

        // Extract measurements from current pose
        while (curr_measurement < tot_num_measurements &&
               gtsam::symbolIndex(measFactors[curr_measurement]->key1()) == t) {
            da_slam::types::Measurement<Point> meas{};
            meas.idx = curr_measurement;
            meas.measurement = measFactors[curr_measurement]->measured();
            meas.noise = measFactors[curr_measurement]->noiseModel();
            timestep.measurements.push_back(meas);
            curr_measurement++;
        }

        timesteps.push_back(timestep);
    }

    return timesteps;
}

template <typename Pose, typename Point>
std::map<uint64_t, gtsam::Key> measurement_landmarks_associations(
    const std::vector<std::shared_ptr<gtsam::PoseToPointFactor<Pose, Point>>>& measFactors,
    const std::vector<da_slam::types::Timestep<Pose, Point>>& timesteps)
{
    std::map<uint64_t, gtsam::Key> meas_lmk_assos;
    for (const auto& timestep : timesteps) {
        for (const auto measurement : timestep.measurements) {
            meas_lmk_assos[measurement.idx] = measFactors[measurement.idx]->key2();
        }
    }

    return meas_lmk_assos;
}

void writeG2o_with_landmarks(const gtsam::NonlinearFactorGraph& graph, const gtsam::Values& estimates,
                             const std::string& filename);

#endif  // DA_SLAM_SLAM_UTILS_G2O_HPP
        // NOLINTEND