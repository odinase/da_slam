// NOLINTBEGIN
#pragma once
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

using gtsam::symbol_shorthand::L;  // gtsam/slam/dataset.cpp

namespace gtsam
{

#define LINESIZE 81920  // dataset.cpp. ignore lines after this

// from gtsam4.0.3/slam/dataset.cpp
static Rot3 NormalizedRot3(double w, double x, double y, double z)
{
    const double norm = sqrt(w * w + x * x + y * y + z * z), f = 1.0 / norm;
    return Rot3::Quaternion(f * w, f * x, f * y, f * z);
}

void saveException(const std::string& g2ofilepath, const std::string& algName, const char* err_msg,
                   const std::string& other_msg)
{
    auto g2opath = boost::filesystem::path(g2ofilepath);
    auto fname = boost::filesystem::path(algName);
    auto savepath = g2opath.parent_path() / fname;
    std::fstream stream(savepath.c_str(), std::fstream::app);
    stream << "\nException: ";
    stream << err_msg << std::endl;
    stream << other_msg << std::endl;
    stream << g2opath.filename() << std::endl;
    stream.close();
}

// save graph errors in the same directory as the g2o file
void saveGraphErrors(const std::string& g2ofilepath, const std::string& algName, const std::vector<double>& errors)
{
    auto g2opath = boost::filesystem::path(g2ofilepath);
    auto fname = boost::filesystem::path(std::string("errors_graph_") + algName + std::string(".txt"));
    auto errorSavePath = g2opath.parent_path() / fname;  // concat
    auto g2ofilename = g2opath.filename();
    // size_t lastindex = g2ofilename.find_last_of(".");
    // string nameonly = g2ofilename.substr(0, lastindex); // remove extension
    // std::cout << "Writing graph errors to " << errorSavePath << "\n";
    std::fstream stream(errorSavePath.c_str(), std::fstream::app);  // append mode ::out overwrites
    std::time_t now = std::time(0);
    char* dt = ctime(&now);
    stream << dt << g2ofilename.c_str() << std::endl;
    for (int i = 0; i < errors.size(); i++) {
        stream << errors[i];
        if (i != errors.size() - 1) {
            stream << ", ";
        }
    }
    stream << std::endl;
    stream.close();
}

void saveVector(const std::string& g2ofilepath, const std::string& saveName, const std::vector<double>& errors)
{
    auto g2opath = boost::filesystem::path(g2ofilepath);
    auto fname = boost::filesystem::path(saveName);
    auto savepath = g2opath.parent_path() / fname;
    // std::cout << "Writing vector to " << savepath << "\n";
    std::fstream stream(savepath.c_str(), std::fstream::out);  // write mode overwrites
    for (int i = 0; i < errors.size(); i++) {
        stream << errors[i];
        if (i != errors.size() - 1) {
            stream << ", ";
        }
    }
    stream.close();
}

std::vector<size_t> findPoseToPointFactors(std::vector<std::shared_ptr<PoseToPointFactor<Pose2, Point2>>>& lmFactors2d,
                                           std::vector<std::shared_ptr<PoseToPointFactor<Pose3, Point3>>>& lmFactors3d,
                                           const NonlinearFactorGraph::shared_ptr& graph)
{
    assert(lmFactors2d.size() == 0 && lmFactors3d.size() == 0);
    std::shared_ptr<PoseToPointFactor<Pose2, Point2>> factor2d;
    std::shared_ptr<PoseToPointFactor<Pose3, Point3>> factor3d;
    size_t factor_idx = 0;
    std::vector<size_t> lmFactorIdx;
    for (const auto& factor_ : *graph) {  // inspired by dataset.cpp/writeG2o
        factor3d = std::dynamic_pointer_cast<PoseToPointFactor<Pose3, Point3>>(factor_);
        factor2d = std::dynamic_pointer_cast<PoseToPointFactor<Pose2, Point2>>(factor_);
        if (factor2d || factor3d) {
            lmFactorIdx.push_back(factor_idx);
        }
        if (factor2d) {
            lmFactors2d.push_back(factor2d);
        }
        if (factor3d) {
            lmFactors3d.push_back(factor3d);
        }
        factor_idx++;
    }
    return lmFactorIdx;
}

std::pair<std::vector<size_t>, std::vector<size_t>> findFactors(
    std::vector<std::shared_ptr<BetweenFactor<Pose2>>>& odomFactors2d,
    std::vector<std::shared_ptr<BetweenFactor<Pose3>>>& odomFactors3d,
    std::vector<std::shared_ptr<PoseToPointFactor<Pose2, Point2>>>& measFactors2d,
    std::vector<std::shared_ptr<PoseToPointFactor<Pose3, Point3>>>& measFactors3d,
    const NonlinearFactorGraph::shared_ptr& graph)
{
    assert(odomFactors2d.size() == 0 && odomFactors3d.size() == 0);
    std::shared_ptr<PoseToPointFactor<Pose2, Point2>> measFactor2d;
    std::shared_ptr<PoseToPointFactor<Pose3, Point3>> measFactor3d;
    std::shared_ptr<BetweenFactor<Pose2>> odomFactor2d;
    std::shared_ptr<BetweenFactor<Pose3>> odomFactor3d;
    size_t factor_idx = 0;
    std::vector<size_t> measFactorIdx;
    std::vector<size_t> odomFactorIdx;
    for (const auto& factor_ : *graph) {  // inspired by dataset.cpp/writeG2o
        measFactor2d = std::dynamic_pointer_cast<PoseToPointFactor<Pose2, Point2>>(factor_);
        measFactor3d = std::dynamic_pointer_cast<PoseToPointFactor<Pose3, Point3>>(factor_);
        odomFactor2d = std::dynamic_pointer_cast<BetweenFactor<Pose2>>(factor_);
        odomFactor3d = std::dynamic_pointer_cast<BetweenFactor<Pose3>>(factor_);
        if (measFactor2d || measFactor3d) {
            measFactorIdx.push_back(factor_idx);
            if (measFactor2d) {
                measFactors2d.push_back(measFactor2d);
            }
            if (measFactor3d) {
                measFactors3d.push_back(measFactor3d);
            }
        }
        if (odomFactor2d || odomFactor3d) {
            odomFactorIdx.push_back(factor_idx);
            if (odomFactor2d) {
                odomFactors2d.push_back(odomFactor2d);
            }
            if (odomFactor3d) {
                odomFactors3d.push_back(odomFactor3d);
            }
        }

        factor_idx++;
    }
    return std::make_pair(odomFactorIdx, measFactorIdx);
}

size_t num_poses(const Values& vals)
{
    return static_cast<size_t>(ranges::count_if(vals.keySet(), [](auto&& k) { return gtsam::symbolChr(k) == 'x'; }));
}

KeyVector findLmKeys(const Values& initial)
{
    return initial.keys()                                                                     //
           | ranges::actions::remove_if([](auto&& k) { return gtsam::symbolChr(k) != 'l'; })  //
           | ranges::actions::unique;
}

KeyVector findLmKeys(const Values::shared_ptr& initial)
{
    return findLmKeys(*initial);
}

KeyVector findPoseKeys(const Values::shared_ptr& initial, const KeyVector& ldmk_keys)
{
    KeyVector pose_keys;
    for (const auto key_value : *initial) {
        if (std::find(ldmk_keys.begin(), ldmk_keys.end(), key_value.key) == ldmk_keys.end()) {
            pose_keys.push_back(key_value.key);
        }
    }
    return pose_keys;
}

// may read landmark measurements if tagged by 'BR' or 'LANDMARK' (see dataset.cpp)
// TODO(LC): what about potential loop closures?
std::pair<NonlinearFactorGraph::shared_ptr, Values::shared_ptr> readG2oOdomOnly(const std::string& g2oFile,
                                                                                const bool is3D,
                                                                                const std::string& kernelType)
{
    NonlinearFactorGraph::shared_ptr odom_graph;
    Values::shared_ptr initial;

    if (kernelType.compare("none") == 0) {
        std::tie(odom_graph, initial) = readG2o(g2oFile, is3D);  // load 2d: 2d pose, 2d ldmk point2
    }
    if (kernelType.compare("huber") == 0) {
        std::cout << "Using robust kernel: huber " << std::endl;
        std::tie(odom_graph, initial) = readG2o(g2oFile, is3D, KernelFunctionTypeHUBER);
    }
    if (kernelType.compare("tukey") == 0) {
        std::cout << "Using robust kernel: tukey " << std::endl;
        std::tie(odom_graph, initial) = readG2o(g2oFile, is3D, KernelFunctionTypeTUKEY);
    }

    // Add prior on the pose having index (key) = 0
    if (is3D) {
        auto priorModel = noiseModel::Diagonal::Variances((Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
        odom_graph->addPrior(0, initial->at<Pose3>(0), priorModel);
    }
    else {
        auto priorModel =  //
            noiseModel::Diagonal::Variances(Vector3(1e-6, 1e-6, 1e-8));
        odom_graph->addPrior(0, initial->at<Pose2>(0), priorModel);
    }
    std::cout << "Adding prior on pose 0." << std::endl;

    return std::make_pair(odom_graph, initial);
}

std::pair<NonlinearFactorGraph::shared_ptr, Values::shared_ptr> readG2owithLmks(const std::string& g2oFile,
                                                                                const bool is3D,
                                                                                const std::string& kernelType)
{
    NonlinearFactorGraph::shared_ptr graph;
    Values::shared_ptr initial;

    std::tie(graph, initial) = readG2oOdomOnly(g2oFile, is3D, kernelType);

    // reading file for landmark estimates and landmark edges (gtsam4.0.3
    // slam/dataset.cpp)
    std::ifstream is(g2oFile.c_str());
    if (!is) throw std::invalid_argument("cannot find file " + g2oFile);
    std::string tag;

    Key id1, id2;
    while (!is.eof()) {
        if (!(is >> tag)) break;

        // add 2d ldmk pose2point factor (if any)
        if (tag == "EDGE_SE2_XY") {
            double lmx, lmy;
            double v1, v2, v3;

            is >> id1 >> id2 >> lmx >> lmy >> v1 >> v2 >> v3;

            // Convert to cov (assuming diagonal mat)
            // v1 = 1.0 / v1;
            // v3 = 1.0 / v3;
            // Create noise model
            // noiseModel::Diagonal::shared_ptr measurementNoise =
            //     noiseModel::Diagonal::Variances((Vector(2) << v1, v3).finished());

            // Create noise model
            Matrix2 info_mat;
            info_mat << v1, v2, v2, v3;
            noiseModel::Gaussian::shared_ptr measurementNoise =
                noiseModel::Gaussian::Information(info_mat, true);  // smart = true

            // Add to graph
            *graph += gtsam::PoseToPointFactor<Pose2, Point2>(id1, L(id2), Point2(lmx, lmy), measurementNoise);
        }

        // add 3d ldmk pose2point factor (if any)
        if (tag == "EDGE_SE3_XYZ") {
            double lmx, lmy, lmz;
            double v11, v12, v13, v22, v23, v33;

            is >> id1 >> id2 >> lmx >> lmy >> lmz >> v11 >> v12 >> v13 >> v22 >> v23 >> v33;

            // Convert to cov (assuming diagonal mat)
            // v11 = 1.0 / v11;
            // v22 = 1.0 / v22;
            // v33 = 1.0 / v33;
            // Create noise model
            // noiseModel::Diagonal::shared_ptr measurementNoise =
            //     noiseModel::Diagonal::Variances(
            //         (Vector(3) << v11, v22, v33).finished());

            // Create noise model
            Matrix3 info_mat;
            info_mat << v11, v12, v13, v12, v22, v23, v13, v23, v33;
            noiseModel::Gaussian::shared_ptr measurementNoise = noiseModel::Gaussian::Information(info_mat, true);

            // Add to graph
            *graph += gtsam::PoseToPointFactor<Pose3, Point3>(id1, L(id2), Point3(lmx, lmy, lmz), measurementNoise);
        }
        is.ignore(LINESIZE, '\n');
    }
    is.clear();
    is.seekg(0, std::ios::beg);  // guess back to beginning

    return std::make_pair(graph, initial);
}

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
                       const std::string& g2ofilepath)
{
    auto g2opath = boost::filesystem::path(g2ofilepath);
    auto fname = boost::filesystem::path(filename);
    auto savepath = g2opath.parent_path() / fname;
    gtsam::writeG2o(graph, estimate, savepath.c_str());
    std::fstream stream(savepath.c_str(), std::fstream::app);
    auto index = [](gtsam::Key key) { return Symbol(key).index(); };
    for (const auto& factor_ : graph) {
        // 2D factor
        auto lmfactor2d = std::dynamic_pointer_cast<PoseToPointFactor<Pose2, Point2>>(factor_);
        if (lmfactor2d) {
            SharedNoiseModel model = lmfactor2d->noiseModel();
            auto gaussModel = std::dynamic_pointer_cast<noiseModel::Gaussian>(model);
            if (!gaussModel) {
                model->print("model\n");
                throw std::invalid_argument("writeG2oLdmkEdges: invalid noise model!");
                // std::cout << "Landmark edges not written and skipped" << std::endl;
                // break;
            }
            else {
                Matrix2 Info = gaussModel->R().transpose() * gaussModel->R();  // or just information()
                Point2 measured2d = lmfactor2d->measured();
                stream << "EDGE_SE2_XY " << index(lmfactor2d->key1()) << " " << index(lmfactor2d->key2()) << " "
                       << std::fixed << std::setprecision(6) << measured2d.x() << " " << measured2d.y();
                for (size_t i = 0; i < 2; i++) {
                    for (size_t j = i; j < 2; j++) {
                        stream << " " << std::fixed << std::setprecision(6) << Info(i, j);
                    }
                }
                stream << std::endl;
            }
        }
        // 3D factor
        auto lmfactor3d = std::dynamic_pointer_cast<PoseToPointFactor<Pose3, Point3>>(factor_);
        if (lmfactor3d) {
            SharedNoiseModel model = lmfactor3d->noiseModel();
            auto gaussModel = std::dynamic_pointer_cast<noiseModel::Gaussian>(model);
            if (!gaussModel) {
                model->print("model\n");
                throw std::invalid_argument("writeG2oLdmkEdges: invalid noise model!");
                // std::cout << "Landmark edges not written and skipped" << std::endl;
                // break;
            }
            else {
                Matrix3 Info = gaussModel->R().transpose() * gaussModel->R();
                Point3 measured3d = lmfactor3d->measured();
                stream << "EDGE_SE3_XYZ " << index(lmfactor3d->key1()) << " " << index(lmfactor3d->key2()) << " "
                       << std::fixed << std::setprecision(9) << measured3d.x() << " " << measured3d.y() << " "
                       << measured3d.z();
                for (size_t i = 0; i < 3; i++) {
                    for (size_t j = i; j < 3; j++) {
                        stream << " " << std::fixed << std::setprecision(9) << Info(i, j);
                    }
                }
                stream << std::endl;
            }
        }
    }
    stream.close();
}

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
                             const std::string& filename)
{
    std::ofstream g2o_file(filename, std::ios_base::app);

    std::shared_ptr<gtsam::PoseToPointFactor<gtsam::Pose2, gtsam::Point2>> measFactor2d;
    std::shared_ptr<gtsam::PoseToPointFactor<gtsam::Pose3, gtsam::Point3>> measFactor3d;
    for (const auto& factor : graph) {
        measFactor2d = std::dynamic_pointer_cast<gtsam::PoseToPointFactor<gtsam::Pose2, gtsam::Point2>>(factor);
        measFactor3d = std::dynamic_pointer_cast<gtsam::PoseToPointFactor<gtsam::Pose3, gtsam::Point3>>(factor);
        if (measFactor2d) {
            // TODO(odin): Not implemented yet
            std::cerr << "writing G2o for 2D not implemented!\n";
        }
        if (measFactor3d) {
            g2o_file << "EDGE_SE3_XYZ " << gtsam::symbolIndex(measFactor3d->key1()) << " "
                     << gtsam::symbolIndex(measFactor3d->key2()) << " " << measFactor3d->measured().transpose() << " ";
            gtsam::Vector sigmas = measFactor3d->noiseModel()->sigmas();
            size_t dim = sigmas.size();
            for (int i = 0; i < dim; i++) {
                for (int j = i; j < dim; j++) {
                    g2o_file << (i == j ? 1.0 / (sigmas(i) * sigmas(i))
                                        : 0.0);  // we store the information matrix, so store 1/\sigma^2
                    g2o_file << (i == dim - 1 ? "\n" : " ");
                }
            }
        }
    }
}

// NOLINTEND