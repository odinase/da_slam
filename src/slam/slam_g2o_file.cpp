
#include <gtsam/geometry/Pose2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/sam/BearingRangeFactor.h>
#include <gtsam/slam/dataset.h>
#include <gtsam_unstable/slam/PoseToPointFactor.h>
// #include "slam/utils.h"
#include <cmath>
#include <fstream>
#include <tuple>
#include <algorithm>

#include <glog/logging.h>

// #include "slam/slam_g2o_file.h"
#include "slam/utils_g2o.h"
#include "slam/slam.h"
#include "slam/types.h"
#include "data_association/ml/MaximumLikelihood.h"

using gtsam::symbol_shorthand::L; // gtsam/slam/dataset.cpp
using namespace std;
using namespace gtsam;

// SLAM_G2O_file::SLAM_G2O_file(const std::string& filename) : g2o_filename_(filename) {
//     boost::tie()
// }

int main(int argc, char **argv)
{
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
    // default
    string g2oFile = findExampleDataFile("noisyToyGraph.txt");
    bool is3D = false;
    double sigmas = 3.0;
    std::string output_file;
    double range_threshold = 1e9;
    // Parse user's inputs
    if (argc > 1)
    {
        g2oFile = argv[1]; // input dataset filename
    }
    if (argc > 2)
    {
        is3D = atoi(argv[2]);
        std::cout << "is3D: " << is3D << std::endl;
    }
    if (argc > 3)
    {
        sigmas = atof(argv[3]);
        std::cout << "sigmas: " << sigmas << std::endl;
    }
    if (argc > 4)
    {
        range_threshold = atof(argv[4]);
        std::cout << "range_threshold: " << range_threshold << std::endl;
    }
    if (argc > 5)
    {
        output_file = argv[5];
        std::cout << "output_file: " << output_file << std::endl;
    }

    vector<boost::shared_ptr<PoseToPointFactor<Pose2, Point2>>> measFactors2d;
    vector<boost::shared_ptr<PoseToPointFactor<Pose3, Point3>>> measFactors3d;

    vector<boost::shared_ptr<BetweenFactor<Pose2>>> odomFactors2d;
    vector<boost::shared_ptr<BetweenFactor<Pose3>>> odomFactors3d;

    // reading file and creating factor graph
    NonlinearFactorGraph::shared_ptr graph;
    Values::shared_ptr initial;
    boost::tie(graph, initial) = readG2owithLmks(g2oFile, is3D, "none");
    auto [odomFactorIdx, measFactorIdx] = findFactors(odomFactors2d, odomFactors3d, measFactors2d, measFactors3d, graph);
    double avg_time = 0.0;
    double total_time = 0.0;
    std::chrono::high_resolution_clock::time_point start_t;
    std::chrono::high_resolution_clock::time_point end_t;
    double final_error;
    Values estimates;
    try
    {
        if (is3D)
        {
            gtsam::Vector pose_prior_noise = (gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished();
            pose_prior_noise = pose_prior_noise.array().sqrt().matrix(); // Calc sigmas from variances
            vector<slam::Timestep3D> timesteps = convert_into_timesteps(odomFactors3d, measFactors3d);
            slam::SLAM3D slam_sys{};
            std::shared_ptr<da::DataAssociation<slam::Measurement<gtsam::Point3>>> data_asso = std::make_shared<da::ml::MaximumLikelihood3D>(sigmas, range_threshold);
            slam_sys.initialize(pose_prior_noise, data_asso);
            int tot_timesteps = timesteps.size();
            for (const auto &timestep : timesteps)
            {
                start_t = std::chrono::high_resolution_clock::now();
                slam_sys.processTimestep(timestep);
                end_t = std::chrono::high_resolution_clock::now();
                double duration = chrono::duration_cast<chrono::nanoseconds>(end_t - start_t).count() * 1e-9;
                avg_time = (timestep.step * avg_time + duration) / (timestep.step + 1.0);
                cout << "Duration: " << duration << " seconds\n"
                     << "Average time one iteration: " << avg_time << " seconds\n";
                cout << "Processed timestep " << timestep.step << ", " << double(timestep.step + 1) / tot_timesteps * 100.0 << "\% complete\n";
                total_time += duration;
                final_error = slam_sys.error();
                estimates = slam_sys.currentEstimates();
            }
            NonlinearFactorGraph::shared_ptr graphNoKernel;
            Values::shared_ptr initial2;
            boost::tie(graphNoKernel, initial2) = readG2o(g2oFile, is3D);
            writeG2o(*graphNoKernel, slam_sys.currentEstimates(), output_file);
            ofstream os("/home/odinase/prog/C++/da-slam/graph.txt");
            slam_sys.getGraph().saveGraph(os, slam_sys.currentEstimates());
            os.close();
        }
        else
        {
            gtsam::Vector pose_prior_noise = Vector3(1e-6, 1e-6, 1e-8);
            pose_prior_noise = pose_prior_noise.array().sqrt().matrix(); // Calc sigmas from variances
            cout << "Start converting into timesteps!\n";
            vector<slam::Timestep2D> timesteps = convert_into_timesteps(odomFactors2d, measFactors2d);
            cout << "Done converting into timesteps!\n";
            slam::SLAM2D slam_sys{};
            std::shared_ptr<da::DataAssociation<slam::Measurement<gtsam::Point2>>> data_asso = std::make_shared<da::ml::MaximumLikelihood2D>(sigmas, range_threshold);
            slam_sys.initialize(pose_prior_noise, data_asso);
            cout << "SLAM system initialized!\n";
            int tot_timesteps = timesteps.size();
            for (const auto &timestep : timesteps)
            {
                start_t = std::chrono::high_resolution_clock::now();
                slam_sys.processTimestep(timestep);
                end_t = std::chrono::high_resolution_clock::now();
                double duration = chrono::duration_cast<chrono::nanoseconds>(end_t - start_t).count() * 1e-9;
                avg_time = (timestep.step * avg_time + duration) / (timestep.step + 1.0);
                cout << "Processed timestep " << timestep.step << ", " << double(timestep.step + 1) / tot_timesteps * 100.0 << "\% complete\n";
                // cout << "Duration: " << duration << " seconds\n"
                //      << "Average time one iteration: " << avg_time << " seconds\n";
                total_time += duration;
                final_error = slam_sys.error();
                estimates = slam_sys.currentEstimates();
            }
            NonlinearFactorGraph::shared_ptr graphNoKernel;
            Values::shared_ptr initial2;
            boost::tie(graphNoKernel, initial2) = readG2o(g2oFile, is3D);
            writeG2o(*graphNoKernel, slam_sys.currentEstimates(), output_file);
                        ofstream os("/home/odinase/prog/C++/da-slam/graph.txt");
            slam_sys.getGraph().saveGraph(os, slam_sys.currentEstimates());
            os.close();

        }
    }
    catch (gtsam::IndeterminantLinearSystemException &indetErr)
    { // when run in terminal: tbb::captured_exception
        std::cout << "Optimization failed" << std::endl;
        std::cout << indetErr.what() << std::endl;
        NonlinearFactorGraph::shared_ptr graphNoKernel;
        Values::shared_ptr initial2;
        boost::tie(graphNoKernel, initial2) = readG2o(g2oFile, is3D);
        estimates = *initial2;
    }
    if (argc < 5)
    {
        estimates.print("results");
    }
    else
    {
        std::cout << "Writing results to file: " << output_file << std::endl;
        NonlinearFactorGraph::shared_ptr graphNoKernel;
        Values::shared_ptr initial2;
        boost::tie(graphNoKernel, initial2) = readG2o(g2oFile, is3D);
        writeG2o(*graphNoKernel, estimates,
                 output_file); // can save pose, ldmk, odom not ldmk measurements
        saveGraphErrors(output_file, std::string("maximum_likelihood"), vector<double>{final_error});
        saveVector(output_file, std::string("errorsGraph.txt"), vector<double>{final_error});
        saveVector(output_file, std::string("runTime.txt"), vector<double>{total_time});
        std::cout << "done! " << std::endl;
    }
}