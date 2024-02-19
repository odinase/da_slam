#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/sam/BearingRangeFactor.h>
#include <gtsam/slam/dataset.h>
#include <gtsam_unstable/slam/PoseToPointFactor.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <tuple>

#include "da_slam/argparse.hpp"
#include "da_slam/config.hpp"
#include "da_slam/data_association/known_data_association.hpp"
#include "da_slam/data_association/maximum_likelihood.hpp"
#include "da_slam/slam/slam.hpp"
#include "da_slam/slam/utils_g2o.hpp"
#include "da_slam/types.hpp"
#include "da_slam/data_association/assignment_solvers/hungarian.hpp"
#include "da_slam/data_association/assignment_solvers/auction.hpp"
#include "da_slam/fmt.hpp"

#include <spdlog/spdlog.h>
#include <magic_enum.hpp>


using namespace std;
using namespace gtsam;
using namespace da_slam;

int main(const int argc, const char* argv[])
{
    const auto [g2oFile, is3D, ic_prob, range_threshold, output_file] = da_slam::argparse::parse_args(argc, argv);

    const auto yaml_path = std::filesystem::current_path() / "config" / "config.yaml";
    const config::Config conf(yaml_path);

    const slam::OptimizationMethod optimization_method = conf.optimization_method;
    const gtsam::Marginals::Factorization marginals_factorization = conf.marginals_factorization;

    vector<std::shared_ptr<PoseToPointFactor<Pose2, Point2>>> measFactors2d;
    vector<std::shared_ptr<PoseToPointFactor<Pose3, Point3>>> measFactors3d;

    vector<std::shared_ptr<BetweenFactor<Pose2>>> odomFactors2d;
    vector<std::shared_ptr<BetweenFactor<Pose3>>> odomFactors3d;

    // reading file and creating factor graph
    NonlinearFactorGraph::shared_ptr graph;
    Values::shared_ptr initial;
    std::tie(graph, initial) = readG2owithLmks(g2oFile, is3D, "none");
    auto [odomFactorIdx, measFactorIdx] =
        findFactors(odomFactors2d, odomFactors3d, measFactors2d, measFactors3d, graph);
    double total_time = 0.0;
    std::chrono::high_resolution_clock::time_point start_t;
    std::chrono::high_resolution_clock::time_point end_t;
    double final_error;
    Values estimates;
    bool caught_exception = false;

    da_slam::data_association::AssociationMethod association_method = conf.association_method;

    spdlog::info("Using association method {}",      magic_enum::enum_name(conf.association_method));
    spdlog::info("Using optimization method {}",     magic_enum::enum_name(conf.optimization_method));
    spdlog::info("Using marginals factorization {}", magic_enum::enum_name(conf.marginals_factorization));

    auto assignment_solver = std::make_unique<data_association::assignment_solvers::Hungarian>();

//     try {
//         if (is3D) {
//             const double sigmas = sqrt(utils::chi2inv(ic_prob, 3));
//             const auto ml_config = data_association::maximum_likelihood::MaximumLikelihoodParameters{sigmas, range_threshold};
//             gtsam::Vector pose_prior_noise = (gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished();
//             pose_prior_noise = pose_prior_noise.array().sqrt().matrix();  // Calc sigmas from variances
//             vector<types::Timestep3D> timesteps = convert_into_timesteps(odomFactors3d, measFactors3d);
//             slam::Slam3D slam_sys{};
//             std::unique_ptr<data_association::IDataAssociation<types::Measurement3D>> data_asso{};

//             switch (association_method) {
//                 case data_association::AssociationMethod::MAXIMUM_LIKELIHOOD:
//                 {
//                     data_asso =
//                         std::make_unique<data_association::maximum_likelihood::MaximumLikelihood3D>(ml_config, std::move(assignment_solver));
//                     break;
//                 }
//                 case data_association::AssociationMethod::KNOWN_DATA_ASSOCIATION:
//                 {
//                     std::map<uint64_t, gtsam::Key> meas_lmk_assos =
//                         measurement_landmarks_associations(measFactors3d, timesteps);
//                     data_asso = std::make_unique<data_association::ground_truth::KnownDataAssociation3D>(meas_lmk_assos);
//                     break;
//                 }
//             }

//             slam_sys.initialize(pose_prior_noise, std::move(data_asso), optimization_method, marginals_factorization);
//             int tot_timesteps = timesteps.size();
//             for (const auto& timestep : timesteps) {
//                 start_t = std::chrono::high_resolution_clock::now();
//                 slam_sys.process_timestep(timestep);
//                 end_t = std::chrono::high_resolution_clock::now();
//                 double duration = chrono::duration_cast<chrono::nanoseconds>(end_t - start_t).count() * 1e-9;
// #ifdef HEARTBEAT
//                 cout << "Processed timestep " << timestep.step << ", "
//                      << double(timestep.step + 1) / tot_timesteps * 100.0 << "\% complete\n";
// #endif
//                 total_time += duration;
//                 final_error = slam_sys.error();
//                 estimates = slam_sys.current_estimates();
//             }
//             NonlinearFactorGraph::shared_ptr graphNoKernel;
//             Values::shared_ptr initial2;
//             std::tie(graphNoKernel, initial2) = readG2o(g2oFile, is3D);
//             writeG2o(*graphNoKernel, slam_sys.current_estimates(), output_file);
//             string viz_filename = string("viz_ml") + string(".g2o");
//             gtsam::NonlinearFactorGraph curr_graph = slam_sys.get_graph();
//             writeG2oLdmkEdges(curr_graph, slam_sys.current_estimates(), viz_filename, output_file);
//             slam_sys.get_graph().saveGraph("/home/odinase/prog/C++/da-slam/graph.txt", slam_sys.current_estimates());
//         }
//         else {
//             double sigmas = sqrt(da_slam::utils::chi2inv(ic_prob, 2));
//             gtsam::Vector pose_prior_noise = Vector3(1e-6, 1e-6, 1e-8);
//             pose_prior_noise = pose_prior_noise.array().sqrt().matrix();  // Calc sigmas from variances
//             vector<types::Timestep2D> timesteps = convert_into_timesteps(odomFactors2d, measFactors2d);
//             slam::Slam2D slam_sys{};
//             std::unique_ptr<data_association::IDataAssociation<types::Measurement2D>> data_asso{};

//             switch (association_method) {
//                 case da_slam::data_association::AssociationMethod::MAXIMUM_LIKELIHOOD:
//                 {
//                     data_asso = std::make_unique<data_association::maximum_likelihood::MaximumLikelihood2D>(sigmas, range_threshold);
//                     break;
//                 }
//                 case da_slam::data_association::AssociationMethod::KNOWN_DATA_ASSOCIATION:
//                 {
//                     std::map<uint64_t, gtsam::Key> meas_lmk_assos =
//                         measurement_landmarks_associations(measFactors2d, timesteps);
//                     data_asso = std::make_unique<data_association::ground_truth::KnownDataAssociation2D>(meas_lmk_assos);
//                     break;
//                 }
//             }

//             slam_sys.initialize(pose_prior_noise, std::move(data_asso), optimization_method, marginals_factorization);
//             int tot_timesteps = timesteps.size();
//             for (const auto& timestep : timesteps) {
//                 start_t = std::chrono::high_resolution_clock::now();
//                 slam_sys.process_timestep(timestep);
//                 end_t = std::chrono::high_resolution_clock::now();
//                 double duration = chrono::duration_cast<chrono::nanoseconds>(end_t - start_t).count() * 1e-9;
// #ifdef LOGGING
//                 avg_time = (timestep.step * avg_time + duration) / (timestep.step + 1.0);
//                 cout << "Duration: " << duration << " seconds\n"
//                      << "Average time one iteration: " << avg_time << " seconds\n";
// #endif
// #ifdef HEARTBEAT
//                 cout << "Processed timestep " << timestep.step << ", "
//                      << double(timestep.step + 1) / tot_timesteps * 100.0 << "\% complete\n";
// #endif
//                 total_time += duration;
//                 final_error = slam_sys.error();
//                 estimates = slam_sys.current_estimates();
//             }
//             NonlinearFactorGraph::shared_ptr graphNoKernel;
//             Values::shared_ptr initial2;
//             std::tie(graphNoKernel, initial2) = readG2o(g2oFile, is3D);
//             writeG2o(*graphNoKernel, slam_sys.current_estimates(), output_file);
//             string viz_filename = string("viz_ml") + string(".g2o");
//             gtsam::NonlinearFactorGraph curr_graph = slam_sys.get_graph();
//             writeG2oLdmkEdges(curr_graph, slam_sys.current_estimates(), viz_filename, output_file);
//             slam_sys.get_graph().saveGraph("/home/odinase/prog/C++/da-slam/graph.txt", slam_sys.current_estimates());
//         }
//     }
//     catch (const gtsam::IndeterminantLinearSystemException& err) {  // when run in terminal: tbb::captured_exception
//         std::cout << "Optimization failed" << std::endl;
//         std::cout << err.what() << std::endl;
//         if (argc > 5) {
//             string other_msg = "None";
//             saveException(output_file, std::string("ExceptionML.txt"), err.what(), other_msg);
//         }

//         NonlinearFactorGraph::shared_ptr graphNoKernel;
//         Values::shared_ptr initial2;
//         std::tie(graphNoKernel, initial2) = readG2o(g2oFile, is3D);
//         estimates = *initial2;
//         caught_exception = true;
//     }
//     if (argc < 5) {
//         if (caught_exception) {
//             cout << "exception caught! printing odometry" << endl;
//         }
//         estimates.print("results");
//     }
//     else {
//         if (!caught_exception) {
//             std::cout << "Writing results to file: " << output_file << std::endl;
//             NonlinearFactorGraph::shared_ptr graphNoKernel;
//             Values::shared_ptr initial2;
//             std::tie(graphNoKernel, initial2) = readG2o(g2oFile, is3D);
//             writeG2o(*graphNoKernel, estimates,
//                      output_file);  // can save pose, ldmk, odom not ldmk measurements
//             saveGraphErrors(output_file, std::string("maximum_likelihood"), vector<double>{final_error});
//             saveVector(output_file, std::string("errorsGraph.txt"), vector<double>{final_error});
//             saveVector(output_file, std::string("runTime.txt"), vector<double>{total_time});
//             std::cout << "done! " << std::endl;
//         }
//     }

    return 0;
}