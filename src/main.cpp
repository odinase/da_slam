#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/sam/BearingRangeFactor.h>
#include <gtsam/slam/dataset.h>
#include <gtsam_unstable/slam/PoseToPointFactor.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <magic_enum.hpp>
#include <tuple>

#include "da_slam/argparse.hpp"
#include "da_slam/config.hpp"
#include "da_slam/data_association/assignment_solvers/auction.hpp"
#include "da_slam/data_association/assignment_solvers/hungarian.hpp"
#include "da_slam/data_association/known_data_association.hpp"
#include "da_slam/data_association/maximum_likelihood.hpp"
#include "da_slam/fmt.hpp"
#include "da_slam/slam/slam.hpp"
#include "da_slam/slam/utils_g2o.hpp"
#include "da_slam/types.hpp"

namespace fs = std::filesystem;

int main(const int argc, const char* argv[])
{
    //  Load in data from command line. Should be in config
    const auto [g2oFile, is3D, ic_prob, range_threshold, output_file] = da_slam::argparse::parse_args(argc, argv);

    const auto yaml_path = fs::current_path() / "config" / "config.yaml";
    const auto conf = da_slam::config::Config{yaml_path};

    const da_slam::slam::OptimizationMethod optimization_method = conf.optimization_method;
    const gtsam::Marginals::Factorization marginals_factorization = conf.marginals_factorization;

    std::vector<std::shared_ptr<gtsam::PoseToPointFactor<gtsam::Pose2, gtsam::Point2>>> measFactors2d{};
    std::vector<std::shared_ptr<gtsam::PoseToPointFactor<gtsam::Pose3, gtsam::Point3>>> measFactors3d{};

    std::vector<std::shared_ptr<gtsam::BetweenFactor<gtsam::Pose2>>> odomFactors2d{};
    std::vector<std::shared_ptr<gtsam::BetweenFactor<gtsam::Pose3>>> odomFactors3d{};

    // reading file and creating factor graph
    gtsam::NonlinearFactorGraph::shared_ptr graph;
    gtsam::Values::shared_ptr initial;
    std::tie(graph, initial) = gtsam::readG2owithLmks(g2oFile, is3D, "none");
    auto [odomFactorIdx, measFactorIdx] =
        findFactors(odomFactors2d, odomFactors3d, measFactors2d, measFactors3d, graph);
    double total_time = 0.0;
    std::chrono::high_resolution_clock::time_point start_t;
    std::chrono::high_resolution_clock::time_point end_t;
    double final_error;
    gtsam::Values estimates;
    bool caught_exception = false;

    da_slam::data_association::AssociationMethod association_method = conf.association_method;

    spdlog::info("Using association method {}", magic_enum::enum_name(conf.association_method));
    spdlog::info("Using optimization method {}", magic_enum::enum_name(conf.optimization_method));
    spdlog::info("Using assignment solver {}", magic_enum::enum_name(conf.assignment_solver));
    spdlog::info("Using marginals factorization {}", magic_enum::enum_name(conf.marginals_factorization));

    std::unique_ptr<da_slam::data_association::assignment_solvers::IAssignmentSolver> assignment_solver{};
    using namespace da_slam::data_association::assignment_solvers;
    switch (conf.assignment_solver) {
        case AssignmentSolver::AUCTION:
        {
            assignment_solver = std::make_unique<Auction>(AuctionParameters{});
            break;
        }
        case AssignmentSolver::HUNGARIAN:
        {
            assignment_solver = std::make_unique<Hungarian>();
            break;
        }
    }

    try {
        if (is3D) {
            const double sigmas = sqrt(da_slam::utils::chi2inv(ic_prob, 3));
            const auto ml_config =
                da_slam::data_association::maximum_likelihood::MaximumLikelihoodParameters{sigmas, range_threshold};
            gtsam::Vector pose_prior_noise = (gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished();
            pose_prior_noise = pose_prior_noise.array().sqrt().matrix();  // Calc sigmas from variances
            std::vector<da_slam::types::Timestep3D> timesteps = convert_into_timesteps(odomFactors3d, measFactors3d);
            da_slam::slam::Slam3D slam_sys{};
            std::unique_ptr<da_slam::data_association::IDataAssociation<da_slam::types::Measurement3D>> data_asso{};

            switch (association_method) {
                case da_slam::data_association::AssociationMethod::MAXIMUM_LIKELIHOOD:
                {
                    data_asso = std::make_unique<da_slam::data_association::maximum_likelihood::MaximumLikelihood3D>(
                        ml_config, std::move(assignment_solver));
                    break;
                }
                case da_slam::data_association::AssociationMethod::KNOWN_DATA_ASSOCIATION:
                {
                    std::map<uint64_t, gtsam::Key> meas_lmk_assos =
                        measurement_landmarks_associations(measFactors3d, timesteps);
                    data_asso = std::make_unique<da_slam::data_association::ground_truth::KnownDataAssociation3D>(
                        meas_lmk_assos);
                    break;
                }
            }

            slam_sys.initialize(pose_prior_noise, std::move(data_asso), optimization_method, marginals_factorization);
            const auto tot_timesteps = timesteps.size();
            for (const auto& timestep : timesteps) {
                start_t = std::chrono::high_resolution_clock::now();
                slam_sys.process_timestep(timestep);
                end_t = std::chrono::high_resolution_clock::now();
                double duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_t - start_t).count() * 1e-9;
                spdlog::info("Processed timestep {}, {}%% complete", timestep.step, static_cast<double>(timestep.step + 1) / tot_timesteps * 100.0);
                total_time += duration;
                final_error = slam_sys.error();
                estimates = slam_sys.current_estimates();
            }
            gtsam::NonlinearFactorGraph::shared_ptr graphNoKernel;
            gtsam::Values::shared_ptr initial2;
            std::tie(graphNoKernel, initial2) = gtsam::readG2o(g2oFile, is3D);
            writeG2o(*graphNoKernel, slam_sys.current_estimates(), output_file);
            std::string viz_filename = std::string("viz_ml") + std::string(".g2o");
            gtsam::NonlinearFactorGraph curr_graph = slam_sys.get_graph();
            writeG2oLdmkEdges(curr_graph, slam_sys.current_estimates(), viz_filename, output_file);
            slam_sys.get_graph().saveGraph("/home/odinase/prog/C++/da-slam/graph.txt", slam_sys.current_estimates());
        }
        else {
            const auto sigmas = sqrt(da_slam::utils::chi2inv(ic_prob, 2));
            const auto ml_config =
                da_slam::data_association::maximum_likelihood::MaximumLikelihoodParameters{sigmas, range_threshold};
            gtsam::Vector pose_prior_noise = gtsam::Vector3(1e-6, 1e-6, 1e-8);
            pose_prior_noise = pose_prior_noise.array().sqrt().matrix();  // Calc sigmas from variances
            std::vector<da_slam::types::Timestep2D> timesteps = convert_into_timesteps(odomFactors2d, measFactors2d);
            da_slam::slam::Slam2D slam_sys{};
            std::unique_ptr<da_slam::data_association::IDataAssociation<da_slam::types::Measurement2D>> data_asso{};

            switch (association_method) {
                case da_slam::data_association::AssociationMethod::MAXIMUM_LIKELIHOOD:
                {
                    data_asso = std::make_unique<da_slam::data_association::maximum_likelihood::MaximumLikelihood2D>(
                        ml_config, std::move(assignment_solver));
                    break;
                }
                case da_slam::data_association::AssociationMethod::KNOWN_DATA_ASSOCIATION:
                {
                    std::map<uint64_t, gtsam::Key> meas_lmk_assos =
                        measurement_landmarks_associations(measFactors2d, timesteps);
                    data_asso = std::make_unique<da_slam::data_association::ground_truth::KnownDataAssociation2D>(
                        meas_lmk_assos);
                    break;
                }
            }

            slam_sys.initialize(pose_prior_noise, std::move(data_asso), optimization_method, marginals_factorization);
            int tot_timesteps = timesteps.size();
            for (const auto& timestep : timesteps) {
                start_t = std::chrono::high_resolution_clock::now();
                slam_sys.process_timestep(timestep);
                end_t = std::chrono::high_resolution_clock::now();
                double duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end_t - start_t).count() * 1e-9;
#ifdef LOGGING
                avg_time = (timestep.step * avg_time + duration) / (timestep.step + 1.0);
                cout << "Duration: " << duration << " seconds\n"
                     << "Average time one iteration: " << avg_time << " seconds\n";
#endif
spdlog::info("Processed timestep {}, {}%% complete", timestep.step, static_cast<double>(timestep.step + 1) / tot_timesteps * 100.0);
                total_time += duration;
                final_error = slam_sys.error();
                estimates = slam_sys.current_estimates();
            }
            gtsam::NonlinearFactorGraph::shared_ptr graphNoKernel;
            gtsam::Values::shared_ptr initial2;
            std::tie(graphNoKernel, initial2) = gtsam::readG2o(g2oFile, is3D);
            writeG2o(*graphNoKernel, slam_sys.current_estimates(), output_file);
            std::string viz_filename = std::string("viz_ml") + std::string(".g2o");
            gtsam::NonlinearFactorGraph curr_graph = slam_sys.get_graph();
            writeG2oLdmkEdges(curr_graph, slam_sys.current_estimates(), viz_filename, output_file);
            slam_sys.get_graph().saveGraph("/home/odinase/prog/C++/da-slam/graph.txt", slam_sys.current_estimates());
        }
    }
    catch (const gtsam::IndeterminantLinearSystemException& err) {  // when run in terminal: tbb::captured_exception
        std::cout << "Optimization failed" << std::endl;
        std::cout << err.what() << std::endl;
        if (argc > 5) {
            std::string other_msg = "None";
            gtsam::saveException(output_file, std::string("ExceptionML.txt"), err.what(), other_msg);
        }

        gtsam::NonlinearFactorGraph::shared_ptr graphNoKernel;
        gtsam::Values::shared_ptr initial2;
        std::tie(graphNoKernel, initial2) = gtsam::readG2o(g2oFile, is3D);
        estimates = *initial2;
        caught_exception = true;
    }
    if (argc < 5) {
        if (caught_exception) {
            spdlog::error("exception caught! printing odometry");
        }
        estimates.print("results");
    }
    else {
        if (!caught_exception) {
            std::cout << "Writing results to file: " << output_file << std::endl;
            gtsam::NonlinearFactorGraph::shared_ptr graphNoKernel;
            gtsam::Values::shared_ptr initial2;
            std::tie(graphNoKernel, initial2) = gtsam::readG2o(g2oFile, is3D);
            writeG2o(*graphNoKernel, estimates,
                     output_file);  // can save pose, ldmk, odom not ldmk measurements
            gtsam::saveGraphErrors(output_file, std::string("maximum_likelihood"), std::vector<double>{final_error});
            gtsam::saveVector(output_file, std::string("errorsGraph.txt"), std::vector<double>{final_error});
            gtsam::saveVector(output_file, std::string("runTime.txt"), std::vector<double>{total_time});
            std::cout << "done! " << std::endl;
        }
    }

    return 0;
}