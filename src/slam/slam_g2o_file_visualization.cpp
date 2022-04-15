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
#include <string>
#include <sstream>

#ifdef GLOG_AVAILABLE
#include <glog/logging.h>
#endif

// #include "slam/slam_g2o_file.h"
#include "slam/utils_g2o.h"
#include "slam/slam.h"
#include "slam/types.h"
#include "data_association/ml/MaximumLikelihood.h"
#include "data_association/gt/KnownDataAssociation.h"
#include "data_association/DataAssociation.h"
#include "visualization/visualization.h"
#include "config/config.h"

using namespace std;
using namespace gtsam;

using gtsam::symbol_shorthand::L;
using gtsam::symbol_shorthand::X;

namespace viz = visualization;

std::optional<std::pair<gtsam::Key, gtsam::Key>> nonlinearFactor2keys(const gtsam::NonlinearFactor::shared_ptr &factor)
{
    boost::shared_ptr<PoseToPointFactor<Pose2, Point2>> measFactor2d = boost::dynamic_pointer_cast<PoseToPointFactor<Pose2, Point2>>(factor);
    if (measFactor2d)
    {
        return std::make_pair(measFactor2d->key1(), measFactor2d->key2());
    }

    boost::shared_ptr<PoseToPointFactor<Pose3, Point3>> measFactor3d = boost::dynamic_pointer_cast<PoseToPointFactor<Pose3, Point3>>(factor);
    if (measFactor3d)
    {
        return std::make_pair(measFactor3d->key1(), measFactor3d->key2());
    }

    boost::shared_ptr<BetweenFactor<Pose2>> odomFactor2d = boost::dynamic_pointer_cast<BetweenFactor<Pose2>>(factor);
    if (odomFactor2d)
    {
        return std::make_pair(odomFactor2d->key1(), odomFactor2d->key2());
    }

    boost::shared_ptr<BetweenFactor<Pose3>> odomFactor3d = boost::dynamic_pointer_cast<BetweenFactor<Pose3>>(factor);
    if (odomFactor3d)
    {
        return std::make_pair(odomFactor3d->key1(), odomFactor3d->key2());
    }

    return std::nullopt;
}

gtsam::KeySet dfs(const gtsam::NonlinearFactorGraph &graph)
{
    if (graph.size() == 0)
    {
        return {};
    }

    gtsam::KeySet key_set;
    std::deque<size_t> factors;
    gtsam::Key k = X(0); // Must at least have the first pose;
    factors.push_front(k);

    while (factors.size() > 0)
    {
        k = factors.front();
        factors.pop_front();
        key_set.insert(k);
        auto finder = [&k](const gtsam::NonlinearFactor::shared_ptr factor) -> bool
        {
            auto keys = nonlinearFactor2keys(factor);
            return (keys ? keys->first == k : false);
        };
        auto it = graph.begin();
        while ((it = std::find_if(it, graph.end(), finder)) != graph.end())
        {
            auto keys = nonlinearFactor2keys(*it);
            if (keys)
            {
                factors.push_front(keys->second);
            }
            ++it;
        }
    }
    return key_set;
}

bool connected_graph(const gtsam::NonlinearFactorGraph &graph, const gtsam::Values &estimates)
{
    gtsam::KeySet keys_from_x0 = dfs(graph);
    for (gtsam::Key k : estimates.keys())
    {
        if (!keys_from_x0.exists(k))
        {
            return false;
        }
    }
    return true;
}

int main(int argc, char **argv)
{
#ifdef GLOG_AVAILABLE
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();
#endif
    // default
    string g2oFile = findExampleDataFile("noisyToyGraph.txt");
    bool is3D = false;
    double ic_prob = 0.99;
    std::string output_file;
    double range_threshold = 1e9;
    // Parse user's inputs
    if (argc > 1)
    {
        if (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0)
        {
            cout << "Input args: <input dataset filename> <is3D> <ic prob> <range threshold> <output dataset filename>\n";
            return 0;
        }
        g2oFile = argv[1]; // input dataset filename
    }
    if (argc > 2)
    {
        is3D = atoi(argv[2]);
        std::cout << "is3D: " << is3D << std::endl;
    }
    if (argc > 3)
    {
        ic_prob = atof(argv[3]);
        std::cout << "ic_prob: " << ic_prob << std::endl;
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
#ifdef LOGGING
    double avg_time = 0.0;
#endif
    double total_time = 0.0;
    std::chrono::high_resolution_clock::time_point start_t;
    std::chrono::high_resolution_clock::time_point end_t;
    double final_error;
    Values estimates;
    NonlinearFactorGraph nlf_graph;
    bool caught_exception = false;

    // Setup visualization
    if (!viz::init())
    {
        cout << "Failed to initialize visualization, aborting!\n";
        return -1;
    }
    else
    {
        cout << "Visualization initialized!\n";
    }

    bool early_stop = false;
    bool next_timestep = true;

    config::Config conf("/home/mrg/prog/C++/da-slam/config/config.yaml");

    bool enable_stepping = conf.enable_stepping;
    bool draw_factor_graph = conf.draw_factor_graph;
    bool enable_step_limit = conf.enable_step_limit;
    int step_to_increment_to = conf.step_to_increment_to;
    bool autofit = conf.autofit;

    bool draw_factor_graph_ground_truth = conf.draw_factor_graph_ground_truth;
    bool enable_factor_graph_window = conf.enable_factor_graph_window;
    int factor_graph_window = enable_factor_graph_window ? conf.factor_graph_window : 0;

    bool with_ground_truth = conf.with_ground_truth;

    slam::OptimizationMethod optimization_method = conf.optimization_method;
    gtsam::Marginals::Factorization marginals_factorization = conf.marginals_factorization;

    std::map<gtsam::Key, bool> lmk_to_draw_covar;
    bool clear_landmarks = true;
    bool check_all_landmarks = true;
    bool uncheck_all_landmarks = false;
    bool stop_at_association_timestep = conf.stop_at_association_timestep;
    bool did_association = false;
    bool proceed_to_next_asso_timestep = !stop_at_association_timestep;
    bool draw_association_hypothesis = conf.draw_association_hypothesis;

    std::cout << "Using association method " << conf.association_method << "\n";
    std::cout << (with_ground_truth ? "Adding" : "Not adding") << " ground truth for comparison\n";

    std::stringstream ss;
    std::string factor_graph_title, lmk_to_draw_covar_label;
    ss << "Factor graph - Data association method: " << conf.association_method;
    factor_graph_title = ss.str();

    std::cout << "Using optimization method " << conf.optimization_method << "\n";
    std::cout << "Using marginals factorization " << (conf.marginals_factorization == gtsam::Marginals::CHOLESKY ? "Cholesky" : "QR") << "\n";

    try
    {
        if (is3D)
        {
            double sigmas = sqrt(da::chi2inv(ic_prob, 3));
            gtsam::Vector pose_prior_noise = (gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished();
            pose_prior_noise = pose_prior_noise.array().sqrt().matrix(); // Calc sigmas from variances
            vector<slam::Timestep3D> timesteps = convert_into_timesteps(odomFactors3d, measFactors3d);
            slam::SLAM3D slam_sys{};

            std::shared_ptr<da::DataAssociation<slam::Measurement<gtsam::Point3>>> data_asso;
            std::shared_ptr<da::DataAssociation<slam::Measurement<gtsam::Point3>>> data_asso_gt;
            slam::SLAM3D slam_sys_gt{};
            if (with_ground_truth)
            {
                data_asso = std::make_shared<da::ml::MaximumLikelihood3D>(sigmas, range_threshold);
                std::map<uint64_t, gtsam::Key> meas_lmk_assos = measurement_landmarks_associations(
                    measFactors3d,
                    timesteps);
                data_asso_gt = std::make_shared<da::gt::KnownDataAssociation3D>(meas_lmk_assos);
                slam_sys_gt.initialize(pose_prior_noise, data_asso_gt);
            }
            else
            {
                switch (conf.association_method)
                {
                case da::AssociationMethod::MaximumLikelihood:
                {
                    data_asso = std::make_shared<da::ml::MaximumLikelihood3D>(sigmas, range_threshold);
                    break;
                }
                case da::AssociationMethod::KnownDataAssociation:
                {
                    std::map<uint64_t, gtsam::Key> meas_lmk_assos = measurement_landmarks_associations(
                        measFactors3d,
                        timesteps);
                    data_asso = std::make_shared<da::gt::KnownDataAssociation3D>(meas_lmk_assos);
                    break;
                }
                }
            }

            slam_sys.initialize(pose_prior_noise, data_asso, optimization_method, marginals_factorization);

            int tot_timesteps = timesteps.size();

            int i = 0;
            int step = timesteps[i].step;
            while (viz::running() && i < tot_timesteps)
            {
                viz::new_frame();

                ImGui::Begin("Menu");

                viz::progress_bar(step, tot_timesteps);
                ImGui::Checkbox("Enable stepping", &enable_stepping);
                if (enable_stepping)
                {
                    ImGui::SameLine(0.0f, 100.0f);
                    next_timestep = ImGui::Button("Next timestep");
                }
                else
                {
                    next_timestep = true;
                }
                ImGui::Checkbox("Set step to increment to", &enable_step_limit);
                if (enable_step_limit)
                {
                    // ImGui::SameLine(0.0f, 100.0f);
                    ImGui::SetNextItemWidth(150.0f);
                    ImGui::InputInt("Step to increment to", &step_to_increment_to);
                }

                ImGui::Checkbox("Draw factor graph", &draw_factor_graph);
                if (with_ground_truth)
                {
                    ImGui::Checkbox("Draw factor graph ground truth", &draw_factor_graph_ground_truth);
                }
                if (draw_factor_graph || (with_ground_truth && draw_factor_graph_ground_truth))
                {
                    ImGui::Checkbox("Enable factor graph window", &enable_factor_graph_window);
                    if (enable_factor_graph_window)
                    {
                        ImGui::SetNextItemWidth(150.0f);
                        ImGui::InputInt("\0", &factor_graph_window);
                        ImGui::SameLine();
                        ImGui::TextWrapped("Factor graph window (how many latest timesteps to draw)");
                    }
                }
                ImGui::Checkbox("Autofit plot", &autofit);
                ImGui::Checkbox("Draw association hypotheses", &draw_association_hypothesis);
                if (draw_association_hypothesis)
                {
                    ImGui::Checkbox("Break at timestep with measurements", &stop_at_association_timestep);
                    if (stop_at_association_timestep)
                    {
                        ImGui::SameLine();
                        proceed_to_next_asso_timestep = ImGui::Button("Proceed to next association timestep");
                    }
                }
                ImGui::End(); // Menu

                if (next_timestep && (!enable_step_limit || step < step_to_increment_to) && (!draw_association_hypothesis || proceed_to_next_asso_timestep || !(stop_at_association_timestep && did_association)))
                {
                    const slam::Timestep3D &timestep = timesteps[i];

                    step = timestep.step;

                    start_t = std::chrono::high_resolution_clock::now();

                    slam_sys.processTimestep(timestep);
                    if (with_ground_truth)
                    {
                        slam_sys_gt.processTimestep(timestep);
                    }

                    // If we received measurements, we must have done data association
                    did_association = timestep.measurements.size() > 0;

                    end_t = std::chrono::high_resolution_clock::now();
                    double duration = chrono::duration_cast<chrono::nanoseconds>(end_t - start_t).count() * 1e-9;
#ifdef LOGGING
                    avg_time = (timestep.step * avg_time + duration) / (timestep.step + 1.0);
                    cout << "Duration: " << duration << " seconds\n"
                         << "Average time one iteration: " << avg_time << " seconds\n";
#endif
#ifdef HEARTBEAT
                    cout << "Processed timestep " << timestep.step << ", " << double(timestep.step + 1) / tot_timesteps * 100.0 << "\% complete\n";
#endif
                    total_time += duration;
                    final_error = slam_sys.error();
                    estimates = slam_sys.currentEstimates();
                    if (enable_stepping)
                    {
                        next_timestep = false;
                    }
                    clear_landmarks = true;
                    i++;
                }

                if (draw_factor_graph || (with_ground_truth && draw_factor_graph_ground_truth))
                {
                    int latest_timestep_to_draw;
                    if (enable_factor_graph_window)
                    {
                        latest_timestep_to_draw = i - factor_graph_window;
                    }
                    else
                    {
                        latest_timestep_to_draw = 0;
                    }
                    ImGui::Begin(factor_graph_title.c_str());
                    if (autofit)
                    {
                        ImPlot::SetNextAxesToFit();
                    }
                    if (ImPlot::BeginPlot("##factor graph", ImVec2(-1, -1)))
                    {
                        if (draw_factor_graph)
                        {
                            viz::draw_factor_graph(slam_sys.getGraph(), slam_sys.currentEstimates(), latest_timestep_to_draw);
                        }
                        if (with_ground_truth && draw_factor_graph_ground_truth)
                        {
                            viz::draw_factor_graph_ground_truth(slam_sys_gt.getGraph(), slam_sys_gt.currentEstimates(), latest_timestep_to_draw);
                        }
                        ImPlot::EndPlot();
                    }
                    ImGui::End(); // Factor graph
                }

                if (draw_association_hypothesis)
                {
                    ImGui::Begin("Association hypothesis");
                    if (ImPlot::BeginPlot("##hypothesis", ImVec2(-1, -1)))
                    {
                        const auto &hypo = slam_sys.latestHypothesis();
                        ss.str("");

                        // Hypothesis with no measurements is no use
                        if (hypo.num_measurements() > 0 && i > 0)
                        {
                            ImGui::Begin("Landmark covariances to draw");
                            check_all_landmarks = ImGui::Button("Check all");
                            ImGui::SameLine();
                            uncheck_all_landmarks = ImGui::Button("Uncheck all");

                            if (clear_landmarks)
                            {
                                lmk_to_draw_covar.clear();
                                clear_landmarks = false;
                                uncheck_all_landmarks = false;
                                check_all_landmarks = true;
                            }
                            gtsam::KeyVector lmk_keys = hypo.associated_landmarks();
                            std::sort(lmk_keys.begin(), lmk_keys.end(), [](gtsam::Key lhs, gtsam::Key rhs)
                                      { return gtsam::symbolIndex(lhs) < gtsam::symbolIndex(rhs); });
                            for (const auto &lmk : lmk_keys)
                            {
                                ss << gtsam::Symbol(lmk);
                                lmk_to_draw_covar_label = ss.str();
                                ss.str("");
                                if (check_all_landmarks)
                                {
                                    lmk_to_draw_covar[lmk] = true;
                                }
                                if (uncheck_all_landmarks)
                                {
                                    lmk_to_draw_covar[lmk] = false;
                                }
                                ImGui::Checkbox(lmk_to_draw_covar_label.c_str(), &lmk_to_draw_covar[lmk]);
                            }
                            ImGui::End();

                            viz::draw_hypothesis(
                                // const da::hypothesis::Hypothesis &hypothesis,
                                hypo,
                                // const slam::Measurements<gtsam::Point3> &measurements,
                                timesteps[i - 1].measurements,
                                // const gtsam::NonlinearFactorGraph &graph,
                                slam_sys.hypothesisGraph(),
                                // const gtsam::Values &estimates,
                                slam_sys.hypothesisEstimates(),
                                // const gtsam::Key x_key,
                                slam_sys.latestPoseKey(),
                                // const gtsam::Pose3 &x_pose,
                                slam_sys.latestPose(),
                                // const double sigmas,
                                sigmas,
                                // const std::map<gtsam::Key, bool> lmk_cov_to_draw
                                lmk_to_draw_covar);
                        }
                        ImPlot::EndPlot();
                    }
                    ImGui::End();
                }

                viz::render();
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
            double sigmas = sqrt(da::chi2inv(ic_prob, 2));
            gtsam::Vector pose_prior_noise = Vector3(1e-6, 1e-6, 1e-8);
            pose_prior_noise = pose_prior_noise.array().sqrt().matrix(); // Calc sigmas from variances
            vector<slam::Timestep2D> timesteps = convert_into_timesteps(odomFactors2d, measFactors2d);
            slam::SLAM2D slam_sys{};
            slam::SLAM2D slam_sys_gt{};

            std::shared_ptr<da::DataAssociation<slam::Measurement2D>> data_asso, data_asso_gt;

            if (with_ground_truth)
            {
                double sigmas = sqrt(da::chi2inv(ic_prob, 2));
                data_asso = std::make_shared<da::ml::MaximumLikelihood2D>(sigmas, range_threshold);
                std::map<uint64_t, gtsam::Key> meas_lmk_assos = measurement_landmarks_associations(
                    measFactors2d,
                    timesteps);
                data_asso_gt = std::make_shared<da::gt::KnownDataAssociation2D>(meas_lmk_assos);
                slam_sys_gt.initialize(pose_prior_noise, data_asso_gt);
            }

            switch (conf.association_method)
            {
            case da::AssociationMethod::MaximumLikelihood:
            {
                double sigmas = sqrt(da::chi2inv(ic_prob, 2));
                data_asso = std::make_shared<da::ml::MaximumLikelihood2D>(sigmas, range_threshold);
                break;
            }
            case da::AssociationMethod::KnownDataAssociation:
            {
                std::map<uint64_t, gtsam::Key> meas_lmk_assos = measurement_landmarks_associations(
                    measFactors2d,
                    timesteps);
                data_asso = std::make_shared<da::gt::KnownDataAssociation2D>(meas_lmk_assos);
                break;
            }
            }

            slam_sys.initialize(pose_prior_noise, data_asso, optimization_method, marginals_factorization);

            int tot_timesteps = timesteps.size();

            int i = 0;
            int step = timesteps[i].step;
            while (viz::running() && i < tot_timesteps)
            {
                viz::new_frame();

                ImGui::Begin("Menu");

                viz::progress_bar(step, tot_timesteps);
                ImGui::Checkbox("Enable stepping", &enable_stepping);
                if (enable_stepping)
                {
                    ImGui::SameLine(0.0f, 100.0f);
                    next_timestep = ImGui::Button("Next timestep");
                }
                else
                {
                    next_timestep = true;
                }
                ImGui::Checkbox("Set step to increment to", &enable_step_limit);
                if (enable_step_limit)
                {
                    // ImGui::SameLine(0.0f, 100.0f);
                    ImGui::SetNextItemWidth(150.0f);
                    ImGui::InputInt("Step to increment to", &step_to_increment_to);
                }
                else
                {
                    step_to_increment_to = std::numeric_limits<int>::max();
                }

                ImGui::Checkbox("Draw factor graph", &draw_factor_graph);
                if (with_ground_truth)
                {
                    ImGui::Checkbox("Draw factor graph ground truth", &draw_factor_graph_ground_truth);
                }
                if (draw_factor_graph || (with_ground_truth && draw_factor_graph_ground_truth))
                {
                    ImGui::Checkbox("Enable factor graph window", &enable_factor_graph_window);
                    if (enable_factor_graph_window)
                    {
                        ImGui::SetNextItemWidth(150.0f);
                        ImGui::InputInt("\0", &factor_graph_window);
                        ImGui::SameLine();
                        ImGui::TextWrapped("Factor graph window (how many latest timesteps to draw)");
                    }
                }
                ImGui::Checkbox("Autofit plot", &autofit);
                ImGui::End(); // Menu

                if (next_timestep && step < step_to_increment_to)
                {
                    const slam::Timestep2D &timestep = timesteps[i];

                    step = timestep.step;

                    start_t = std::chrono::high_resolution_clock::now();

                    slam_sys.processTimestep(timestep);
                    if (with_ground_truth)
                    {
                        slam_sys_gt.processTimestep(timestep);
                    }

                    end_t = std::chrono::high_resolution_clock::now();
                    double duration = chrono::duration_cast<chrono::nanoseconds>(end_t - start_t).count() * 1e-9;
#ifdef LOGGING
                    avg_time = (timestep.step * avg_time + duration) / (timestep.step + 1.0);
                    cout << "Duration: " << duration << " seconds\n"
                         << "Average time one iteration: " << avg_time << " seconds\n";
#endif
#ifdef HEARTBEAT
                    cout << "Processed timestep " << timestep.step << ", " << double(timestep.step + 1) / tot_timesteps * 100.0 << "\% complete\n";
#endif
                    total_time += duration;
                    final_error = slam_sys.error();
                    estimates = slam_sys.currentEstimates();
                    if (enable_stepping)
                    {
                        next_timestep = false;
                    }
                    i++;
                }

                if (draw_factor_graph || (with_ground_truth && draw_factor_graph_ground_truth))
                {
                    int latest_timestep_to_draw;
                    if (enable_factor_graph_window)
                    {
                        latest_timestep_to_draw = i - factor_graph_window;
                    }
                    else
                    {
                        latest_timestep_to_draw = 0;
                    }
                    ImGui::Begin(factor_graph_title.c_str());
                    if (autofit)
                    {
                        ImPlot::SetNextAxesToFit();
                    }
                    if (ImPlot::BeginPlot("##factor graph", ImVec2(-1, -1)))
                    {
                        if (draw_factor_graph)
                        {
                            viz::draw_factor_graph(slam_sys.getGraph(), slam_sys.currentEstimates(), latest_timestep_to_draw);
                        }
                        if (with_ground_truth && draw_factor_graph_ground_truth)
                        {
                            viz::draw_factor_graph_ground_truth(slam_sys_gt.getGraph(), slam_sys_gt.currentEstimates(), latest_timestep_to_draw);
                        }
                        ImPlot::EndPlot();
                    }
                    ImGui::End(); // Factor graph
                }

                viz::render();
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
    catch (slam::IndeterminantLinearSystemExceptionWithGraphValues &indetErr)
    { // when run in terminal: tbb::captured_exception
        std::cout << "Optimization failed" << std::endl;
        std::cout << indetErr.what() << std::endl;
        std::cout << "Error occured when:\n"
                  << indetErr.when << "\n";

        const gtsam::NonlinearFactorGraph &graph = indetErr.graph;
        const gtsam::Values &values = indetErr.values;

        if (connected_graph(graph, values))
        {
            cout << "Connected graph!\n";
        }
        else
        {
            cout << "Not connected graph!\n";
        }

        bool autofit_plot_toggle = true;

        while (viz::running())
        {
            viz::new_frame();
            ImGui::Begin("Config");
            ImGui::Checkbox("Autofit plot toggle", &autofit_plot_toggle);
            ImGui::End();
            ImGui::Begin(factor_graph_title.c_str());
            if (autofit_plot_toggle)
            {
                ImPlot::SetNextAxesToFit();
            }
            if (ImPlot::BeginPlot("##factor graph", ImVec2(-1, -1)))
            {
                viz::draw_factor_graph(graph, values);

                gtsam::Point2 p;
                if (gtsam::symbolChr(indetErr.nearbyVariable()) == 'l')
                {
                    if (is3D)
                    {
                        const auto &l = values.at<gtsam::Point3>(indetErr.nearbyVariable());
                        p << l.x(), l.y();
                    }
                    else
                    {
                        p = values.at<gtsam::Point2>(indetErr.nearbyVariable());
                    }
                }
                else if (gtsam::symbolChr(indetErr.nearbyVariable()) == 'x')
                {
                    if (is3D)
                    {
                        const auto &x = values.at<gtsam::Pose3>(indetErr.nearbyVariable());
                        p << x.x(), x.y();
                    }
                    else
                    {
                        p = values.at<gtsam::Pose2>(indetErr.nearbyVariable()).translation();
                    }
                }
                viz::draw_circle(p);
                ImGui::Begin("Debugging");
                ImGui::Text("Problem variable: %s at\n[%f, %f]", gtsam::Symbol(indetErr.nearbyVariable()).string().c_str(), p(0), p(1));
                ImGui::TextWrapped("When did failure occur: %s", indetErr.when.c_str());
                ImGui::End();

                ImPlot::EndPlot();
            }
            ImGui::End();

            viz::render();
        }

        if (argc > 5)
        {
            string other_msg = "None";
            saveException(output_file, std::string("ExceptionML.txt"), indetErr.what(), other_msg);
        }
        NonlinearFactorGraph::shared_ptr graphNoKernel;
        Values::shared_ptr initial2;
        boost::tie(graphNoKernel, initial2) = readG2o(g2oFile, is3D);
        estimates = *initial2;
        caught_exception = true;
    }
    if (argc < 5)
    {
        if (caught_exception)
        {
            cout << "exception caught! printing odometry" << endl;
        }
        estimates.print("results");
    }
    else
    {
        if (!caught_exception)
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

    viz::shutdown();
}