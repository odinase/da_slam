#include "da_slam/config.hpp"

#include <yaml-cpp/yaml.h>

#include <iostream>

namespace da_slam::config
{

namespace da = data_association;

Config::Config(const char* filename)
{
    const auto yaml = YAML::LoadFile(filename);

    enable_stepping = yaml["enable_stepping"].as<bool>();
    draw_factor_graph = yaml["draw_factor_graph"].as<bool>();
    enable_step_limit = yaml["enable_step_limit"].as<bool>();
    step_to_increment_to = yaml["step_to_increment_to"].as<int>();
    autofit = yaml["autofit"].as<bool>();

    draw_factor_graph_ground_truth = yaml["draw_factor_graph_ground_truth"].as<bool>();
    enable_factor_graph_window = yaml["enable_factor_graph_window"].as<bool>();
    factor_graph_window = yaml["factor_graph_window"].as<int>();

    association_method = yaml["association_method"].as<std::string>();

    with_ground_truth = yaml["with_ground_truth"].as<bool>();

    const auto optim = yaml["optimization_method"].as<int>();
    switch (optim) {
        case 0:
        case 1:
        {
            optimization_method = static_cast<slam::OptimizationMethod>(optim);
            break;
        }
        default:
        {
            std::cout << "Unknown vaule passed in, got " << optim << ", using GN\n";
            optimization_method = slam::OptimizationMethod::GaussNewton;
            break;
        }
    }

    const auto fact = yaml["marginals_factorization"].as<int>();
    switch (fact) {
        case 0:
        {
            marginals_factorization = gtsam::Marginals::CHOLESKY;
            break;
        }
        case 1:
        {
            marginals_factorization = gtsam::Marginals::QR;
            break;
        }
        default:
        {
            std::cout << "Unknown vaule passed in, got " << fact << ", using Cholesky\n";
            marginals_factorization = gtsam::Marginals::CHOLESKY;
            break;
        }
    }

    stop_at_association_timestep = yaml["stop_at_association_timestep"].as<bool>();
    draw_association_hypothesis = yaml["draw_association_hypothesis"].as<bool>();
    break_at_misassociation = yaml["break_at_misassociation"].as<bool>();
}

}  // namespace da_slam::config
