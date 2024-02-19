#include "da_slam/config.hpp"

#include <yaml-cpp/yaml.h>

#include <iostream>
#include <magic_enum.hpp>

namespace da_slam::config
{

Config::Config(const char* filename)
{
    const auto yaml = YAML::LoadFile(filename);

    // Data association

    const auto data_association_yaml = yaml["data_association"];

    association_method = magic_enum::enum_cast<data_association::AssociationMethod>(
                             data_association_yaml["method"].as<std::string>(), magic_enum::case_insensitive)
                             .value();
    assignment_solver = magic_enum::enum_cast<data_association::assignment_solvers::AssignmentSolver>(
                            data_association_yaml["assignment_solver"].as<std::string>(), magic_enum::case_insensitive)
                            .value();

    // Optimization

    const auto optimization_yaml = yaml["optimization"];

    optimization_method = magic_enum::enum_cast<slam::OptimizationMethod>(optimization_yaml["method"].as<std::string>(),
                                                                          magic_enum::case_insensitive)
                              .value();

    marginals_factorization =
        magic_enum::enum_cast<gtsam::Marginals::Factorization>(
            optimization_yaml["marginals_factorization"].as<std::string>(), magic_enum::case_insensitive)
            .value();

    // Visualization

    const auto visualization_yaml = yaml["visualization"];

    enable_stepping = visualization_yaml["enable_stepping"].as<bool>();
    draw_factor_graph = visualization_yaml["draw_factor_graph"].as<bool>();
    enable_step_limit = visualization_yaml["enable_step_limit"].as<bool>();
    step_to_increment_to = visualization_yaml["step_to_increment_to"].as<int>();
    autofit = visualization_yaml["autofit"].as<bool>();
    draw_factor_graph_ground_truth = visualization_yaml["draw_factor_graph_ground_truth"].as<bool>();
    enable_factor_graph_window = visualization_yaml["enable_factor_graph_window"].as<bool>();
    factor_graph_window = visualization_yaml["factor_graph_window"].as<int>();
    with_ground_truth = visualization_yaml["with_ground_truth"].as<bool>();
    stop_at_association_timestep = visualization_yaml["stop_at_association_timestep"].as<bool>();
    draw_association_hypothesis = visualization_yaml["draw_association_hypothesis"].as<bool>();
    break_at_misassociation = visualization_yaml["break_at_misassociation"].as<bool>();
}

}  // namespace da_slam::config
