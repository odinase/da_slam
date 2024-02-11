#include "da_slam/config.hpp"

#include <yaml-cpp/yaml.h>

#include <iostream>
#include <magic_enum.hpp>

namespace da_slam::config
{

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

    association_method = magic_enum::enum_cast<data_association::AssociationMethod>(
                             yaml["association_method"].as<std::string>(), magic_enum::case_insensitive)
                             .value();

    with_ground_truth = yaml["with_ground_truth"].as<bool>();

    optimization_method = magic_enum::enum_cast<slam::OptimizationMethod>(yaml["optimization_method"].as<std::string>(),
                                                                          magic_enum::case_insensitive)
                              .value();

    marginals_factorization = magic_enum::enum_cast<gtsam::Marginals::Factorization>(
                                  yaml["marginals_factorization"].as<std::string>(), magic_enum::case_insensitive)
                                  .value();

    stop_at_association_timestep = yaml["stop_at_association_timestep"].as<bool>();
    draw_association_hypothesis = yaml["draw_association_hypothesis"].as<bool>();
    break_at_misassociation = yaml["break_at_misassociation"].as<bool>();
}

}  // namespace da_slam::config
