#ifndef DA_SLAM_CONFIG_CONFIG_HPP
#define DA_SLAM_CONFIG_CONFIG_HPP

#include <gtsam/nonlinear/Marginals.h>

#include <string>

#include "da_slam/data_association/data_association_interface.hpp"
#include "da_slam/slam/slam.hpp"

namespace da_slam::config
{

struct Config
{
    Config() = default;
    explicit Config(const char* filename);
    explicit Config(const std::string& filename) : Config(filename.c_str())
    {
    }

    bool enable_stepping;
    bool draw_factor_graph;
    bool enable_step_limit;
    int step_to_increment_to;
    bool autofit;

    bool draw_factor_graph_ground_truth;
    bool enable_factor_graph_window;
    int factor_graph_window;

    data_association::AssociationMethod association_method;

    bool with_ground_truth;
    bool stop_at_association_timestep;
    bool draw_association_hypothesis;

    slam::OptimizationMethod optimization_method;
    gtsam::Marginals::Factorization marginals_factorization;

    bool break_at_misassociation;
};

}  // namespace da_slam::config

#endif  // ~DA_SLAM_CONFIG_CONFIG_HPP