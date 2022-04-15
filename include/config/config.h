#pragma once

#include <string>
#include "data_association/DataAssociation.h"
#include "slam/slam.h"
#include <gtsam/nonlinear/Marginals.h>

namespace config {

struct Config {
    Config() = default;
    Config(const char* filename);
    Config(const std::string& filename) : Config(filename.c_str()) {}

    bool enable_stepping;
    bool draw_factor_graph;
    bool enable_step_limit;
    int step_to_increment_to;
    bool autofit;

    bool draw_factor_graph_ground_truth;
    bool enable_factor_graph_window;
    int factor_graph_window;

    da::AssociationMethod association_method;

    bool with_ground_truth;
    bool stop_at_association_timestep;

    slam::OptimizationMethod optimization_method;
    gtsam::Marginals::Factorization marginals_factorization;
};

} // namespace config