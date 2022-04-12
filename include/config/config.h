#pragma once

#include <string>
#include "data_association/DataAssociation.h"

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

    da::AssociationMethod association_method;
};

} // namespace config