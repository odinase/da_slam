#include "config/config.h"
#include <opencv2/core.hpp>
#include <iostream>

namespace config
{
    Config::Config(const char *filename)
    {
        cv::FileStorage yaml(filename, cv::FileStorage::READ);

        yaml["enable_stepping"] >> enable_stepping;
        yaml["draw_factor_graph"] >> draw_factor_graph;
        yaml["enable_step_limit"] >> enable_step_limit;
        yaml["step_to_increment_to"] >> step_to_increment_to;
        yaml["autofit"] >> autofit;

        std::cout << "Using config\n";
        std::cout << "enable_stepping: " << enable_stepping << "\n";
        std::cout << "draw_factor_graph: " << draw_factor_graph << "\n";
        std::cout << "enable_step_limit: " << enable_step_limit << "\n";
        std::cout << "step_to_increment_to: " << step_to_increment_to << "\n";
        std::cout << "autofit: " << autofit << "\n";
    }

} // namespace config
