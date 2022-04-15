#include "config/config.h"
#include <opencv2/core.hpp>
#include <iostream>

// Stolen from https://stackoverflow.com/questions/62303440/opencv-yaml-parser-does-not-recognize-true-false-values
namespace cv
{
    // Define a new bool reader in order to accept "true/false"-like values.
    void read_bool(const cv::FileNode &node, bool &value, const bool &default_value)
    {
        std::string s(static_cast<std::string>(node));
        if (s == "y" || s == "Y" || s == "yes" || s == "Yes" || s == "YES" || s == "true" || s == "True" || s == "TRUE" || s == "on" || s == "On" || s == "ON")
        {
            value = true;
            return;
        }
        if (s == "n" || s == "N" || s == "no" || s == "No" || s == "NO" || s == "false" || s == "False" || s == "FALSE" || s == "off" || s == "Off" || s == "OFF")
        {
            value = false;
            return;
        }
        value = static_cast<int>(node);
    }
    // Specialize cv::operator>> for bool.
    template <>
    inline void operator>>(const cv::FileNode &n, bool &value)
    {
        read_bool(n, value, false);
    }
} // cv

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

        yaml["draw_factor_graph_ground_truth"] >> draw_factor_graph_ground_truth;
        yaml["enable_factor_graph_window"] >> enable_factor_graph_window;
        yaml["factor_graph_window"] >> factor_graph_window;

        int asso_method;
        yaml["association_method"] >> asso_method;
        switch (asso_method)
        {
        case 0:
        case 1:
        {
            association_method = static_cast<da::AssociationMethod>(asso_method);
            break;
        }
        default:
        {
            std::cout << "Unknown vaule passed in, got " << asso_method << ", using ML\n";
            association_method = da::AssociationMethod::MaximumLikelihood;
            break;
        }
        }

        yaml["with_ground_truth"] >> with_ground_truth;

        int optim;
        yaml["optimization_method"] >> optim;
        switch (optim)
        {
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

        int fact;
        yaml["marginals_factorization"] >> fact;
        switch (fact)
        {
        case 0: {
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

        yaml["stop_at_association_timestep"] >> stop_at_association_timestep;
        yaml["draw_association_hypothesis"] >> draw_association_hypothesis;
    }

} // namespace config
