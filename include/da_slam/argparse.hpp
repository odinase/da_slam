#ifndef DA_SLAM_ARGPARSE_HPP
#define DA_SLAM_ARGPARSE_HPP

#include <argparse/argparse.hpp>

namespace da_slam::argparse
{

struct ParsedArgs
{
    std::string dataset_path{};
    bool is_3d{};
    double ic_prob{};
    double range_threshold{};
    std::string output_file{};
};

ParsedArgs parse_args(const int argc, const char* argv[]);

}  // namespace da_slam::argparse

#endif  // DA_SLAM_ARGPARSE_HPP