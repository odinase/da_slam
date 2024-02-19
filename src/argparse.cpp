#include "da_slam/argparse.hpp"

namespace da_slam::argparse
{

constexpr auto PROGRAM_NAME = "da_slam";

ParsedArgs parse_args(const int argc, const char* argv[])
{
    ::argparse::ArgumentParser argparser{PROGRAM_NAME};

    argparser.add_argument("dataset_path").help("Path to dataset to run through. Must be g2o file.");

    argparser
        .add_argument("is3D")                                                                         //
        .help("To indicate whether to treat the input dataset as a 2d (false) or 3D (true) dataset")  //
        .default_value(0)
        .scan<'i', int>();

    argparser
        .add_argument("ic_prob")                                                                                    //
        .help("individual compatibility probability to use for maximum likelihood. Determines the chi2 thresold.")  //
        .default_value(0.99)
        .scan<'g', double>();

    argparser
        .add_argument("range_threshold")  //
        .help(
            "Heuristic distance to determine whether a landmark and measurement can be associated with each other.")  //
        .default_value(1e9)
        .scan<'g', double>();

    argparser
        .add_argument("output_file")  //
        .help("File path to where optimization errors are logged.");

    argparser.parse_args(argc, argv);

    return ParsedArgs{argparser.get<std::string>("dataset_path"), static_cast<bool>(argparser.get<int>("is3D")),
                      argparser.get<double>("ic_prob"), argparser.get<double>("range_threshold"),
                      argparser.get<std::string>("output_file")};
}

}  // namespace da_slam::argparse