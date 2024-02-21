#ifndef DA_SLAM_FMT_HPP
#define DA_SLAM_FMT_HPP

#include <fmt/format.h>
#include <fmt/ostream.h>
#include <fmt/ranges.h>
#include <gtsam/inference/Symbol.h>

#include <Eigen/Core>
#include <iostream>

namespace fmt
{

template <typename Derived>
struct formatter<Eigen::MatrixBase<Derived>> : ostream_formatter
{
};

template <>
struct formatter<gtsam::Symbol> : ostream_formatter
{
};

}  // namespace fmt

#endif  // DA_SLAM_FMT_HPP
