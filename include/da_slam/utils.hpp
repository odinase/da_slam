#ifndef DA_SLAM_UTILS_HPP
#define DA_SLAM_UTILS_HPP

#include <gtsam/inference/Symbol.h>
#include <boost/math/distributions.hpp>

namespace da_slam::utils
{

inline gtsam::Key pose_key(const uint64_t idx)
{
    return gtsam::symbol_shorthand::X(idx);
}

inline gtsam::Key lmk_key(const uint64_t idx)
{
    return gtsam::symbol_shorthand::L(idx);
}

inline double chi2inv(const double p, const uint32_t dim)
{
    boost::math::chi_squared dist(dim);
    return quantile(dist, p);
}

}  // namespace da_slam::utils

#endif  // DA_SLAM_UTILS_HPP