#ifndef DATA_ASSOCIATION_H
#define DATA_ASSOCIATION_H

#include <boost/math/distributions.hpp>
#include <cmath>
#include <deque>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <utility>

#include "da_slam/data_association/hypothesis.hpp"
#include "da_slam/types.hpp"
#include "gtsam/nonlinear/Marginals.h"

namespace da_slam::data_association
{

enum class AssociationMethod : uint8_t {
    MAXIMUM_LIKELIHOOD = 0,
    KNOWN_DATA_ASSOCIATION = 1,
};

template <typename Measurement>
class IDataAssociation
{
   public:
    virtual hypothesis::Hypothesis associate(const gtsam::Values& estimates, const gtsam::Marginals& marginals,
                                             const gtsam::FastVector<Measurement>& measurements) = 0;

    virtual ~IDataAssociation() = default;
    IDataAssociation() = default;
    IDataAssociation(const IDataAssociation&) = delete;
    IDataAssociation(IDataAssociation&& rhs) noexcept = delete;
    IDataAssociation& operator=(const IDataAssociation& rhs) = delete;
    IDataAssociation& operator=(IDataAssociation&& rhs) noexcept = delete;
};

double chi2inv(double p, unsigned int dim);
std::vector<int> auction(const Eigen::MatrixXd& problem, double eps = 1e-3, uint64_t max_iterations = 10'000);
std::vector<int> hungarian(const Eigen::MatrixXd& cost_matrix);

}  // namespace da_slam::data_association

std::ostream& operator<<(std::ostream& os, const da_slam::data_association::AssociationMethod& asso_method);

#endif  // DATA_ASSOCIATION_H