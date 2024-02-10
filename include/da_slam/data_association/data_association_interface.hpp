#ifndef DATA_ASSOCIATION_H
#define DATA_ASSOCIATION_H

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
                                             const gtsam::FastVector<Measurement>& measurements) const = 0;

    virtual ~IDataAssociation() = default;
    IDataAssociation() = default;
    IDataAssociation(const IDataAssociation&) = delete;
    IDataAssociation(IDataAssociation&& rhs) noexcept = delete;
    IDataAssociation& operator=(const IDataAssociation& rhs) = delete;
    IDataAssociation& operator=(IDataAssociation&& rhs) noexcept = delete;
};

}  // namespace da_slam::data_association

std::ostream& operator<<(std::ostream& os, const da_slam::data_association::AssociationMethod& asso_method);

#endif  // DATA_ASSOCIATION_H