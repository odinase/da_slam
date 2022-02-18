#pragma once

#include "data_association/Hypothesis.h"

namespace data_association {

class DataAssociation {
    public:
    virtual double individual_compatability(const hypothesis::Association &a) const;
    virtual hypothesis::Hypothesis associate() const = 0;
    virtual ~DataAssociation();
};

} // namespace data_association