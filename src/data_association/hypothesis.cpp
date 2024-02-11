#include "da_slam/data_association/hypothesis.hpp"

#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/inference/Key.h>

#include <algorithm>
#include <range/v3/all.hpp>
#include <unordered_set>

namespace rng = ranges;
namespace rngv = ranges::views;

namespace da_slam::data_association::hypothesis
{

Association::Association(int m) : measurement(m), landmark(std::nullopt)
{
}
Association::Association(int m, gtsam::Key l, const gtsam::Matrix& Hx, const gtsam::Matrix& Hl,
                         const gtsam::Vector& error)
: measurement(m), landmark(l), Hx(Hx), Hl(Hl), error(error)
{
}
Association::Association(int m, gtsam::Key l) : measurement(m), landmark(l)
{
}

int Hypothesis::num_associations() const
{
    return std::count_if(m_assos.cbegin(), m_assos.cend(),
                         [](const Association::shared_ptr& a) { return a->associated(); });
}

int Hypothesis::num_measurements() const
{
    return m_assos.size();
}

gtsam::KeyVector Hypothesis::associated_landmarks() const
{
    return m_assos                                                         //
           | rngv::filter([](auto&& asso) { return asso->associated(); })  //
           | rngv::transform([](auto&& asso) { return *asso->landmark; })  //
           | rng::to<gtsam::KeyVector>();
}
// Needed for min heap
bool Hypothesis::operator<(const Hypothesis& rhs) const
{
    return num_associations() > rhs.num_associations() ||
           (num_associations() == rhs.num_associations() && m_nis < rhs.m_nis);
}

// Needed for min heap
bool Hypothesis::operator>(const Hypothesis& rhs) const
{
    return num_associations() < rhs.num_associations() ||
           (num_associations() == rhs.num_associations() && m_nis > rhs.m_nis);
}

// Added for completion of comparision operators
bool Hypothesis::operator==(const Hypothesis& rhs) const
{
    return num_associations() == rhs.num_associations() && m_nis == rhs.m_nis;
}

// Added for completion of comparision operators
bool Hypothesis::operator<=(const Hypothesis& rhs) const
{
    return *this < rhs || *this == rhs;
}

// Added for completion of comparision operators
bool Hypothesis::operator>=(const Hypothesis& rhs) const
{
    return *this > rhs || *this == rhs;
}

bool Hypothesis::better_than(const Hypothesis& other) const
{
    return *this < other;
}

Hypothesis Hypothesis::empty_hypothesis()
{
    return Hypothesis{{}, std::numeric_limits<double>::infinity()};
}

void Hypothesis::extend(const Association::shared_ptr& a)
{
    m_assos.push_back(a);
}

Hypothesis Hypothesis::extended(const Association::shared_ptr& a) const
{
    Hypothesis h(*this);
    h.extend(a);
    return h;
}

gtsam::FastVector<std::pair<int, gtsam::Key>> Hypothesis::measurement_landmark_associations() const
{
    gtsam::FastVector<std::pair<int, gtsam::Key>> measurement_landmark_associations;
    for (const auto& a : m_assos) {
        if (a->associated()) {
            measurement_landmark_associations.push_back({a->measurement, *a->landmark});
        }
    }

    return measurement_landmark_associations;
}

// Will be a pretty naive implementation, but w/e
void Hypothesis::fill_with_unassociated_measurements(int tot_num_measurements)
{
    std::vector<int> all_measurements;
    for (int i = 0; i < tot_num_measurements; i++) {
        all_measurements.push_back(i);
    }

    std::vector<int> meas_in_hypothesis;
    for (const auto& a : m_assos) {
        meas_in_hypothesis.push_back(a->measurement);
    }

    std::vector<int> v(tot_num_measurements);
    std::vector<int>::iterator it;

    // Don't think we need to sort this...
    std::sort(all_measurements.begin(), all_measurements.end());
    std::sort(meas_in_hypothesis.begin(), meas_in_hypothesis.end());

    it = std::set_difference(all_measurements.begin(), all_measurements.end(), meas_in_hypothesis.begin(),
                             meas_in_hypothesis.end(), v.begin());
    v.resize(it - v.begin());

    for (const auto& vv : v) {
        m_assos.push_back(std::make_shared<Association>(vv));
    }
    std::sort(m_assos.begin(), m_assos.end(), [](const auto& lhs, const auto& rhs) { return (*lhs) < (*rhs); });
}

// Compute map that, for each measurement, check whether a hypothesis is equal to another
std::map<int, bool> Hypothesis::compare(const Hypothesis& other) const
{
    std::map<int, bool> equal_assos;
    for (const auto& asso : m_assos) {
        int m = asso->measurement;
        const auto it = std::find_if(other.m_assos.begin(), other.m_assos.end(),
                                     [&](const auto& elem) { return elem->measurement == asso->measurement; });
        if (it != other.m_assos.end()) {
            equal_assos[m] = *asso == **it;
        }
        else {
            equal_assos[m] = false;
        }
    }

    return equal_assos;
}

}  // namespace da_slam::data_association::hypothesis