#include <unordered_set>
#include <algorithm>
#include <gtsam/geometry/Pose2.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/inference/Key.h>

#include "data_association/Hypothesis.h"

namespace da {

namespace hypothesis
{
    Association::Association(int m) : measurement(m), landmark(std::nullopt) {}
    Association::Association(int m, gtsam::Key l, const gtsam::Matrix &Hx, const gtsam::Matrix &Hl, const gtsam::Vector &error) : measurement(m), landmark(l), Hx(Hx), Hl(Hl), error(error) {}
    Association::Association(int m, gtsam::Key l) : measurement(m), landmark(l) {}

    int Hypothesis::num_associations() const
    {
        return std::count_if(assos_.cbegin(), assos_.cend(), [](const Association::shared_ptr &a)
                             { return a->associated(); });
    }

    int Hypothesis::num_measurements() const
    {
        return assos_.size();
    }

    gtsam::KeyVector Hypothesis::associated_landmarks() const
    {
        gtsam::KeyVector landmarks;
        for (const auto &a : assos_)
        {
            if (a->associated())
            {
                landmarks.push_back(*a->landmark);
            }
        }
        return landmarks;
    }
    // Needed for min heap
    bool Hypothesis::operator<(const Hypothesis &rhs) const
    {
        return num_associations() > rhs.num_associations() || (num_associations() == rhs.num_associations() && nis_ < rhs.nis_);
    }

    // Needed for min heap
    bool Hypothesis::operator>(const Hypothesis &rhs) const
    {
        return num_associations() < rhs.num_associations() || (num_associations() == rhs.num_associations() && nis_ > rhs.nis_);
    }

    // Added for completion of comparision operators
    bool Hypothesis::operator==(const Hypothesis &rhs) const
    {
        return num_associations() == rhs.num_associations() && nis_ == rhs.nis_;
    }

    // Added for completion of comparision operators
    bool Hypothesis::operator<=(const Hypothesis &rhs) const
    {
        return *this < rhs || *this == rhs;
    }

    // Added for completion of comparision operators
    bool Hypothesis::operator>=(const Hypothesis &rhs) const
    {
        return *this > rhs || *this == rhs;
    }

    bool Hypothesis::better_than(const Hypothesis &other) const
    {
        return *this < other;
    }

    Hypothesis Hypothesis::empty_hypothesis()
    {
        return Hypothesis{{}, std::numeric_limits<double>::infinity()};
    }

    void Hypothesis::extend(const Association::shared_ptr &a)
    {
        assos_.push_back(a);
    }

    Hypothesis Hypothesis::extended(const Association::shared_ptr &a) const
    {
        Hypothesis h(*this);
        h.extend(a);
        return h;
    }

    gtsam::FastVector<std::pair<int, gtsam::Key>> Hypothesis::measurement_landmark_associations() const
    {
        gtsam::FastVector<std::pair<int, gtsam::Key>> measurement_landmark_associations;
        for (const auto &a : assos_)
        {
            if (a->associated())
            {
                measurement_landmark_associations.push_back({a->measurement,
                                                             *a->landmark});
            }
        }

        return measurement_landmark_associations;
    }

    // Will be a pretty naive implementation, but w/e
    void Hypothesis::fill_with_unassociated_measurements(int tot_num_measurements)
    {
        std::vector<int> all_measurements;
        for (int i = 0; i < tot_num_measurements; i++)
        {
            all_measurements.push_back(i);
        }
        std::vector<int> meas_in_hypothesis;
        for (const auto &a : assos_)
        {
            meas_in_hypothesis.push_back(a->measurement);
        }

        std::vector<int> v(tot_num_measurements);
        std::vector<int>::iterator it;

        // Don't think we need to sort this...
        std::sort(all_measurements.begin(), all_measurements.end());
        std::sort(meas_in_hypothesis.begin(), meas_in_hypothesis.end());

        it = std::set_difference(all_measurements.begin(), all_measurements.end(),
                                 meas_in_hypothesis.begin(), meas_in_hypothesis.end(),
                                 v.begin());
        v.resize(it - v.begin());

        for (const auto &vv : v)
        {
            assos_.push_back(std::make_shared<Association>(vv));
        }
        std::sort(assos_.begin(), assos_.end(), [](const auto& lhs, const auto& rhs) {return (*lhs) < (*rhs);});
    }

    // Compute map that, for each measurement, check whether a hypothesis is equal to another
    std::map<int, bool> Hypothesis::compare(const Hypothesis& other) const {
        std::map<int, bool> equal_assos;
        for (const auto& asso : assos_) {
            int m = asso->measurement;
            const auto it = std::find_if(other.assos_.begin(), other.assos_.end(), [&](const auto& elem) { return elem->measurement == asso->measurement; });
            if (it != other.assos_.end()) {
                equal_assos[m] = *asso == **it;
            } else {
                equal_assos[m] = false;
            }
        }

        return equal_assos;
    }

} // namespace hypothesis
} // namespace da
