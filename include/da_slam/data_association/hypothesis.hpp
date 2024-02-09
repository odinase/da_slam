#ifndef DA_SLAM_DATA_ASSOCIATION_HYPOTHESIS_HPP
#define DA_SLAM_DATA_ASSOCIATION_HYPOTHESIS_HPP

#include <gtsam/base/Matrix.h>
#include <gtsam/inference/Key.h>
#include <gtsam/nonlinear/Marginals.h>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <memory>
#include <optional>
#include <unordered_set>
#include <vector>

#include "da_slam/types.hpp"

namespace da_slam::data_association::hypothesis
{

struct Association
{
    explicit Association(int m);
    Association(int m, gtsam::Key l, const gtsam::Matrix& Hx, const gtsam::Matrix& Hl, const gtsam::Vector& error);
    Association(int m, gtsam::Key l);  // For ML
    using shared_ptr = std::shared_ptr<Association>;
    int measurement;
    std::optional<gtsam::Key> landmark;
    gtsam::Matrix Hx;
    gtsam::Matrix Hl;
    bool associated() const
    {
        return bool(landmark);
    }
    // double nis(const Eigen::VectorXd &z, const Eigen::VectorXd &zbar, const Marginals &S);
    gtsam::Vector error;

    bool operator<(const Association& rhs) const
    {
        return measurement < rhs.measurement;
    }

    bool operator>(const Association& rhs) const
    {
        return measurement > rhs.measurement;
    }

    bool operator==(const Association& rhs) const
    {
        bool same_measurement = measurement == rhs.measurement;
        bool both_unassociated = !associated() && !rhs.associated();
        bool both_associated = associated() && rhs.associated();
        bool associated_to_same_landmark = true;
        if (both_associated) {
            associated_to_same_landmark = *landmark == *rhs.landmark;
        }
        return same_measurement && (both_unassociated || both_associated) && associated_to_same_landmark;
    }

    bool operator<=(const Association& rhs) const
    {
        return (*this) < rhs || (*this) == rhs;
    }

    bool operator>=(const Association& rhs) const
    {
        return (*this) > rhs || (*this) == rhs;
    }
};

class Hypothesis
{
   private:
    double nis_;
    gtsam::FastVector<Association::shared_ptr> assos_;

   public:
    typedef std::shared_ptr<Hypothesis> shared_ptr;

    Hypothesis() = default;
    Hypothesis(const gtsam::FastVector<Association::shared_ptr>& associations, double nis)
    : nis_(nis), assos_(associations)
    {
    }
    int num_associations() const;
    int num_measurements() const;
    void set_nis(double nis)
    {
        nis_ = nis;
    }
    double get_nis() const
    {
        return nis_;
    }

    gtsam::KeyVector associated_landmarks() const;
    // Needed for min heap
    bool operator<(const Hypothesis& rhs) const;

    // Needed for min heap
    bool operator>(const Hypothesis& rhs) const;

    // Added for completion of comparision operators
    bool operator==(const Hypothesis& rhs) const;

    // Added for completion of comparision operators
    bool operator<=(const Hypothesis& rhs) const;

    // Added for completion of comparision operators
    bool operator>=(const Hypothesis& rhs) const;

    bool better_than(const Hypothesis& other) const;

    static Hypothesis empty_hypothesis();

    void extend(const Association::shared_ptr& a);

    Hypothesis extended(const Association::shared_ptr& a) const;

    const gtsam::FastVector<Association::shared_ptr>& associations() const
    {
        return assos_;
    }

    // Compute map that, for each measurement, check whether a hypothesis is equal to another
    std::map<int, bool> compare(const Hypothesis& other) const;

    gtsam::FastVector<std::pair<int, gtsam::Key>> measurement_landmark_associations() const;
    void fill_with_unassociated_measurements(int tot_num_measurements);
};

}  // namespace da_slam::data_association::hypothesis

#endif  // DA_SLAM_DATA_ASSOCIATION_HYPOTHESIS_HPP