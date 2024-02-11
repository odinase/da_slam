#ifndef DA_SLAM_DATA_ASSOCIATION_ASSIGNMENT_SOLVERS_HUNGARIAN_HPP
#define DA_SLAM_DATA_ASSOCIATION_ASSIGNMENT_SOLVERS_HUNGARIAN_HPP

#include "da_slam/data_association/assignment_solvers/assignment_solver_interface.hpp"

namespace da_slam::data_association::assignment_solvers
{

class Hungarian final : public IAssignmentSolver
{
   public:
    std::vector<int> solve(const Eigen::Ref<const Eigen::MatrixXd>& cost_matrix) const override;
};

}  // namespace da_slam::data_association::assignment_solvers

#endif  // DA_SLAM_DATA_ASSOCIATION_ASSIGNMENT_SOLVERS_HUNGARIAN_HPP