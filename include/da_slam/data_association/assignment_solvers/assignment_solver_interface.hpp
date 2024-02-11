#ifndef DA_SLAM_DATA_ASSOCIATION_ASSIGNMENT_SOLVERS_ASSIGNMENT_SOLVER_INTERFACE_HPP
#define DA_SLAM_DATA_ASSOCIATION_ASSIGNMENT_SOLVERS_ASSIGNMENT_SOLVER_INTERFACE_HPP

#include <Eigen/Core>

namespace da_slam::data_association::assignment_solvers
{

enum class AssignmentSolver : uint8_t {
    AUCTION = 0,
    HUNGARIAN = 1,
};

class IAssignmentSolver
{
   public:
    virtual std::vector<int> solve(const Eigen::Ref<const Eigen::MatrixXd>& cost_matrix) const = 0;

    virtual ~IAssignmentSolver() = default;
    IAssignmentSolver() = default;
    IAssignmentSolver(const IAssignmentSolver&) = delete;
    IAssignmentSolver(IAssignmentSolver&& rhs) noexcept = delete;
    IAssignmentSolver& operator=(const IAssignmentSolver& rhs) = delete;
    IAssignmentSolver& operator=(IAssignmentSolver&& rhs) noexcept = delete;
};

}  // namespace da_slam::data_association::assignment_solvers

#endif  // DA_SLAM_DATA_ASSOCIATION_ASSIGNMENT_SOLVERS_ASSIGNMENT_SOLVER_INTERFACE_HPP