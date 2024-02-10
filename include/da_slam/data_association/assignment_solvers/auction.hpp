#ifndef DA_SLAM_DATA_ASSOCIATION_ASSIGNMENT_SOLVERS_AUCTION_HPP
#define DA_SLAM_DATA_ASSOCIATION_ASSIGNMENT_SOLVERS_AUCTION_HPP

#include "da_slam/data_association/assignment_solvers/assignment_solver_interface.hpp"

namespace da_slam::data_association::assignment_solvers
{

struct AuctionParameters
{
    static constexpr double DEFAULT_EPSILON = 1e-3;
    static constexpr uint64_t DEFAULT_MAX_ITERATIONS = 10'000;

    double epsilon{};
    uint64_t max_iterations{};

    AuctionParameters();
    AuctionParameters(const double eps, const uint64_t max_iter);
};

class Auction final : public IAssignmentSolver
{
   public:
    explicit Auction(const AuctionParameters& params);
    std::vector<int> solve(const Eigen::Ref<const Eigen::MatrixXd>& cost_matrix) const override;

   private:
    double m_eps{};
    uint64_t m_max_iter{};
};

}  // namespace da_slam::data_association::assignment_solvers

#endif  // DA_SLAM_DATA_ASSOCIATION_ASSIGNMENT_SOLVERS_AUCTION_HPP