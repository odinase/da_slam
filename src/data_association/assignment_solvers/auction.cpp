#include "da_slam/data_association/assignment_solvers/auction.hpp"

#include <spdlog/spdlog.h>

#include <deque>
#include <range/v3/all.hpp>

namespace da_slam::data_association::assignment_solvers
{

std::vector<int> Auction::solve(const Eigen::Ref<const Eigen::MatrixXd>& cost_matrix) const
{
    const auto m = cost_matrix.rows();
    const auto n = cost_matrix.cols();

    spdlog::info("Starting auction with cost_matrix size ({}, {})", m, n);

    std::deque<int> unassigned_queue;
    std::vector<int> assigned_landmarks;

    // Initilize
    for (int i = 0; i < n; i++) {
        unassigned_queue.push_back(i);
        assigned_landmarks.push_back(-1);
    }

    // Use Eigen vector for convenience below
    Eigen::VectorXd prices(m);
    for (int i = 0; i < m; i++) {
        prices(i) = 0;
    }

    uint64_t curr_iter = 0;

    while (!unassigned_queue.empty() && curr_iter < m_max_iter) {
        const auto l_star = unassigned_queue.front();
        unassigned_queue.pop_front();

        if (curr_iter > m_max_iter) {
            break;
        }
        Eigen::MatrixXd::Index i_star;
        const auto val_max = (cost_matrix.col(l_star) - prices).maxCoeff(&i_star);

        auto prev_owner = std::find(assigned_landmarks.begin(), assigned_landmarks.end(), i_star);
        assigned_landmarks[l_star] = i_star;

        if (prev_owner != assigned_landmarks.end()) {
            // The item has a previous owner
            *prev_owner = -1;
            const auto pos = std::distance(assigned_landmarks.begin(), prev_owner);
            unassigned_queue.push_back(static_cast<int>(pos));
        }

        double y = cost_matrix(i_star, l_star) - val_max;
        prices(i_star) += y + m_eps;
        curr_iter++;
    }

    if (curr_iter >= m_max_iter) {
        spdlog::error("Auction terminated early!");
    }
    else {
        spdlog::info("Auction terminated successfully after {} iterations!", curr_iter);
    }

    spdlog::info("Solution from auction:");
    for (auto&& [lmk_idx, meas_idx] : assigned_landmarks | ranges::views::enumerate) {
        spdlog::info("Landmark {} with measurement {}", lmk_idx, meas_idx);
    }

    return assigned_landmarks;
}

}  // namespace da_slam::data_association::assignment_solvers