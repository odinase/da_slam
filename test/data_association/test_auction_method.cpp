#include <gtest/gtest.h>

#include <Eigen/Core>
#include <limits>
#include <iostream>

#include "data_association/DataAssociation.h"

TEST(DataAssociationTests, TestAuctionMethod) {
    constexpr double inf = std::numeric_limits<double>::infinity();

        Eigen::MatrixXd A(4, 5);
    A << 10, 19, 8, 15, 0,
        10, 18, 7, 17, 0 ,
        13, 16, 9, 14, 0,
        12, 19, 8, 18, 0;

    A = -A;

    std::cout << "A:\n" << A << "\n";

double cost = 0.0;
    std::vector<int> assigned_landmarks = da::auction(A);
    for (int t = 0; t < assigned_landmarks.size(); t++) {
        int j = assigned_landmarks[t];
        std::cout << "" << j << "," << t << "\t";
        cost += A(j, t);
    }
    std::cout << "\ncost: " << -cost << "\n";
}