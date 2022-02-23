#include <Eigen/Core>
#include <glog/logging.h>

#include <limits>
#include <iostream>

#include "data_association/DataAssociation.h"

int main(int argc, char **argv)
{
    google::InitGoogleLogging(argv[0]);
    google::ParseCommandLineFlags(&argc, &argv, true);
    google::InstallFailureSignalHandler();

    constexpr double inf = std::numeric_limits<double>::infinity();

    Eigen::MatrixXd A(7, 3);
    A << -5.69, 5.37, -inf,
        -inf, -3.8, 6.58,
        4.78, -inf, -inf,
        -inf, 5.36, -inf,
        -0.46, -inf, -inf,
        -inf, -0.52, -inf,
        -inf, -inf, -0.60;

    std::vector<int> assigned_landmarks = da::auction(A);
    for (int t = 0; t < assigned_landmarks.size(); t++) {
        //     for t, j in enumerate(assignments):
        // print(f"a({t+1}) = {j+1}")
        int j = assigned_landmarks[t];
        std::cout << "a(" << t + 1 << ") = " << j + 1 << "\n";
    }
}