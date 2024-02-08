#include <Eigen/Core>

#include <limits>
#include <iostream>

#include "data_association/DataAssociation.h"

int main(int argc, char **argv)
{
    google::InitGoogleLogging(argv[0]);
    google::InstallFailureSignalHandler();

    Eigen::MatrixXd A(4, 5);
    A << 10, 19, 8, 15, 0,
         10, 18, 7, 17, 0,
         13, 16, 9, 14, 0,
         12, 19, 8, 18, 0;

    std::cout << "A:\n"
              << A << "\n";

    double cost = 0.0;
    // assigned_measurements is num measurements long, where each value corresponds to the index of the landmark associated with the corresponding measurement with the entry index. 
    std::vector<int> assigned_measurements = da::hungarian(A);
    for (int m = 0; m < assigned_measurements.size(); m++)
    {
        int l = assigned_measurements[m];
        std::cout << "measurement " << m << " associated with landmark " << l << "\n";
        cost += A(m, l);
    }
    std::cout << "\ncost: " << cost << "\n";
}