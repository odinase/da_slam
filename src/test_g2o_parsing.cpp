#include "slam/utils_g2o.h"

using namespace std;
using namespace gtsam;

void print_odom(const Pose2& odom) {
        cout << odom;
}

void print_odom(const Pose3& odom) {
    cout << odom.translation().transpose() << " " << odom.rotation().toQuaternion().coeffs().transpose();
}

void print_cov_mat(const Matrix& cov) {
    int rows = cov.rows();
    for (int i = 0; i < rows; i++) {
        for (int j = i; j < rows; j++) {
            cout << cov(i,j) << " ";
        }
    }
}

template<class POSE, class POINT>
void read_out_timesteps(const vector<slam::Timestep<POSE, POINT>>& timesteps) {
    size_t num_timesteps = timesteps.size();
    cout << "There are "<< num_timesteps << " num time steps\n";
    for (const auto& timestep: timesteps) {
        cout << "::::::::::::::::::::::::::::::::::::::::\n::::::::::::::::::::::::::::\n";
        cout << "Timestep " << timestep.step << "\n";
        cout << "Odom is:\n";
        print_odom(timestep.odom.odom);
        if (timestep.odom.noise) {
        Matrix odom_noise = timestep.odom.noise->sigmas().array().inverse().square().matrix().asDiagonal();
         cout << "\n\nwith noise\n\n";
         print_cov_mat(odom_noise);
         cout << "\n\n";
        }
        cout << "Measurements are:\n";
        int i = 1;
        for (const auto& measurement: timestep.measurements) {
            cout << "Measurement " << i++ << ":\n";
            Matrix noise = measurement.noise->sigmas().array().inverse().square().matrix().asDiagonal();
            cout << measurement.measurement << "\n\nwith noise\n\n";
            print_cov_mat(noise);
            cout << "\n\n"; 
        }
        cout << "\n";
    }
}


int main(int argc, char **argv)
{
    string g2oFile;
    bool is3D = false;
    if (argc > 1)
    {
        g2oFile = argv[1]; // input dataset filename
    }
    if (argc > 2)
    {
        is3D = atoi(argv[2]);
    }

    vector<boost::shared_ptr<PoseToPointFactor<Pose2, Point2>>> measFactors2d;
    vector<boost::shared_ptr<PoseToPointFactor<Pose3, Point3>>> measFactors3d;

    vector<boost::shared_ptr<BetweenFactor<Pose2>>> odomFactors2d;
    vector<boost::shared_ptr<BetweenFactor<Pose3>>> odomFactors3d;

    // reading file and creating factor graph
    NonlinearFactorGraph::shared_ptr graph;
    Values::shared_ptr initial;
    boost::tie(graph, initial) = readG2owithLmks(g2oFile, is3D, "none");
    auto [odomFactorIdx, measFactorIdx] = findFactors(odomFactors2d, odomFactors3d, measFactors2d, measFactors3d, graph);
    if (is3D)
    {
        vector<slam::Timestep3D> timesteps = convert_into_timesteps(odomFactors3d, measFactors3d);
        read_out_timesteps(timesteps);
    }
    else
    {
        vector<slam::Timestep2D> timesteps = convert_into_timesteps(odomFactors2d, measFactors2d);
        read_out_timesteps(timesteps);
    }
}