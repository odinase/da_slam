import numpy as np
import gtsam
import matplotlib.pyplot as plt

from gtsam.symbol_shorthand import X, L

class PoseToPoint:
    def __init__(self, key1, key2, measured, model):
        self.key1 = key1
        self.key2 = key2
        self.model = model
        self.measured = measured

    def evaluateError(self, w_T_b, w_b, H1 = None, H2 = None):
        return w_T_b.transformTo(w_b, H1, H2) - self.measured


def draw_factor_graph(log_path, is3D):
    graph_filename = log_path + "/factor_graph.g2o"
    graph, estimates = gtsam.readG2o(graph_filename, is3D)

    if is3D:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
    else:
        fig, ax = plt.subplots()

    for f in range(graph.nrFactors()):
        factor = graph.at(f)
        print(type(factor))
        print(dir(factor))
        print(factor.keys())
        print()

    return fig, ax


if __name__ == "__main__":
    # for o in dir(gtsam.gtsam):
    #     print(o)

    noise = gtsam.noiseModel.Diagonal.Sigmas(
                 np.array([0.3, 0.3]))

    factor = PoseToPoint(X(0), L(0), gtsam.Point2(1, 1), noise)
    x = gtsam.Pose2(1.1, 1.1, np.pi/4)
    l = gtsam.Point2(2.0, 1.0)
    H1, H2 = np.empty((2, 3)), np.empty((2, 2))
    err = factor.evaluateError(x, l, H1, H2)
    path = "/home/mrg/prog/C++/da-slam/log/timestep_60"
    is3D = True
    draw_factor_graph(path, is3D)

    plt.show()