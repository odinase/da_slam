import numpy as np
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as Rot
from typing import List
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from typing import Optional
import scipy.linalg as la
from scipy.stats import chi2


# TODO: Make this adapt to 2D and 3D
def x2pR(x):
    """
    Transform state x = [p, q] to tuple (p, R)
    """
    assert len(x.shape) == 1 and x.shape[0] == 3 + 4, f"x must be shape (7,), is now {x.shape}"
    p = x[:3]
    q = x[3:][[1, 2, 3, 0]] # Scipy needs real part last
    R = Rot.from_quat(q).as_matrix()

    return p, R


@dataclass
class Pose:
    position: np.ndarray
    rotation: np.ndarray
    id: int

    def __init__(self, x: np.ndarray, id: int):
        p, R = x2pR(x)
        self.position = p
        self.rotation = R
        self.id = id


@dataclass
class Landmark:
    position: np.ndarray
    id: int


@dataclass
class Odometry:
    translation: np.ndarray
    rotation: np.ndarray
    information_matrix: np.ndarray
    from_id: int
    to_id: int

    def __init__(self, dx: np.ndarray, inf_mat_triu: np.ndarray, from_id: int, to_id: int) -> None:
        dt, dR = x2pR(dx)
        self.translation = dt
        self.rotation = dR

        dim = 3 if dx.shape[0] == 3 else 6
        self.information_matrix = np.empty((dim, dim))
        upper = np.triu_indices_from(self.information_matrix)
        self.information_matrix[upper] = inf_mat_triu
        self.information_matrix.T[upper] = inf_mat_triu

        assert to_id == from_id + 1, f"Odometry must be between consecutive poses, got from {from_id} to {to_id}"
        self.from_id = from_id
        self.to_id = to_id


@dataclass
class Measurement:
    translation: np.ndarray
    information_matrix: np.ndarray
    from_id: int
    to_id: int

    def __init__(self, dt: np.ndarray, inf_mat_triu: np.ndarray, from_id: int, to_id: int) -> None:
        self.translation = dt

        dim = dt.shape[0]
        self.information_matrix = np.empty((dim, dim))
        upper = np.triu_indices_from(self.information_matrix)
        self.information_matrix[upper] = inf_mat_triu
        self.information_matrix.T[upper] = inf_mat_triu

        self.from_id = from_id
        self.to_id = to_id


@dataclass
class FactorGraph:
    poses: List[Pose]
    lmks: List[Landmark]
    odometry: List[Odometry]
    measurements: List[Measurement]

    def find_pose(self, id):
        for pose in self.poses:
            if pose.id == id:
                return pose

    def find_lmk(self, id):
        for lmk in self.lmks:
            if lmk.id == id:
                return lmk



def readG2o(filename):
    with open(filename) as f:
        poses = []
        lmks = []
        odometry = []
        measurements = []

        for line in map(lambda s: s.strip().split(), f):
            ## 2D graph

            if line[0] == "VERTEX_SE2" or line[0] == "VERTEX2":
                pass
            elif line[0] == "VERTEX_XY":
                pass
            elif line[0] == "EDGE_SE2" or line[0] == "EDGE2" or line[0] == "EDGE_SE2_SWITCHABLE":
                pass
            elif line[0] == "EDGE_SE2_XY":
                pass
            ## 3D graph

            elif line[0] == "VERTEX_SE3:QUAT":
                id = int(line[1])
                x = np.array(line[2:], dtype=float)
                poses.append(Pose(x, id))

            elif line[0] == "VERTEX_TRACKXYZ":
                id = int(line[1])
                p = np.array(line[2:], dtype=float)
                lmks.append(Landmark(p, id))

            elif line[0] == "EDGE_SE3:QUAT":
                from_id = int(line[1])
                to_id = int(line[2])
                dx = np.array(line[3:10], dtype=float)
                inf_mat_triu = np.array(line[10:], dtype=float)
                odometry.append(Odometry(dx, inf_mat_triu, from_id, to_id))

            elif line[0] == "EDGE_SE3_XYZ":
                from_id = int(line[1])
                to_id = int(line[2])
                dt = np.array(line[3:6], dtype=float)
                inf_mat_triu = np.array(line[6:], dtype=float)
                measurements.append(Measurement(dt, inf_mat_triu, from_id, to_id))

        poses.sort(key=lambda p: p.id)
        lmks.sort(key=lambda l: l.id)
        odometry.sort(key=lambda odom: odom.from_id)
        measurements.sort(key=lambda m: m.from_id)

        return FactorGraph(poses, lmks, odometry, measurements)

@dataclass
class Association:
    m: str
    meas: np.ndarray
    l: Optional[str] = None
    S: Optional[np.ndarray] = None

    @property
    def m_idx(self):
        return int(self.m[1:])

    @property
    def l_idx(self):
        return int(self.l[1:])

    def associated(self):
        return self.l is not None
 
def readHypothesis(filename):
    assos = []
    with open(filename) as f:
        for line in map(lambda s: s.strip().split(), f):
            if line[0] == '2d':
                pass
            elif line[0] == '3d':
                meas, x, y, z = line[1:5]
                asso = Association(meas, np.array([x, y, z], dtype=float))
                if len(line[5:]) > 0: # We have association to add
                    asso.l = line[5]
                    rows, cols = line[6:8]
                    rows = int(rows)
                    cols = int(cols)
                    asso.S = np.array(line[8:], dtype=float).reshape(rows, cols)
            else:
                print(f"Unknown measurement format! Expected '2d' or '3d', got '{line[0]}'")
                continue

            assos.append(asso)

    return assos

def ellipse3d(l, S, s = 5.0, n = 200):
    L = la.cholesky(S).T

    # Set of all spherical angles:
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, n)

    # Cartesian coordinates that correspond to the spherical angles:
    # (this is the equation of an ellipsoid):
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    R = np.stack((x, y, z))

    R = R.transpose((1, 2, 0)).reshape(-1, 3).T

    ell = l + (s * L @ R).T

    ell = ell.reshape(n, n, 3).transpose((2, 0, 1))
    return ell

def plot_hypothesis(assos: List[Association], assos_fg: FactorGraph):
    fig3d = plt.figure()  # Square figure
    ax3d = fig3d.add_subplot(111, projection='3d')
    sigma = np.sqrt(chi2.ppf(0.9, 3))
    for asso in assos:
        line = np.zeros((2, 3))
        meas = asso.meas
        line[1] = meas
        ax3d.plot(*line)
        if asso.associated():
            l = fg.find_lmk(asso.l_idx).position
            S = asso.S
            x, y, z = ellipse3d(meas, S, sigma)
            ax3d.plot_surface(x, y, z,  rstride=4, cstride=4, color='b', alpha=0.2)


if __name__ == "__main__":
    filename = "./log/timestep_60/factor_graph.g2o"
    fg = readG2o(filename)
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # Draw all poses
    pose_positions = np.array([p.position for p in fg.poses])
    ax.plot(*pose_positions.T, 'bo')

    # Draw all landmarks
    lmk_positions = np.array([l.position for l in fg.lmks])
    ax.plot(*lmk_positions.T, 'go')

    # Draw odometry factors
    for odom in fg.odometry:
        def find_pose(id):
            for pose in fg.poses:
                if pose.id == id:
                    return pose

        from_pose = find_pose(odom.from_id)
        to_pose = find_pose(odom.to_id)

        line = np.array([from_pose.position, to_pose.position])

        ax.plot(*line.T, 'r')


    # Draw measurement factors
    for measurement in fg.measurements:
        def find_pose(id):
            for pose in fg.poses:
                if pose.id == id:
                    return pose

        def find_lmk(id):
            for lmk in fg.lmks:
                if lmk.id == id:
                    return lmk

        
        from_pose = find_pose(measurement.from_id)
        to_lmk = find_lmk(measurement.to_id)

        line = np.array([from_pose.position, to_lmk.position])

        ax.plot(*line.T, 'm', alpha=0.3)

    filename_asso = "./log/timestep_60/association_hypothesis.txt"
    filename_asso_graph = "./log/timestep_60/association_graph.g2o"

    assos = readHypothesis(filename_asso)
    assos_fg = readG2o(filename_asso_graph)

    plot_hypothesis(assos, assos_fg)

    plt.show()
