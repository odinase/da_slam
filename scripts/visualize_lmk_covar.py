import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
from scipy.stats import chi2
import os
from auction import auction

def ellipse(mu, P, s, n):
    thetas = np.linspace(0, 2*np.pi, n)
    ell = mu + s * (la.cholesky(P).T @ np.array([np.cos(thetas), np.sin(thetas)])).T
    return ell


def plot_meas(ax, meas, text=None):
    ax.plot(*meas, 'rx')
    if not text is None:
        ax.text(*meas, text)

def plot_lmk(ax, lmk_mu, lmk_cov, s, text=None):
    ell = ellipse(lmk_mu, lmk_cov, s, 200)
    ax.plot(*ell.T, 'g')
    ax.plot(*lmk_mu, 'gx')
    if not text is None:
        ax.text(*lmk_mu, text)


def find_files_with_lmks_and_meas():
    files = []
    for filename in os.listdir('./logs'):
        filename = './logs/' + filename
        with open(filename) as f:
            data = [line.replace('\n', '') for line in f.readlines()]
            if any(s.startswith('l') for s in data):
                files.append(filename)

    return files


if __name__ == "__main__":
    s = 3.1 # std

    files = find_files_with_lmks_and_meas()

    for ff in files:
        print(f"Checking file {ff}")
        with open(ff, 'r') as file:
            data = [line.replace('\n', '') for line in file.readlines()]

        fig, ax = plt.subplots()

        m = {}
        l = []
        C = []

        for line in data:
            p = line.split()
            if len(p) == 0:
                continue
            if p[0].startswith('z'):
                meas = np.array([float(s) for s in p[1:]])
                plot_meas(ax, meas, p[0][1:])
                z = p[0]
                m[z] = []

            elif p[0].startswith('l'):
                mu = np.array([float(s) for s in p[1:3]])
                l = mu
                cov = np.array([float(s) for s in p[3:]]).reshape(2, 2)
                C = cov
                plot_lmk(ax, mu, cov, s, p[0][1:])

                m[z].append((l, C))

            elif p[0].startswith('c'): # Cost matrix
                rows = int(p[1])
                cols = int(p[2])
                cost_matrix = np.array([float(s) for s in p[3:]]).reshape(rows, cols)
                print(f"rows: {rows}\ncols: {cols}\n{cost_matrix}")
                meas_rows = rows - cols
                reduced_mat = cost_matrix[:meas_rows]
                finite_cols = np.any(np.isfinite(reduced_mat), axis=0)
                lmk_map, = np.where(finite_cols)
                for k, lmk in enumerate(lmk_map):
                    print(f"lmk {lmk} corresponds to {k}")
                reduced_matrix = cost_matrix[:, finite_cols]
                finite_rows = np.any(np.isfinite(reduced_matrix), axis=1)
                reduced_mat = reduced_matrix[finite_rows, :]
                print(f"reduced_mat:\n{reduced_mat}")
                if reduced_mat.size > 0:
                    solution = auction(reduced_mat)
                    print(f"solution to auction:\n{solution}")
                    invalid_asso = (solution > (meas_rows - 1)) | (solution == -1)
                    valid_asso = ~invalid_asso
                    print(f"without nonassos: {solution[valid_asso]}")
                    
                solution_big = auction(cost_matrix)
                print(f"solution to auction, big matrix:\n{solution_big}")
                invalid_asso = (solution_big > (meas_rows - 1)) | (solution_big == -1)
                valid_asso = ~invalid_asso
                print(f"without nonassos: {solution_big[valid_asso]}")

            # print(f"norm: {np.linalg.norm(l - m)}")
            # nis = (m - l)@np.linalg.solve(C, m - l)
            # print(f"nis: {nis} < {s**2}: {nis < s**2}")

        timestep = os.path.basename(ff)[:9][:-4]
        ax.set_title(f"Timestep {timestep}")

        plt.show()