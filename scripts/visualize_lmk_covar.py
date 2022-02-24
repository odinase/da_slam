import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
from scipy.stats import chi2
import os

def ellipse(mu, P, s, n):
    thetas = np.linspace(0, 2*np.pi, n)
    ell = mu + s * (la.cholesky(P).T @ np.array([np.cos(thetas), np.sin(thetas)])).T
    return ell


def plot_meas(ax, meas):
    ax.plot(*meas, 'rx')

def plot_lmk(ax, lmk_mu, lmk_cov, s):
    ell = ellipse(lmk_mu, lmk_cov, s, 200)
    ax.plot(*ell.T, 'g')
    ax.plot(*lmk_mu, 'gx')


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

        m = None
        l = None
        C = None

        for line in data:
            p = line.split()
            if len(p) == 0:
                continue
            if p[0].startswith('z'):
                meas = np.array([float(s) for s in p[1:]])
                m = meas
                plot_meas(ax, meas)

            elif p[0].startswith('l'):
                mu = np.array([float(s) for s in p[1:3]])
                l = mu
                cov = np.array([float(s) for s in p[3:]]).reshape(2, 2)
                C = cov
                plot_lmk(ax, mu, cov, s)

        print(f"norm: {np.linalg.norm(l - m)}")
        nis = (m - l)@np.linalg.solve(C, m - l)
        print(f"nis: {nis} < {s**2}: {nis < s**2}")

        plt.show()