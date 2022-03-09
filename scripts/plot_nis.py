import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2


if __name__ == "__main__":
    nis_log = open("./nis_log.txt").read().splitlines()
    dof = int(nis_log[0])
    N = len(nis_log[1:])
    nises = np.empty(N, dtype=float)
    num_assos = np.empty(N, dtype=int)
    for k, line in enumerate(nis_log[1:]):
        e, n = line.split()
        nises[k] = float(e)
        num_assos[k] = int(n)

    dofs = num_assos*dof
    alpha = 0.05 # 95% confidence interval
    confidence_interval = np.array([0.05, 0.95]).reshape(-1,1)
    lower_bounds, upper_bounds = chi2.ppf(confidence_interval, dofs)/N
    avg_nises = nises/N

    fig, ax = plt.subplots()

    ax.plot(lower_bounds, 'r--', label="Lower bound")
    ax.plot(upper_bounds, 'g--', label="Upper bound")
    ax.plot(avg_nises, label="Average NIS")


    num_inside = ((lower_bounds <= avg_nises) & (avg_nises <= upper_bounds)).sum()
    ratio_inside = num_inside/N

    ax.set_title(f"Consistency: {ratio_inside*100:.2f}% inside 95% confidence interval")
    ax.legend()

    plt.show()