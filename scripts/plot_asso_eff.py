import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    asso_effs = np.array([float(d) for d in open("./asso_eff.txt")])
    N = asso_effs.shape[0]
    asso_effs_avg = np.cumsum(asso_effs) / (1 + np.arange(N))
    plt.plot(asso_effs, label='association efficiency')
    plt.plot(asso_effs_avg, label='Running average association efficiency')

    tot_avg = asso_effs.mean()
    plt.axhline(tot_avg, color='green', lw=2, ls="--", label='Total average')

    plt.legend()
    plt.show()