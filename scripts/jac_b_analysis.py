import numpy as np


if __name__ == "__main__":
    A = np.loadtxt("./A.txt")
    b = np.loadtxt("./b.txt")
    keys = [f for f in open("./keys.txt").read().split("\n") if not f == ""]
    print(keys)
    x = keys.index('x71')
    print(len(keys))

    print(f"A shape: {A.shape}")
    print(f"A rank: {np.linalg.matrix_rank(A)}")
    print(f"A condition number: {np.linalg.cond(A)}")
    s = np.linalg.svd(A, compute_uv=False)
    print(f"A min svd: {s.min()}, max: {s.max()}")

