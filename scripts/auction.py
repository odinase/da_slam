import numpy as np
import matplotlib.pyplot as plt



def unassigned_customers_exist(customers):
    return (customers < 0).any() # We use -1 for unassigned


def auction(A, eps=0.001, N=10_000):
    m, n = A.shape
    unassigned_queue = np.arange(n)
    assigned_tracks = np.full(n, -1, dtype=int) # -1 indicates unassigned track
    prices = np.zeros(m)

    i = 0

    while unassigned_queue.size > 0:
        t_star = int(unassigned_queue[0])
        unassigned_queue = unassigned_queue[1:] # Poor man's pop

        # # This for loop probably not needed??
        # for k, rewards in zip(range(n), A.T):
        #     preffered_items[k] = int((rewards - prices).argmax())
        
        # i_star = int(preffered_items[t_star])
        i_star = (A.T[t_star] - prices).argmax()
        prev_owner, = np.where(assigned_tracks==i_star)
        assigned_tracks[t_star] = i_star
        if prev_owner.size > 0: # The item has a previous owner
            assert prev_owner.shape[0] == 1, f"multiple owners of same item, prev_owner = {prev_owner}"
            assigned_tracks[prev_owner] = -1
            unassigned_queue = np.append(unassigned_queue, prev_owner)

        values = np.delete(A.T[t_star] - prices, i_star)
        y = A[i_star, t_star] - values.max()
        prices[i_star] = prices[i_star] + y + eps

        i += 1

    return assigned_tracks


def calc_reward(problem_solution_pair):
    As, Ap = problem_solution_pair
    items = As
    customers = np.arange(As.shape[0])
    reward = Ap[items, customers].sum()

    return reward


def find_best_problem_solution_pair_idx(problem_solution_set):
    best_reward = -np.inf
    idx = 0
    for k, ps_pair in enumerate(problem_solution_set):
        curr_reward = calc_reward(ps_pair)
        if curr_reward > best_reward:
            best_reward = curr_reward
            idx = k

    return idx

def valid_solution(ps_pair):
    reward = calc_reward(ps_pair)
    return np.isfinite(reward)


def solution_problem_pair_exists(pair, L):
    As, Ap = pair
    for (Qs, Qp) in L:
        if np.allclose(As, Qs):
            return True

    return False


def murtys(A, N):
    m, n = A.shape
    As = auction(A)
    L = [(As, A)]
    R = []

    while len(L) > 0:
        k = find_best_problem_solution_pair_idx(L)
        Ms, Mp = L.pop(k)
        R.append((Ms, Mp))

        if len(R) == N:
            break

        P = Mp.copy() # Do we need copy here?
        i = Ms[0]

        locked_targets = [] # For keeping track of associations made so far
        item_idxs = np.arange(Mp.shape[0]) # For mapping between reduced-problem measurement index and original problem measurement index

        for t in range(n):
            # Step (a): Solve current problem by prohibiting first tracks original association. Will always be column 0
            P[i, 0] = -np.inf
            if not (~np.isfinite(P[:,0])).all():
                S = auction(P)
                if valid_solution((S, P)):
                    # The solution Qs will in general miss the removed track associations, append them here before storing 
                    Qs = np.append(locked_targets, item_idxs[S]).astype(int)

                    # Construct copy of original problem. All previous targets that are removed from current reduced problem will have correct association value here, we only need to change current target
                    Qp = Mp.copy()
                    org_i = item_idxs[i] # Look up what row in original problem 
                    Qp[org_i,t] = -np.inf

                    pair = (Qs, Qp)

                    if not solution_problem_pair_exists(pair, L):
                        L.append(pair)


            locked_targets.append(item_idxs[i])
            item_idxs = np.delete(item_idxs, i)

            P = np.delete(P[:,1:], i, axis=0) # Remove current target and its association
            if P.size == 0:
                break # If we have no more targets to associate, we are at the bottom
            # S = auction(P) # Rerun auction on reduced problem
            # i = S[0]
            i, = np.where(item_idxs==Ms[t+1])[0]

    return R


def compute_number_of_possible_assos(A):
    if A.shape[1] == 1:
        return np.isfinite(A).sum()
    
    s = 0
    A0 = A[:,0]
    valid_choices = np.isfinite(A0)
    idxs, = np.where(valid_choices)
    for i in idxs:
        Asub = np.delete(A[:,1:], i, axis=0)
        s += compute_number_of_possible_assos(Asub)

    return s


if __name__ == "__main__":
    # A = np.array([
    #     [  -5.69,    5.37, -np.inf],
    #     [-np.inf,    -3.8,    6.58],
    #     [   4.78, -np.inf, -np.inf],
    #     [-np.inf,    5.36, -np.inf],
    #     [  -0.46, -np.inf, -np.inf],
    #     [-np.inf,   -0.52, -np.inf],
    #     [-np.inf, -np.inf,   -0.60]
    # ])
    A = -np.array([
        [10, 19, 8, 15, 0],
        [10, 18, 7, 17, 0],
        [13, 16, 9, 14, 0],
        [12, 19, 8, 18, 0]
    ])

    assignments = auction(A)

    for t, j in enumerate(assignments):
        print(f"a({t+1}) = {j+1}")


    # s = compute_number_of_possible_assos(A)
    # print(f"number of possible assos: {s}")

    # N = 19

    # import time
    # start = time.time()
    # R = murtys(A, N)
    # stop = time.time()
    # t_us = (stop-start)*1e6
    # print(f"ran in {t_us:.2f} us")

    # rewards = np.empty(N)

    # for k, (assignments, problem) in enumerate(R):
    #     print(f"---------------\nassignement {k+1}")
    #     reward = calc_reward((assignments, problem))
    #     rewards[k] = reward
    #     print(f"reward: {reward}")
    #     for t, j in enumerate(assignments):
    #         print(f"a({t+1}) = {j+1}")

    # assert (np.abs(np.diff(rewards)) > 1e-6).all(), "some rewards are very similar"

    # plt.plot(np.arange(N)+1, rewards, 'o-')
    # plt.xticks(np.arange(N)+1)
    # plt.ylabel('Reward')
    # plt.xlabel('"Optimality" of solution')
    # plt.title(f"{s} possible hypothesises in total")
    # plt.show()