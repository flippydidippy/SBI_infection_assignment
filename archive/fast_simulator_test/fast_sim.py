import numpy as np

def simulate_fast(beta, gamma, rho, N=200, p_edge=0.05, n_infected0=5, T=200, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    # Undirected adjacency matrix
    upper = rng.random((N, N)) < p_edge
    upper = np.triu(upper, 1)
    A = upper | upper.T   # bool matrix

    # States: 0=S, 1=I, 2=R
    state = np.zeros(N, dtype=np.int8)
    initial_infected = rng.choice(N, size=n_infected0, replace=False)
    state[initial_infected] = 1

    infected_fraction = np.zeros(T + 1, dtype=float)
    rewire_counts = np.zeros(T + 1, dtype=np.int64)
    infected_fraction[0] = np.mean(state == 1)

    all_nodes = np.arange(N)

    for t in range(1, T + 1):
        susceptible = (state == 0)
        infected = (state == 1)

        # Phase 1: infection
        # For each susceptible node, count infected neighbors
        infected_neighbor_count = A[:, infected].sum(axis=1)

        # Probability a susceptible node avoids infection from all infected neighbors:
        # (1 - beta)^k, so infection probability is 1 - (1 - beta)^k
        p_inf = 1.0 - np.power(1.0 - beta, infected_neighbor_count)
        new_infected = susceptible & (rng.random(N) < p_inf)
        state[new_infected] = 1

        # Phase 2: recovery
        infected = (state == 1)
        recover = infected & (rng.random(N) < gamma)
        state[recover] = 2

        # Phase 3: rewiring
        susceptible = (state == 0)
        infected = (state == 1)

        # S-I edges: rows are susceptible, cols are infected
        SI = A & susceptible[:, None] & infected[None, :]
        s_nodes, i_nodes = np.where(SI)

        rewire_count = 0
        for s_node, i_node in zip(s_nodes, i_nodes):
            if rng.random() >= rho:
                continue

            # Edge may already have been removed by an earlier rewiring
            if not A[s_node, i_node]:
                continue

            # Remove old edge
            A[s_node, i_node] = False
            A[i_node, s_node] = False

            # New partner cannot be self or existing neighbor
            candidates = all_nodes[~A[s_node]]
            candidates = candidates[candidates != s_node]

            if candidates.size > 0:
                new_partner = rng.choice(candidates)
                A[s_node, new_partner] = True
                A[new_partner, s_node] = True
                rewire_count += 1

        infected_fraction[t] = np.mean(state == 1)
        rewire_counts[t] = rewire_count

    degrees = A.sum(axis=1)
    degree_histogram = np.bincount(np.minimum(degrees, 30), minlength=31)

    return infected_fraction, rewire_counts, degree_histogram