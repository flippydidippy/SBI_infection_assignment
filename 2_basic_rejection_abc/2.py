import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# FAST SIMULATOR
# ============================================================

def simulate_fast(beta, gamma, rho, N=200, p_edge=0.05, n_infected0=5, T=200, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    # Undirected adjacency matrix
    upper = rng.random((N, N)) < p_edge
    upper = np.triu(upper, 1)
    A = upper | upper.T

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
        infected_neighbor_count = A[:, infected].sum(axis=1)
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

        SI = A & susceptible[:, None] & infected[None, :]
        s_nodes, i_nodes = np.where(SI)

        rewire_count = 0
        for s_node, i_node in zip(s_nodes, i_nodes):
            if rng.random() >= rho:
                continue

            if not A[s_node, i_node]:
                continue

            A[s_node, i_node] = False
            A[i_node, s_node] = False

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


# ============================================================
# STEP 1: LOAD OBSERVED DATA
# ============================================================

def load_observed_data(
    infected_file="data/infected_timeseries.csv",
    rewiring_file="data/rewiring_timeseries.csv",
    degree_file="data/final_degree_histograms.csv",
):
    infected_df = pd.read_csv(infected_file)
    rewiring_df = pd.read_csv(rewiring_file)
    degree_df = pd.read_csv(degree_file)

    return infected_df, rewiring_df, degree_df


# ============================================================
# STEP 2: TURN OBSERVED DATA INTO ARRAYS BY REPLICATE
# ============================================================

def make_observed_arrays(infected_df, rewiring_df, degree_df):
    infected_pivot = infected_df.pivot(index="replicate_id", columns="time", values="infected_fraction")
    rewiring_pivot = rewiring_df.pivot(index="replicate_id", columns="time", values="rewire_count")
    degree_pivot = degree_df.pivot(index="replicate_id", columns="degree", values="count")

    infected_arr = infected_pivot.sort_index().to_numpy()
    rewiring_arr = rewiring_pivot.sort_index().to_numpy()
    degree_arr = degree_pivot.sort_index().reindex(columns=range(31), fill_value=0).to_numpy()

    return infected_arr, rewiring_arr, degree_arr


# ============================================================
# STEP 3: SUMMARY STATISTICS
# ============================================================
#
# We do not compare raw data point-by-point for all replicates directly.
# Instead, we compress the observed data into a smaller summary vector.
#
# For Question 2, keep it simple and reasonable:
# - mean infected curve across replicates
# - mean rewiring curve across replicates
# - mean final degree histogram across replicates
#
# Then we downsample the two time series so the vector is not too large.
# ============================================================

def build_summary_vector(infected_arr, rewiring_arr, degree_arr, time_stride=10):
    mean_infected = infected_arr.mean(axis=0)
    mean_rewiring = rewiring_arr.mean(axis=0)
    mean_degree = degree_arr.mean(axis=0)

    infected_sub = mean_infected[::time_stride]
    rewiring_sub = mean_rewiring[::time_stride]

    summary = np.concatenate([infected_sub, rewiring_sub, mean_degree])
    return summary


# ============================================================
# STEP 4: SIMULATE MULTIPLE REPLICATES UNDER ONE PARAMETER DRAW
# ============================================================

def simulate_replicates_fast(beta, gamma, rho, R, N=200, p_edge=0.05, n_infected0=5, T=200, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    infected_list = []
    rewiring_list = []
    degree_list = []

    for _ in range(R):
        inf, rew, deg = simulate_fast(
            beta=beta,
            gamma=gamma,
            rho=rho,
            N=N,
            p_edge=p_edge,
            n_infected0=n_infected0,
            T=T,
            rng=rng,
        )
        infected_list.append(inf)
        rewiring_list.append(rew)
        degree_list.append(deg)

    infected_arr = np.array(infected_list)
    rewiring_arr = np.array(rewiring_list)
    degree_arr = np.array(degree_list)

    return infected_arr, rewiring_arr, degree_arr


# ============================================================
# STEP 5: NORMALIZATION FOR DISTANCE
# ============================================================
#
# Different components live on different scales, so we normalize.
# A simple and standard choice is to divide each summary coordinate
# by an estimate of its scale. Here we use prior-predictive standard
# deviations.
# ============================================================

def sample_prior(rng):
    beta = rng.uniform(0.05, 0.50)
    gamma = rng.uniform(0.02, 0.20)
    rho = rng.uniform(0.0, 0.8)
    return beta, gamma, rho


def estimate_summary_scale(observed_R, n_scale_sims=200, time_stride=10, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    summaries = []

    for _ in range(n_scale_sims):
        beta, gamma, rho = sample_prior(rng)
        infected_arr, rewiring_arr, degree_arr = simulate_replicates_fast(
            beta, gamma, rho,
            R=observed_R,
            rng=rng
        )
        s = build_summary_vector(infected_arr, rewiring_arr, degree_arr, time_stride=time_stride)
        summaries.append(s)

    summaries = np.array(summaries)
    scale = summaries.std(axis=0, ddof=1)

    # Avoid divide-by-zero
    scale[scale < 1e-8] = 1.0
    return scale


def weighted_distance(sim_summary, obs_summary, scale):
    z = (sim_summary - obs_summary) / scale
    return np.sqrt(np.sum(z ** 2))


# ============================================================
# STEP 6: REJECTION ABC
# ============================================================

def rejection_abc(
    obs_summary,
    obs_R,
    n_draws=5000,
    accept_frac=0.01,
    time_stride=10,
    scale=None,
    rng=None,
    verbose=True,
):
    if rng is None:
        rng = np.random.default_rng()

    accepted = []
    distances = []

    for i in range(n_draws):
        beta, gamma, rho = sample_prior(rng)

        infected_arr, rewiring_arr, degree_arr = simulate_replicates_fast(
            beta, gamma, rho,
            R=obs_R,
            rng=rng
        )
        sim_summary = build_summary_vector(
            infected_arr, rewiring_arr, degree_arr, time_stride=time_stride
        )

        d = weighted_distance(sim_summary, obs_summary, scale)

        accepted.append([beta, gamma, rho, d])
        distances.append(d)

        if verbose and ((i + 1) % 500 == 0):
            print(f"Completed {i+1}/{n_draws} draws")

    accepted = np.array(accepted)
    distances = accepted[:, 3]

    n_keep = max(1, int(accept_frac * n_draws))
    keep_idx = np.argsort(distances)[:n_keep]
    posterior_samples = accepted[keep_idx, :3]
    kept_distances = distances[keep_idx]

    return posterior_samples, kept_distances, accepted


# ============================================================
# STEP 7: PLOTS
# ============================================================

def plot_posterior_histograms(samples):
    names = [r"$\beta$", r"$\gamma$", r"$\rho$"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for j in range(3):
        axes[j].hist(samples[:, j], bins=30, density=True)
        axes[j].set_title(f"Posterior of {names[j]}")
        axes[j].set_xlabel(names[j])
        axes[j].set_ylabel("Density")

    plt.tight_layout()
    plt.show()


def plot_pairwise(samples):
    names = [r"$\beta$", r"$\gamma$", r"$\rho$"]
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    for i in range(3):
        for j in range(3):
            if i == j:
                axes[i, j].hist(samples[:, j], bins=25, density=True)
            else:
                axes[i, j].scatter(samples[:, j], samples[:, i], s=8, alpha=0.5)

            if i == 2:
                axes[i, j].set_xlabel(names[j])
            if j == 0:
                axes[i, j].set_ylabel(names[i])

    plt.tight_layout()
    plt.show()


# ============================================================
# STEP 8: FULL DRIVER
# ============================================================

def run_question2_abc(
    infected_file="data/infected_timeseries.csv",
    rewiring_file="data/rewiring_timeseries.csv",
    degree_file="data/final_degree_histograms.csv",
    n_draws=5000,
    accept_frac=0.01,
    time_stride=10,
    n_scale_sims=200,
    seed=123,
):
    rng = np.random.default_rng(seed)

    print("Loading observed data...")
    infected_df, rewiring_df, degree_df = load_observed_data(
        infected_file, rewiring_file, degree_file
    )

    print("Converting observed data into arrays...")
    infected_obs, rewiring_obs, degree_obs = make_observed_arrays(
        infected_df, rewiring_df, degree_df
    )

    obs_R = infected_obs.shape[0]
    print(f"Number of observed replicates: {obs_R}")

    print("Building observed summary vector...")
    obs_summary = build_summary_vector(
        infected_obs, rewiring_obs, degree_obs, time_stride=time_stride
    )

    print("Estimating normalization scale from prior predictive simulations...")
    scale = estimate_summary_scale(
        observed_R=obs_R,
        n_scale_sims=n_scale_sims,
        time_stride=time_stride,
        rng=rng,
    )

    print("Running rejection ABC...")
    posterior_samples, kept_distances, all_draws = rejection_abc(
        obs_summary=obs_summary,
        obs_R=obs_R,
        n_draws=n_draws,
        accept_frac=accept_frac,
        time_stride=time_stride,
        scale=scale,
        rng=rng,
        verbose=True,
    )

    print("\nDone.")
    print(f"Accepted {len(posterior_samples)} samples out of {n_draws}")
    print("Posterior means:")
    print("beta  =", posterior_samples[:, 0].mean())
    print("gamma =", posterior_samples[:, 1].mean())
    print("rho   =", posterior_samples[:, 2].mean())

    plot_posterior_histograms(posterior_samples)
    plot_pairwise(posterior_samples)

    return {
        "posterior_samples": posterior_samples,
        "kept_distances": kept_distances,
        "all_draws": all_draws,
        "obs_summary": obs_summary,
        "scale": scale,
    }

results = run_question2_abc()