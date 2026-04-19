import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit, prange


# ============================================================
# NUMBA SIMULATOR
# ============================================================

@njit
def simulate_fast(beta, gamma, rho, N, p_edge, n_infected0, T):
    nbrs = np.full((N, N), np.int32(-1), dtype=np.int32)
    deg  = np.zeros(N, dtype=np.int32)

    for i in range(N):
        for j in range(i + 1, N):
            if np.random.random() < p_edge:
                nbrs[i, deg[i]] = j
                deg[i] += 1
                nbrs[j, deg[j]] = i
                deg[j] += 1

    state = np.zeros(N, dtype=np.int8)
    perm  = np.random.permutation(N)
    for k in range(n_infected0):
        state[perm[k]] = 1

    infected_fraction = np.zeros(T + 1, dtype=np.float64)
    rewire_counts     = np.zeros(T + 1, dtype=np.int64)
    infected_fraction[0] = n_infected0 / N

    si_s = np.empty(N * N, dtype=np.int32)
    si_i = np.empty(N * N, dtype=np.int32)

    for t in range(1, T + 1):
        inf_count = 0
        for i in range(N):
            if state[i] == 1:
                inf_count += 1
        if inf_count == 0:
            break

        inf_nbr = np.zeros(N, dtype=np.int32)
        for i in range(N):
            for ki in range(deg[i]):
                if state[nbrs[i, ki]] == 1:
                    inf_nbr[i] += 1

        for i in range(N):
            if state[i] == 0 and inf_nbr[i] > 0:
                p_inf = 1.0 - (1.0 - beta) ** inf_nbr[i]
                if np.random.random() < p_inf:
                    state[i] = 1

        for i in range(N):
            if state[i] == 1 and np.random.random() < gamma:
                state[i] = 2

        n_si = 0
        for i in range(N):
            if state[i] == 0:
                for ki in range(deg[i]):
                    j = nbrs[i, ki]
                    if state[j] == 1:
                        si_s[n_si] = i
                        si_i[n_si] = j
                        n_si += 1

        rewire_count = 0
        for k in range(n_si):
            if np.random.random() < rho:
                s_node = int(si_s[k])
                i_node = int(si_i[k])
                edge_exists = False
                for idx in range(deg[s_node]):
                    if nbrs[s_node, idx] == i_node:
                        edge_exists = True
                        break
                if not edge_exists:
                    continue
                for idx in range(deg[s_node]):
                    if nbrs[s_node, idx] == i_node:
                        nbrs[s_node, idx] = nbrs[s_node, deg[s_node] - 1]
                        nbrs[s_node, deg[s_node] - 1] = -1
                        deg[s_node] -= 1
                        break
                for idx in range(deg[i_node]):
                    if nbrs[i_node, idx] == s_node:
                        nbrs[i_node, idx] = nbrs[i_node, deg[i_node] - 1]
                        nbrs[i_node, deg[i_node] - 1] = -1
                        deg[i_node] -= 1
                        break
                n_cand = N - 1 - deg[s_node]
                if n_cand > 0:
                    for _ in range(N):
                        c = int(np.random.random() * N)
                        if c == s_node:
                            continue
                        already = False
                        for idx in range(deg[s_node]):
                            if nbrs[s_node, idx] == c:
                                already = True
                                break
                        if not already:
                            nbrs[s_node, deg[s_node]] = c
                            deg[s_node] += 1
                            nbrs[c, deg[c]] = s_node
                            deg[c] += 1
                            rewire_count += 1
                            break

        n_inf = 0
        for i in range(N):
            if state[i] == 1:
                n_inf += 1
        infected_fraction[t] = n_inf / N
        rewire_counts[t] = rewire_count

    degree_histogram = np.zeros(31, dtype=np.int64)
    for i in range(N):
        d = int(min(deg[i], 30))
        degree_histogram[d] += 1

    return infected_fraction, rewire_counts, degree_histogram


@njit(parallel=True)
def _simulate_replicates_parallel(beta, gamma, rho, R, N, p_edge, n_infected0, T):
    infected_arr = np.zeros((R, T + 1))
    rewiring_arr = np.zeros((R, T + 1))
    degree_arr   = np.zeros((R, 31))
    for r in prange(R):
        inf, rew, deg = simulate_fast(beta, gamma, rho, N, p_edge, n_infected0, T)
        infected_arr[r] = inf
        rewiring_arr[r] = rew
        degree_arr[r]   = deg
    return infected_arr, rewiring_arr, degree_arr


def simulate_replicates_fast(beta, gamma, rho, R, N=200, p_edge=0.05,
                              n_infected0=5, T=200, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    np.random.seed(rng.integers(0, 2**31))
    return _simulate_replicates_parallel(beta, gamma, rho, R, N, p_edge, n_infected0, T)


# ============================================================
# LOAD + PREPROCESS OBSERVED DATA
# ============================================================

def load_observed_data(
    infected_file="data/infected_timeseries.csv",
    rewiring_file="data/rewiring_timeseries.csv",
    degree_file="data/final_degree_histograms.csv",
):
    infected_df = pd.read_csv(infected_file)
    rewiring_df = pd.read_csv(rewiring_file)
    degree_df   = pd.read_csv(degree_file)
    return infected_df, rewiring_df, degree_df


def make_observed_arrays(infected_df, rewiring_df, degree_df):
    infected_arr = (infected_df.pivot(index="replicate_id", columns="time",
                                      values="infected_fraction")
                    .sort_index().to_numpy())
    rewiring_arr = (rewiring_df.pivot(index="replicate_id", columns="time",
                                      values="rewire_count")
                    .sort_index().to_numpy())
    degree_arr   = (degree_df.pivot(index="replicate_id", columns="degree", values="count")
                    .sort_index().reindex(columns=range(31), fill_value=0).to_numpy())
    return infected_arr, rewiring_arr, degree_arr


# ============================================================
# SUMMARY STATISTICS
# ============================================================

def build_summary_vector(infected_arr, *_, time_stride=10):
    """S1: infected fraction curve (downsampled)."""
    return infected_arr.mean(axis=0)[::time_stride]


def weighted_distance(sim_summary, obs_summary, scale):
    z = (sim_summary - obs_summary) / scale
    return np.sqrt(np.sum(z ** 2))


# ============================================================
# PRIOR
# ============================================================

def sample_prior(rng):
    beta  = rng.uniform(0.05, 0.50)
    gamma = rng.uniform(0.02, 0.20)
    rho   = rng.uniform(0.0,  0.8)
    return beta, gamma, rho


# ============================================================
# ESTIMATE SCALE + RUN ABC
# ============================================================

def estimate_summary_scale(observed_R, n_scale_sims=200, time_stride=10, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    summaries = []
    for i in range(n_scale_sims):
        beta, gamma, rho = sample_prior(rng)
        inf_arr, rew_arr, deg_arr = simulate_replicates_fast(
            beta, gamma, rho, R=observed_R, rng=rng)
        summaries.append(build_summary_vector(inf_arr, rew_arr, deg_arr,
                                              time_stride=time_stride))
        if (i + 1) % 50 == 0:
            print(f"  scale sim {i+1}/{n_scale_sims}")
    summaries = np.array(summaries)
    scale = summaries.std(axis=0, ddof=1)
    scale[scale < 1e-8] = 1.0
    return scale


def rejection_abc(obs_summary, obs_R, scale,
                  n_draws=20000, accept_frac=0.01, time_stride=10,
                  rng=None, verbose=True):
    if rng is None:
        rng = np.random.default_rng()
    rows = []
    for i in range(n_draws):
        beta, gamma, rho = sample_prior(rng)
        inf_arr, rew_arr, deg_arr = simulate_replicates_fast(
            beta, gamma, rho, R=obs_R, rng=rng)
        sim_summary = build_summary_vector(inf_arr, rew_arr, deg_arr,
                                           time_stride=time_stride)
        d = weighted_distance(sim_summary, obs_summary, scale)
        rows.append((beta, gamma, rho, d))
        if verbose and (i + 1) % 2000 == 0:
            print(f"  {i+1}/{n_draws}")

    all_draws = np.array(rows)
    distances = all_draws[:, 3]
    n_keep    = max(1, int(accept_frac * n_draws))
    keep_idx  = np.argsort(distances)[:n_keep]
    return all_draws[keep_idx, :3], distances[keep_idx], all_draws


# ============================================================
# PLOTS
# ============================================================

def plot_posterior_histograms(samples):
    names  = [r"$\beta$", r"$\gamma$", r"$\rho$"]
    bounds = [(0.05, 0.50), (0.02, 0.20), (0.0, 0.8)]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for j, (lo, hi) in enumerate(bounds):
        axes[j].hist(samples[:, j], bins=30, density=True)
        axes[j].axhline(1.0 / (hi - lo), color="gray", linestyle="--", linewidth=1, label="prior")
        axes[j].set_xlim(lo, hi)
        axes[j].set_title(f"Posterior of {names[j]}")
        axes[j].set_xlabel(names[j])
        axes[j].set_ylabel("Density")
        axes[j].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("q2_posterior_histograms.png", dpi=150, bbox_inches="tight")
    print("Saved: q2_posterior_histograms.png")
    plt.show()


def plot_joint_posterior(samples):
    names  = ["β", "γ", "ρ"]
    bounds = [(0.05, 0.50), (0.02, 0.20), (0.0, 0.8)]
    pairs  = [(0, 2, "β", "ρ"), (0, 1, "β", "γ"), (1, 2, "γ", "ρ")]

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))

    for col, (name, (lo, hi)) in enumerate(zip(names, bounds)):
        ax = axes[0, col]
        ax.hist(samples[:, col], bins=30, density=True,
                color="steelblue", alpha=0.75)
        ax.axhline(1.0 / (hi - lo), color="gray", linestyle="--",
                   linewidth=1, label="prior")
        ax.set_xlim(lo, hi)
        ax.set_xlabel(name, fontsize=12)
        ax.set_ylabel("Density")
        ax.set_title(f"Posterior of {name}  "
                     f"(mean={samples[:,col].mean():.3f}, "
                     f"std={samples[:,col].std():.3f})")
        if col == 0:
            ax.legend(fontsize=8)

    for col, (i, j, xi, xj) in enumerate(pairs):
        ax = axes[1, col]
        ax.scatter(samples[:, i], samples[:, j], s=8, alpha=0.4, color="steelblue")
        corr = np.corrcoef(samples[:, i], samples[:, j])[0, 1]
        lo_i, hi_i = bounds[i]
        lo_j, hi_j = bounds[j]
        ax.set_xlim(lo_i, hi_i)
        ax.set_ylim(lo_j, hi_j)
        ax.set_xlabel(xi, fontsize=12)
        ax.set_ylabel(xj, fontsize=12)
        ax.set_title(f"{xi}–{xj}  (r = {corr:.3f})")

    fig.suptitle("Rejection ABC (S1) — marginal and joint posteriors", fontsize=12)
    plt.tight_layout()
    plt.savefig("q2_joint_posterior.png", dpi=150, bbox_inches="tight")
    print("Saved: q2_joint_posterior.png")
    plt.show()


# ============================================================
# MAIN DRIVER
# ============================================================

def run_question2_abc(
    infected_file="data/infected_timeseries.csv",
    rewiring_file="data/rewiring_timeseries.csv",
    degree_file="data/final_degree_histograms.csv",
    n_draws=20000,
    accept_frac=0.01,
    time_stride=10,
    n_scale_sims=200,
    seed=42,
):
    rng = np.random.default_rng(seed)

    print("Compiling simulator (one-time Numba JIT)...")
    np.random.seed(0)
    _ = simulate_fast(0.2, 0.08, 0.3, 50, 0.05, 3, 10)
    _ = _simulate_replicates_parallel(0.2, 0.08, 0.3, 2, 50, 0.05, 3, 10)
    print("  Done.\n")

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

    print(f"Estimating normalization scale ({n_scale_sims} prior-predictive sims)...")
    scale = estimate_summary_scale(
        observed_R=obs_R, n_scale_sims=n_scale_sims,
        time_stride=time_stride, rng=rng,
    )

    print(f"Running rejection ABC ({n_draws} draws)...")
    posterior_samples, kept_distances, all_draws = rejection_abc(
        obs_summary=obs_summary, obs_R=obs_R, scale=scale,
        n_draws=n_draws, accept_frac=accept_frac,
        time_stride=time_stride, rng=rng, verbose=True,
    )

    print(f"\nDone. Accepted {len(posterior_samples)} samples out of {n_draws}")
    print(f"Posterior means:  beta={posterior_samples[:,0].mean():.3f}  "
          f"gamma={posterior_samples[:,1].mean():.3f}  "
          f"rho={posterior_samples[:,2].mean():.3f}")

    plot_posterior_histograms(posterior_samples)
    plot_joint_posterior(posterior_samples)

    return {
        "posterior_samples": posterior_samples,
        "kept_distances":    kept_distances,
        "all_draws":         all_draws,
        "obs_summary":       obs_summary,
        "scale":             scale,
    }


if __name__ == "__main__":
    results = run_question2_abc()
