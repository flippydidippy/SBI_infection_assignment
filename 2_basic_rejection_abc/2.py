import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import os


# ============================================================
# FAST SIMULATOR
# ============================================================

def simulate_fast(beta, gamma, rho, N=200, p_edge=0.05, n_infected0=5, T=200, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    upper = rng.random((N, N)) < p_edge
    upper = np.triu(upper, 1)
    A = upper | upper.T

    state = np.zeros(N, dtype=np.int8)
    state[rng.choice(N, size=n_infected0, replace=False)] = 1

    infected_fraction = np.zeros(T + 1, dtype=float)
    rewire_counts     = np.zeros(T + 1, dtype=np.int64)
    infected_fraction[0] = n_infected0 / N

    all_nodes = np.arange(N)

    for t in range(1, T + 1):
        i_idx = np.flatnonzero(state == 1)
        if len(i_idx) == 0:
            break

        # Phase 1: infection
        infected_neighbor_count = A[:, i_idx].sum(axis=1)
        p_inf = 1.0 - np.power(1.0 - beta, infected_neighbor_count)
        state[(state == 0) & (rng.random(N) < p_inf)] = 1

        # Phase 2: recovery
        infected_mask = state == 1
        state[infected_mask & (rng.random(N) < gamma)] = 2

        # Phase 3: rewiring — vectorised over all S-I edges at once
        s_idx = np.flatnonzero(state == 0)
        i_idx = np.flatnonzero(state == 1)

        rewire_count = 0
        if len(s_idx) and len(i_idx):
            A_sub = A[np.ix_(s_idx, i_idx)]
            sub_rs, sub_cs = np.where(A_sub)

            if len(sub_rs) > 0:
                sel      = rng.random(len(sub_rs)) < rho
                s_rewire = s_idx[sub_rs[sel]]
                i_rewire = i_idx[sub_cs[sel]]

                A[s_rewire, i_rewire] = False
                A[i_rewire, s_rewire] = False

                for s_node in s_rewire:
                    cand_mask        = ~A[s_node]
                    cand_mask[s_node] = False
                    candidates       = all_nodes[cand_mask]
                    if candidates.size > 0:
                        new_partner            = rng.choice(candidates)
                        A[s_node, new_partner] = True
                        A[new_partner, s_node] = True
                        rewire_count          += 1

        infected_fraction[t] = (state == 1).sum() / N
        rewire_counts[t]     = rewire_count

    degrees          = A.sum(axis=1)
    degree_histogram = np.bincount(np.minimum(degrees, 30), minlength=31)
    return infected_fraction, rewire_counts, degree_histogram


def simulate_replicates_fast(beta, gamma, rho, R, N=200, p_edge=0.05, n_infected0=5, T=200, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    results      = [simulate_fast(beta=beta, gamma=gamma, rho=rho,
                                  N=N, p_edge=p_edge, n_infected0=n_infected0,
                                  T=T, rng=rng)
                    for _ in range(R)]
    infected_arr = np.array([r[0] for r in results])
    rewiring_arr = np.array([r[1] for r in results])
    degree_arr   = np.array([r[2] for r in results])
    return infected_arr, rewiring_arr, degree_arr


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
# PARALLEL WORKERS
# ============================================================

def _scale_worker(args):
    obs_R, time_stride, seed = args
    rng = np.random.default_rng(seed)
    beta, gamma, rho = sample_prior(rng)
    inf_arr, rew_arr, deg_arr = simulate_replicates_fast(beta, gamma, rho, R=obs_R, rng=rng)
    return build_summary_vector(inf_arr, rew_arr, deg_arr, time_stride=time_stride)


def _abc_worker(args):
    obs_summary, obs_R, time_stride, scale, seed = args
    rng = np.random.default_rng(seed)
    beta, gamma, rho = sample_prior(rng)
    inf_arr, rew_arr, deg_arr = simulate_replicates_fast(beta, gamma, rho, R=obs_R, rng=rng)
    sim_summary = build_summary_vector(inf_arr, rew_arr, deg_arr, time_stride=time_stride)
    d = weighted_distance(sim_summary, obs_summary, scale)
    return beta, gamma, rho, d


# ============================================================
# ESTIMATE SCALE + RUN ABC (parallelised)
# ============================================================

def estimate_summary_scale(observed_R, n_scale_sims=200, time_stride=10, rng=None, n_workers=None):
    if rng is None:
        rng = np.random.default_rng()
    if n_workers is None:
        n_workers = os.cpu_count()

    seeds = rng.integers(0, 2**31, size=n_scale_sims)
    args  = [(observed_R, time_stride, int(s)) for s in seeds]

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        summaries = list(ex.map(_scale_worker, args, chunksize=20))

    summaries = np.array(summaries)
    scale = summaries.std(axis=0, ddof=1)
    scale[scale < 1e-8] = 1.0
    return scale


def rejection_abc(obs_summary, obs_R, scale,
                  n_draws=5000, accept_frac=0.01, time_stride=10,
                  rng=None, n_workers=None, verbose=True):
    if rng is None:
        rng = np.random.default_rng()
    if n_workers is None:
        n_workers = os.cpu_count()

    seeds = rng.integers(0, 2**31, size=n_draws)
    args  = [(obs_summary, obs_R, time_stride, scale, int(s)) for s in seeds]

    rows = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for i, result in enumerate(ex.map(_abc_worker, args, chunksize=50)):
            rows.append(result)
            if verbose and (i + 1) % 500 == 0:
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
# MAIN DRIVER
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
    plot_pairwise(posterior_samples)

    return {
        "posterior_samples": posterior_samples,
        "kept_distances":    kept_distances,
        "all_draws":         all_draws,
        "obs_summary":       obs_summary,
        "scale":             scale,
    }


if __name__ == "__main__":
    results = run_question2_abc()
