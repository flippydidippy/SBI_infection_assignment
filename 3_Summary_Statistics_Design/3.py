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
    rewire_counts = np.zeros(T + 1, dtype=np.int64)
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

        # Phase 3: rewiring — |S|×|I| submatrix
        s_idx = np.flatnonzero(state == 0)
        i_idx = np.flatnonzero(state == 1)

        rewire_count = 0
        if len(s_idx) and len(i_idx):
            A_sub = A[np.ix_(s_idx, i_idx)]
            sub_rs, sub_cs = np.where(A_sub)

            if len(sub_rs) > 0:
                sel = rng.random(len(sub_rs)) < rho
                s_rewire = s_idx[sub_rs[sel]]
                i_rewire = i_idx[sub_cs[sel]]

                A[s_rewire, i_rewire] = False
                A[i_rewire, s_rewire] = False

                for s_node in s_rewire:
                    cand_mask = ~A[s_node]
                    cand_mask[s_node] = False
                    candidates = all_nodes[cand_mask]
                    if candidates.size > 0:
                        new_partner = rng.choice(candidates)
                        A[s_node, new_partner] = True
                        A[new_partner, s_node] = True
                        rewire_count += 1

        infected_fraction[t] = (state == 1).sum() / N
        rewire_counts[t] = rewire_count

    degrees = A.sum(axis=1)
    degree_histogram = np.bincount(np.minimum(degrees, 30), minlength=31)
    return infected_fraction, rewire_counts, degree_histogram


def simulate_replicates_fast(beta, gamma, rho, R, N=200, p_edge=0.05, n_infected0=5, T=200, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    results = [simulate_fast(beta=beta, gamma=gamma, rho=rho, N=N,
                             p_edge=p_edge, n_infected0=n_infected0, T=T, rng=rng)
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
    infected_arr = (infected_df.pivot(index="replicate_id", columns="time", values="infected_fraction")
                    .sort_index().to_numpy())
    rewiring_arr = (rewiring_df.pivot(index="replicate_id", columns="time", values="rewire_count")
                    .sort_index().to_numpy())
    degree_arr   = (degree_df.pivot(index="replicate_id", columns="degree", values="count")
                    .sort_index().reindex(columns=range(31), fill_value=0).to_numpy())
    return infected_arr, rewiring_arr, degree_arr


# ============================================================
# SUMMARY STATISTIC SETS
#
# S1 — Infected curve only
#      Captures epidemic size/timing but cannot separate β from ρ:
#      both suppress the epidemic through different mechanisms.
#
# S2 — Infected + Rewiring curves
#      Rewiring counts are directly driven by ρ (and the number of
#      active S-I edges), so this helps identify ρ separately.
#
# S3 — Infected + Rewiring + Degree histogram  [Q2 baseline]
#      The final degree distribution reflects how much rewiring
#      occurred (high ρ breaks high-degree nodes), adding structural
#      information about the network.
#
# S4 — Scalar summaries
#      A compact hand-crafted vector: epidemic scalars + network scalars.
#      Loses temporal detail but is low-dimensional.
# ============================================================

def build_summary_S1(infected_arr, *_, time_stride=10):
    """Infected fraction curve (downsampled)."""
    return infected_arr.mean(axis=0)[::time_stride]


def build_summary_S2(infected_arr, rewiring_arr, *_, time_stride=10):
    """Infected curve + rewiring counts curve."""
    inf = infected_arr.mean(axis=0)[::time_stride]
    rew = rewiring_arr.mean(axis=0)[::time_stride]
    return np.concatenate([inf, rew])


def build_summary_S3(infected_arr, rewiring_arr, degree_arr, time_stride=10):
    """Infected + rewiring + degree histogram  (Q2 baseline)."""
    inf = infected_arr.mean(axis=0)[::time_stride]
    rew = rewiring_arr.mean(axis=0)[::time_stride]
    deg = degree_arr.mean(axis=0)
    return np.concatenate([inf, rew, deg])


def build_summary_S4(infected_arr, rewiring_arr, degree_arr, **_):
    """Scalar summaries: 7 hand-crafted statistics."""
    mean_inf = infected_arr.mean(axis=0)
    peak_inf     = mean_inf.max()
    t_peak       = mean_inf.argmax() / len(mean_inf)   # normalised
    epidemic_auc = mean_inf.mean()                      # ~ total burden

    mean_rew     = rewiring_arr.mean(axis=0)
    total_rew    = mean_rew.sum()
    peak_rew     = mean_rew.max()

    deg_bins     = np.arange(31)
    mean_deg_dist = degree_arr.mean(axis=0)
    denom        = mean_deg_dist.sum() if mean_deg_dist.sum() > 0 else 1.0
    mean_deg     = (mean_deg_dist * deg_bins).sum() / denom
    var_deg      = (mean_deg_dist * deg_bins**2).sum() / denom - mean_deg**2
    std_deg      = np.sqrt(max(0.0, var_deg))

    return np.array([peak_inf, t_peak, epidemic_auc, total_rew, peak_rew, mean_deg, std_deg])


SUMMARY_FNS = {
    "S1": build_summary_S1,
    "S2": build_summary_S2,
    "S3": build_summary_S3,
    "S4": build_summary_S4,
}

SUMMARY_LABELS = {
    "S1": "S1: Infected curve",
    "S2": "S2: + Rewiring curve",
    "S3": "S3: + Degree hist.",
    "S4": "S4: Scalar summaries",
}


# ============================================================
# PRIOR + DISTANCE
# ============================================================

def sample_prior(rng):
    beta  = rng.uniform(0.05, 0.50)
    gamma = rng.uniform(0.02, 0.20)
    rho   = rng.uniform(0.0,  0.8)
    return beta, gamma, rho


def weighted_distance(sim_summary, obs_summary, scale):
    z = (sim_summary - obs_summary) / scale
    return np.sqrt(np.sum(z ** 2))


# ============================================================
# PARALLEL WORKERS
# ============================================================

def _scale_worker(args):
    obs_R, time_stride, seed, summary_key = args
    rng = np.random.default_rng(seed)
    beta, gamma, rho = sample_prior(rng)
    inf_arr, rew_arr, deg_arr = simulate_replicates_fast(beta, gamma, rho, R=obs_R, rng=rng)
    return SUMMARY_FNS[summary_key](inf_arr, rew_arr, deg_arr, time_stride=time_stride)


def _abc_worker(args):
    obs_summary, obs_R, time_stride, scale, seed, summary_key = args
    rng = np.random.default_rng(seed)
    beta, gamma, rho = sample_prior(rng)
    inf_arr, rew_arr, deg_arr = simulate_replicates_fast(beta, gamma, rho, R=obs_R, rng=rng)
    sim_summary = SUMMARY_FNS[summary_key](inf_arr, rew_arr, deg_arr, time_stride=time_stride)
    d = weighted_distance(sim_summary, obs_summary, scale)
    return beta, gamma, rho, d


# ============================================================
# ESTIMATE SCALE + RUN ABC (parallelised)
# ============================================================

def estimate_scale(obs_R, summary_key, n_sims=200, time_stride=10, rng=None, n_workers=None):
    if rng is None:
        rng = np.random.default_rng()
    if n_workers is None:
        n_workers = os.cpu_count()

    seeds = rng.integers(0, 2**31, size=n_sims)
    args  = [(obs_R, time_stride, int(s), summary_key) for s in seeds]

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        summaries = list(ex.map(_scale_worker, args, chunksize=20))

    summaries = np.array(summaries)
    scale = summaries.std(axis=0, ddof=1)
    scale[scale < 1e-8] = 1.0
    return scale


def run_abc(obs_summary, obs_R, summary_key, scale,
            n_draws=5000, accept_frac=0.01, time_stride=10,
            rng=None, n_workers=None, verbose=True):
    if rng is None:
        rng = np.random.default_rng()
    if n_workers is None:
        n_workers = os.cpu_count()

    seeds = rng.integers(0, 2**31, size=n_draws)
    args  = [(obs_summary, obs_R, time_stride, scale, int(s), summary_key) for s in seeds]

    rows = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for i, result in enumerate(ex.map(_abc_worker, args, chunksize=50)):
            rows.append(result)
            if verbose and (i + 1) % 500 == 0:
                print(f"  [{summary_key}] {i+1}/{n_draws}")

    all_draws = np.array(rows)
    distances = all_draws[:, 3]
    n_keep    = max(1, int(accept_frac * n_draws))
    keep_idx  = np.argsort(distances)[:n_keep]
    return all_draws[keep_idx, :3], distances[keep_idx]


# ============================================================
# PLOTS
# ============================================================

def plot_posterior_comparison(posteriors: dict, param_names=("β", "γ", "ρ"), prior_bounds=None):
    """
    Grid: rows = parameters, cols = summary sets.
    Shows how posterior concentrates (or fails to) under each summary.
    """
    if prior_bounds is None:
        prior_bounds = {"β": (0.05, 0.50), "γ": (0.02, 0.20), "ρ": (0.0, 0.8)}

    keys = list(posteriors.keys())
    n_sets = len(keys)
    n_params = len(param_names)

    fig, axes = plt.subplots(n_params, n_sets, figsize=(3.5 * n_sets, 3 * n_params),
                             sharey="row")

    for col, key in enumerate(keys):
        samples = posteriors[key]   # shape (n_accepted, 3)
        for row, param in enumerate(param_names):
            ax = axes[row, col]
            ax.hist(samples[:, row], bins=30, density=True, color="steelblue", alpha=0.75)
            lo, hi = prior_bounds[param]
            ax.axhline(1.0 / (hi - lo), color="gray", linestyle="--", linewidth=1,
                       label="prior" if row == 0 and col == 0 else None)
            ax.set_xlim(lo, hi)
            if row == 0:
                ax.set_title(SUMMARY_LABELS[key], fontsize=10)
            if col == 0:
                ax.set_ylabel(f"Posterior of {param}", fontsize=10)

    fig.suptitle("Posterior distributions under different summary statistic sets", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig("posterior_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_posterior_widths(posteriors: dict, param_names=("β", "γ", "ρ"), prior_stds=None):
    """
    Bar chart: posterior std for each parameter under each summary set,
    normalised by the prior std. Lower = more informative.
    """
    if prior_stds is None:
        prior_stds = np.array([
            (0.50 - 0.05) / np.sqrt(12),   # β ~ Uniform(0.05, 0.50)
            (0.20 - 0.02) / np.sqrt(12),   # γ ~ Uniform(0.02, 0.20)
            (0.80 - 0.00) / np.sqrt(12),   # ρ ~ Uniform(0.0, 0.8)
        ])

    keys  = list(posteriors.keys())
    stds  = np.array([[posteriors[k][:, j].std() / prior_stds[j]
                       for j in range(3)]
                      for k in keys])   # (n_sets, 3)

    x     = np.arange(len(keys))
    width = 0.25
    colors = ["#4c72b0", "#55a868", "#c44e52"]
    labels = [f"{p}" for p in param_names]

    fig, ax = plt.subplots(figsize=(9, 4))
    for j in range(3):
        ax.bar(x + j * width, stds[:, j], width, label=labels[j], color=colors[j], alpha=0.85)

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="prior (baseline)")
    ax.set_xticks(x + width)
    ax.set_xticklabels([SUMMARY_LABELS[k] for k in keys], rotation=10, ha="right")
    ax.set_ylabel("Posterior std / Prior std  (lower = more informative)")
    ax.set_title("Information content: how much each summary set constrains each parameter")
    ax.legend()
    plt.tight_layout()
    plt.savefig("posterior_widths.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_sensitivity(obs_R, seed=42):
    """
    Vary one parameter at a time (others at central values) and plot
    how each summary statistic responds. Shows which summaries are
    sensitive to which parameters.
    """
    rng = np.random.default_rng(seed)

    # Central parameter values
    beta_c, gamma_c, rho_c = 0.20, 0.08, 0.30

    sweeps = {
        "β":  (np.linspace(0.05, 0.50, 12), "beta",  [(b, gamma_c, rho_c)  for b in np.linspace(0.05, 0.50, 12)]),
        "γ":  (np.linspace(0.02, 0.20, 12), "gamma", [(beta_c, g, rho_c)   for g in np.linspace(0.02, 0.20, 12)]),
        "ρ":  (np.linspace(0.0,  0.80, 12), "rho",   [(beta_c, gamma_c, r) for r in np.linspace(0.0, 0.80, 12)]),
    }

    # Stats to track: peak infection, total rewirings, mean final degree
    stat_labels = ["Peak infection", "Total rewirings", "Mean final degree"]

    fig, axes = plt.subplots(len(sweeps), 3, figsize=(12, 9), sharey=False)

    for row, (param_name, (vals, _, param_triples)) in enumerate(sweeps.items()):
        peaks, total_rews, mean_degs = [], [], []

        for triple in param_triples:
            beta, gamma, rho = triple
            inf_arr, rew_arr, deg_arr = simulate_replicates_fast(
                beta, gamma, rho, R=obs_R, rng=rng
            )
            mean_inf = inf_arr.mean(axis=0)
            peaks.append(mean_inf.max())
            total_rews.append(rew_arr.mean(axis=0).sum())
            deg_bins = np.arange(31)
            mean_deg_dist = deg_arr.mean(axis=0)
            denom = mean_deg_dist.sum() if mean_deg_dist.sum() > 0 else 1.0
            mean_degs.append((mean_deg_dist * deg_bins).sum() / denom)

        for col, (stat, label) in enumerate(zip([peaks, total_rews, mean_degs], stat_labels)):
            ax = axes[row, col]
            ax.plot(vals, stat, marker="o", markersize=4)
            ax.set_xlabel(param_name)
            if col == 0:
                ax.set_ylabel(f"Varying {param_name}")
            if row == 0:
                ax.set_title(label)

    fig.suptitle("Sensitivity: how each summary statistic responds to each parameter", fontsize=12)
    plt.tight_layout()
    plt.savefig("sensitivity.png", dpi=150, bbox_inches="tight")
    plt.show()


# ============================================================
# MAIN DRIVER
# ============================================================

def run_question3(
    infected_file="data/infected_timeseries.csv",
    rewiring_file="data/rewiring_timeseries.csv",
    degree_file="data/final_degree_histograms.csv",
    summary_keys=("S1", "S2", "S3", "S4"),
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
    infected_obs, rewiring_obs, degree_obs = make_observed_arrays(
        infected_df, rewiring_df, degree_df
    )
    obs_R = infected_obs.shape[0]
    print(f"Observed replicates: {obs_R}")

    # ---- Sensitivity plot (uses simulation, not ABC) ----
    print("\nGenerating sensitivity plots...")
    plot_sensitivity(obs_R=obs_R, seed=seed)

    # ---- Run ABC under each summary set ----
    posteriors = {}

    for key in summary_keys:
        print(f"\n--- Summary set {key}: {SUMMARY_LABELS[key]} ---")

        # Build observed summary for this statistic set
        fn = SUMMARY_FNS[key]
        obs_summary = fn(infected_obs, rewiring_obs, degree_obs, time_stride=time_stride)

        print(f"  Estimating scale ({n_scale_sims} prior-predictive sims)...")
        scale = estimate_scale(
            obs_R=obs_R, summary_key=key,
            n_sims=n_scale_sims, time_stride=time_stride, rng=rng,
        )

        print(f"  Running ABC ({n_draws} draws)...")
        post_samples, _ = run_abc(
            obs_summary=obs_summary, obs_R=obs_R,
            summary_key=key, scale=scale,
            n_draws=n_draws, accept_frac=accept_frac,
            time_stride=time_stride, rng=rng,
        )
        posteriors[key] = post_samples

        print(f"  Accepted {len(post_samples)} samples")
        print(f"  Posterior means:  β={post_samples[:,0].mean():.3f}  "
              f"γ={post_samples[:,1].mean():.3f}  ρ={post_samples[:,2].mean():.3f}")
        print(f"  Posterior stds:   β={post_samples[:,0].std():.3f}  "
              f"γ={post_samples[:,1].std():.3f}  ρ={post_samples[:,2].std():.3f}")

    # ---- Comparison plots ----
    print("\nGenerating comparison plots...")
    plot_posterior_comparison(posteriors)
    plot_posterior_widths(posteriors)

    return posteriors


if __name__ == "__main__":
    posteriors = run_question3()
