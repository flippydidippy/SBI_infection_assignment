import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit, prange


# ============================================================
# FAST SIMULATOR  (Numba JIT)
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
# ESTIMATE SCALE + RUN ABC
# ============================================================

def estimate_scale(obs_R, summary_key, n_sims=200, time_stride=10, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    summaries = []
    for i in range(n_sims):
        beta, gamma, rho = sample_prior(rng)
        inf_arr, rew_arr, deg_arr = simulate_replicates_fast(beta, gamma, rho, R=obs_R, rng=rng)
        summaries.append(SUMMARY_FNS[summary_key](inf_arr, rew_arr, deg_arr, time_stride=time_stride))
        if (i + 1) % 50 == 0:
            print(f"    scale sim {i+1}/{n_sims}")
    scale = np.array(summaries).std(axis=0, ddof=1)
    scale[scale < 1e-8] = 1.0
    return scale


def run_abc(obs_summary, obs_R, summary_key, scale,
            n_draws=5000, accept_frac=0.01, time_stride=10,
            rng=None, verbose=True):
    if rng is None:
        rng = np.random.default_rng()
    params, dists = [], []
    for i in range(n_draws):
        beta, gamma, rho = sample_prior(rng)
        inf_arr, rew_arr, deg_arr = simulate_replicates_fast(beta, gamma, rho, R=obs_R, rng=rng)
        sim_s = SUMMARY_FNS[summary_key](inf_arr, rew_arr, deg_arr, time_stride=time_stride)
        params.append((beta, gamma, rho))
        dists.append(weighted_distance(sim_s, obs_summary, scale))
        if verbose and (i + 1) % 500 == 0:
            print(f"  [{summary_key}] {i+1}/{n_draws}")
    params = np.array(params)
    dists  = np.array(dists)
    n_keep = max(1, int(accept_frac * n_draws))
    idx    = np.argsort(dists)[:n_keep]
    return params[idx], dists[idx]


# ============================================================
# PLOTS
# ============================================================

def plot_joint_posterior(samples, title, filename, param_names=("β", "γ", "ρ")):
    """
    All three pairwise joint posteriors as scatter plots.
    Reports posterior correlation for each pair.
    """
    prior_bounds = [(0.05, 0.50), (0.02, 0.20), (0.0, 0.8)]
    pairs = [(0, 2, "β", "ρ"), (0, 1, "β", "γ"), (1, 2, "γ", "ρ")]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (i, j, xi, xj) in zip(axes, pairs):
        ax.scatter(samples[:, i], samples[:, j], s=6, alpha=0.4, color="steelblue")
        corr = np.corrcoef(samples[:, i], samples[:, j])[0, 1]
        lo_i, hi_i = prior_bounds[i]
        lo_j, hi_j = prior_bounds[j]
        ax.set_xlim(lo_i, hi_i)
        ax.set_ylim(lo_j, hi_j)
        ax.set_xlabel(xi, fontsize=12)
        ax.set_ylabel(xj, fontsize=12)
        ax.set_title(f"{xi}–{xj}  (r = {corr:.3f})")

    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.show()


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


def plot_beta_rho_correlation(posteriors: dict, save="beta_rho_correlation_s1_s4.png"):
    """
    β–ρ joint scatter for each summary set side by side.
    Shows how much confounding the rewiring/degree statistics remove
    as we go from S1 (infected curve only) to S4 (all scalars).
    """
    keys   = list(posteriors.keys())
    colors = {"S1": "steelblue", "S2": "darkorange", "S3": "seagreen", "S4": "mediumpurple"}

    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 4), sharey=True)
    if len(keys) == 1:
        axes = [axes]

    for ax, key in zip(axes, keys):
        post = posteriors[key]
        corr = np.corrcoef(post[:, 0], post[:, 2])[0, 1]
        ax.scatter(post[:, 0], post[:, 2], s=6, alpha=0.4,
                   color=colors.get(key, "gray"))
        ax.set_xlim(0.05, 0.50)
        ax.set_ylim(0.00, 0.80)
        ax.set_xlabel("β", fontsize=12)
        if ax is axes[0]:
            ax.set_ylabel("ρ", fontsize=12)
        ax.set_title(f"{SUMMARY_LABELS[key]}\nr(β, ρ) = {corr:.3f}", fontsize=10)

    fig.suptitle(
        "β–ρ joint posterior across summary sets\n"
        "Tighter / rounder cloud = less confounding between β and ρ",
        fontsize=11)
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save}")


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
    n_draws=20000,
    accept_frac=0.01,
    time_stride=10,
    n_scale_sims=200,
    seed=42,
):
    rng = np.random.default_rng(seed)

    print("Compiling Numba simulator...")
    np.random.seed(0)
    _ = simulate_fast(0.2, 0.08, 0.3, 50, 0.05, 3, 10)
    _ = _simulate_replicates_parallel(0.2, 0.08, 0.3, 2, 50, 0.05, 3, 10)
    print("  Done.\n")

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
    plot_beta_rho_correlation(posteriors)
    plot_joint_posterior(
        posteriors["S4"],
        title="Joint posterior — Rejection ABC (S4)",
        filename="q3_s4_joint_posterior.png",
    )

    return posteriors


if __name__ == "__main__":
    posteriors = run_question3()
