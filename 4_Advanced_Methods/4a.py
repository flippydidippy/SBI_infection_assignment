"""
4a. Advanced Methods — Local Linear Regression Adjustment (Beaumont et al. 2002)

Key idea: rejection ABC keeps samples whose simulated summaries s_i are *close*
to s_obs, not exactly equal.  That residual gap biases the posterior.  Regression
adjustment fits, on the accepted set only:

    θ_i = α + β^T (s_i − s_obs) + ε_i

and then shifts each accepted sample to what its parameter "would have been" if
its summary had landed exactly at s_obs:

    θ_i* = θ_i − β̂^T (s_i − s_obs)

This sharpens and de-biases the posterior with zero new simulations.

Implementation choices:
  - Summary set : S4 (7 scalar statistics) — low-dimensional, well-conditioned
                  for regression.  S3's 73 dims would need >> accepted samples.
  - Weights     : Epanechnikov kernel  K(d/h),  h = max accepted distance.
  - Transforms  : logit-transform each bounded parameter before regression so
                  adjusted values stay inside the prior support.
  - accept_frac : raised to 0.05 (5 %) — regression corrects the extra bias,
                  so we gain more samples for the same final quality.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import os


# ============================================================
# SIMULATOR  (identical to 3.py)
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

        infected_neighbor_count = A[:, i_idx].sum(axis=1)
        p_inf = 1.0 - np.power(1.0 - beta, infected_neighbor_count)
        state[(state == 0) & (rng.random(N) < p_inf)] = 1

        infected_mask = state == 1
        state[infected_mask & (rng.random(N) < gamma)] = 2

        s_idx = np.flatnonzero(state == 0)
        i_idx = np.flatnonzero(state == 1)

        rewire_count = 0
        if len(s_idx) and len(i_idx):
            A_sub  = A[np.ix_(s_idx, i_idx)]
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
                        new_partner              = rng.choice(candidates)
                        A[s_node, new_partner]   = True
                        A[new_partner, s_node]   = True
                        rewire_count            += 1

        infected_fraction[t] = (state == 1).sum() / N
        rewire_counts[t]     = rewire_count

    degrees          = A.sum(axis=1)
    degree_histogram = np.bincount(np.minimum(degrees, 30), minlength=31)
    return infected_fraction, rewire_counts, degree_histogram


def simulate_replicates_fast(beta, gamma, rho, R, N=200, p_edge=0.05,
                              n_infected0=5, T=200, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    results     = [simulate_fast(beta=beta, gamma=gamma, rho=rho,
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
    degree_arr   = (degree_df.pivot(index="replicate_id", columns="degree",
                                    values="count")
                    .sort_index().reindex(columns=range(31), fill_value=0).to_numpy())
    return infected_arr, rewiring_arr, degree_arr


# ============================================================
# SUMMARY STATISTIC S4  (7 hand-crafted scalars)
# ============================================================

def build_summary_S4(infected_arr, rewiring_arr, degree_arr, **_):
    """Scalar summaries: peak infection, t_peak, AUC, total/peak rewirings,
    mean and std of final degree distribution."""
    mean_inf     = infected_arr.mean(axis=0)
    peak_inf     = mean_inf.max()
    t_peak       = mean_inf.argmax() / len(mean_inf)
    epidemic_auc = mean_inf.mean()

    mean_rew  = rewiring_arr.mean(axis=0)
    total_rew = mean_rew.sum()
    peak_rew  = mean_rew.max()

    deg_bins      = np.arange(31)
    mean_deg_dist = degree_arr.mean(axis=0)
    denom         = mean_deg_dist.sum() if mean_deg_dist.sum() > 0 else 1.0
    mean_deg      = (mean_deg_dist * deg_bins).sum() / denom
    var_deg       = (mean_deg_dist * deg_bins**2).sum() / denom - mean_deg**2
    std_deg       = np.sqrt(max(0.0, var_deg))

    return np.array([peak_inf, t_peak, epidemic_auc, total_rew, peak_rew,
                     mean_deg, std_deg])


# ============================================================
# PRIOR + DISTANCE
# ============================================================

PRIOR_BOUNDS = [(0.05, 0.50), (0.02, 0.20), (0.00, 0.80)]   # β, γ, ρ
PARAM_NAMES  = ["β", "γ", "ρ"]


def sample_prior(rng):
    beta  = rng.uniform(0.05, 0.50)
    gamma = rng.uniform(0.02, 0.20)
    rho   = rng.uniform(0.0,  0.80)
    return beta, gamma, rho


def weighted_distance(sim_summary, obs_summary, scale):
    z = (sim_summary - obs_summary) / scale
    return np.sqrt(np.sum(z ** 2))


# ============================================================
# PARALLEL WORKERS
# ============================================================

def _scale_worker(args):
    obs_R, seed = args
    rng = np.random.default_rng(seed)
    beta, gamma, rho = sample_prior(rng)
    inf_arr, rew_arr, deg_arr = simulate_replicates_fast(beta, gamma, rho, R=obs_R, rng=rng)
    return build_summary_S4(inf_arr, rew_arr, deg_arr)


def _abc_worker(args):
    obs_summary, obs_R, scale, seed = args
    rng = np.random.default_rng(seed)
    beta, gamma, rho = sample_prior(rng)
    inf_arr, rew_arr, deg_arr = simulate_replicates_fast(beta, gamma, rho, R=obs_R, rng=rng)
    sim_summary = build_summary_S4(inf_arr, rew_arr, deg_arr)
    d = weighted_distance(sim_summary, obs_summary, scale)
    return beta, gamma, rho, d, sim_summary


# ============================================================
# ESTIMATE SCALE + RUN ABC (parallelised)
# ============================================================

def estimate_scale(obs_R, n_sims=200, rng=None, n_workers=None):
    if rng is None:
        rng = np.random.default_rng()
    if n_workers is None:
        n_workers = os.cpu_count()

    seeds = rng.integers(0, 2**31, size=n_sims)
    args  = [(obs_R, int(s)) for s in seeds]

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        summaries = list(ex.map(_scale_worker, args, chunksize=20))

    summaries = np.array(summaries)
    scale = summaries.std(axis=0, ddof=1)
    scale[scale < 1e-8] = 1.0
    return scale


def run_abc(obs_summary, obs_R, scale,
            n_draws=5000, accept_frac=0.05,
            rng=None, n_workers=None, verbose=True):
    """
    Rejection ABC returning accepted params, their simulated summaries,
    and their distances — all needed for the regression adjustment.

    accept_frac is 0.05 (5 %) rather than the usual 1 %: the regression
    step corrects the extra bias introduced by the looser tolerance, giving
    more samples at no quality cost.
    """
    if rng is None:
        rng = np.random.default_rng()
    if n_workers is None:
        n_workers = os.cpu_count()

    seeds = rng.integers(0, 2**31, size=n_draws)
    args  = [(obs_summary, obs_R, scale, int(s)) for s in seeds]

    params_list    = []
    summaries_list = []
    dists_list     = []

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for i, (beta, gamma, rho, d, sim_s) in enumerate(
                ex.map(_abc_worker, args, chunksize=50)):
            params_list.append([beta, gamma, rho])
            summaries_list.append(sim_s)
            dists_list.append(d)
            if verbose and (i + 1) % 500 == 0:
                print(f"  {i+1}/{n_draws}")

    all_params    = np.array(params_list)
    all_summaries = np.array(summaries_list)
    all_dists     = np.array(dists_list)

    n_keep   = max(1, int(accept_frac * n_draws))
    keep_idx = np.argsort(all_dists)[:n_keep]

    return (all_params[keep_idx],
            all_summaries[keep_idx],
            all_dists[keep_idx])


# ============================================================
# LOCAL LINEAR REGRESSION ADJUSTMENT  (Beaumont et al. 2002)
# ============================================================

def _logit(x, lo, hi):
    """Map x in (lo, hi) to the real line."""
    p = np.clip((x - lo) / (hi - lo), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def _sigmoid(x, lo, hi):
    """Inverse logit — map real line back to (lo, hi)."""
    return lo + (hi - lo) / (1.0 + np.exp(-x))


def regression_adjust(params, summaries, obs_summary, distances):
    """
    Beaumont, Zhang & Balding (2002) local linear regression adjustment.

    For each parameter θ_j, fits (in logit-transformed space):
        φ_j(s_i) = α_j + β_j^T (s_i − s_obs) + ε_i
    using Epanechnikov-kernel weighted least squares, then adjusts:
        φ_j*(i) = φ_j(i) − β̂_j^T (s_i − s_obs)
    and maps back through the sigmoid to the original bounded space.

    Parameters
    ----------
    params      : (n, 3)  accepted parameter samples [β, γ, ρ]
    summaries   : (n, 7)  accepted simulated summaries (S4)
    obs_summary : (7,)    observed S4 summary
    distances   : (n,)    accepted distances (used for kernel weights)

    Returns
    -------
    adj_params  : (n, 3)  adjusted parameters
    diagnostics : dict
        'r2'        : (3,)  weighted R² of the local fit per parameter
        'shrinkage' : (3,)  std(adjusted) / std(raw) per parameter
    """
    n = len(params)
    delta = summaries - obs_summary[np.newaxis, :]      # (n, 7)

    # Epanechnikov kernel weights
    h = distances.max()
    if h < 1e-12:
        h = 1.0
    u = distances / h
    w = np.maximum(0.0, 0.75 * (1.0 - u ** 2))         # (n,)
    sqrt_w = np.sqrt(w)

    # Design matrix  [1 | delta]  — intercept + summary deviations
    X = np.column_stack([np.ones(n), delta])             # (n, 8)

    adj_params  = np.empty_like(params, dtype=float)
    r2          = np.zeros(3)

    for j, (lo, hi) in enumerate(PRIOR_BOUNDS):
        phi = _logit(params[:, j], lo, hi)               # logit-transform

        # Weighted least squares via row-scaling: min ||sqrt(w) * (Xβ - φ)||²
        Xw  = X   * sqrt_w[:, np.newaxis]
        phiw = phi * sqrt_w
        coef, _, _, _ = np.linalg.lstsq(Xw, phiw, rcond=None)  # (8,)

        beta_vec = coef[1:]                               # (7,) slope terms
        phi_adj  = phi - delta @ beta_vec

        # Weighted R²
        phi_wmean = np.average(phi, weights=w)
        ss_tot = np.sum(w * (phi - phi_wmean) ** 2)
        phi_hat = X @ coef
        ss_res  = np.sum(w * (phi - phi_hat) ** 2)
        r2[j]   = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        adj_params[:, j] = _sigmoid(phi_adj, lo, hi)    # back-transform

    raw_std = params.std(axis=0)
    adj_std = adj_params.std(axis=0)
    shrinkage = adj_std / np.where(raw_std > 1e-12, raw_std, 1.0)

    diagnostics = {"r2": r2, "shrinkage": shrinkage}
    return adj_params, diagnostics


# ============================================================
# PLOTS
# ============================================================

def plot_raw_vs_adjusted(raw_params, adj_params, diagnostics):
    """
    3-panel figure (one per parameter) showing raw ABC posterior (blue)
    overlaid with regression-adjusted posterior (orange).
    Subtitle shows R² and shrinkage for each parameter.
    """
    prior_bounds = {"β": (0.05, 0.50), "γ": (0.02, 0.20), "ρ": (0.0, 0.8)}
    prior_stds   = np.array([(hi - lo) / np.sqrt(12)
                              for lo, hi in PRIOR_BOUNDS])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for j, (param, ax) in enumerate(zip(PARAM_NAMES, axes)):
        lo, hi = PRIOR_BOUNDS[j]

        ax.hist(raw_params[:, j], bins=35, density=True,
                color="steelblue", alpha=0.55, label="Raw ABC")
        ax.hist(adj_params[:, j], bins=35, density=True,
                color="darkorange", alpha=0.55, label="Adjusted")
        ax.axhline(1.0 / (hi - lo), color="gray", linestyle="--",
                   linewidth=1, label="Prior")
        ax.set_xlim(lo, hi)
        ax.set_xlabel(param, fontsize=12)
        ax.set_ylabel("Density", fontsize=10)

        r2  = diagnostics["r2"][j]
        shr = diagnostics["shrinkage"][j]
        raw_std = raw_params[:, j].std()
        adj_std = adj_params[:, j].std()
        ax.set_title(
            f"{param}    R²={r2:.3f}    shrinkage={shr:.2f}\n"
            f"std: {raw_std:.3f} → {adj_std:.3f}",
            fontsize=9,
        )

        if j == 0:
            ax.legend(fontsize=9)

    fig.suptitle(
        "Regression adjustment (Beaumont et al. 2002) — S4 scalar summaries\n"
        f"n_accepted={len(raw_params)}  accept_frac=5%",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig("4a_raw_vs_adjusted.png", dpi=150, bbox_inches="tight")
    print("Saved: 4a_raw_vs_adjusted.png")
    plt.show()


def plot_posterior_widths(raw_params, adj_params):
    """
    Bar chart: posterior std / prior std for raw and adjusted, per parameter.
    """
    prior_stds = np.array([(hi - lo) / np.sqrt(12) for lo, hi in PRIOR_BOUNDS])

    raw_norm = raw_params.std(axis=0) / prior_stds
    adj_norm = adj_params.std(axis=0) / prior_stds

    x      = np.arange(3)
    width  = 0.35
    colors = ["steelblue", "darkorange"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, raw_norm, width, label="Raw ABC",   color=colors[0], alpha=0.85)
    ax.bar(x + width / 2, adj_norm, width, label="Adjusted", color=colors[1], alpha=0.85)
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="Prior (baseline)")
    ax.set_xticks(x)
    ax.set_xticklabels(PARAM_NAMES, fontsize=12)
    ax.set_ylabel("Posterior std / Prior std  (lower = more informative)")
    ax.set_title("Information gain: raw ABC vs regression-adjusted (S4)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("4a_posterior_widths.png", dpi=150, bbox_inches="tight")
    print("Saved: 4a_posterior_widths.png")
    plt.show()


# ============================================================
# MAIN DRIVER
# ============================================================

def run_question4a(
    infected_file="data/infected_timeseries.csv",
    rewiring_file="data/rewiring_timeseries.csv",
    degree_file="data/final_degree_histograms.csv",
    n_draws=5000,
    accept_frac=0.05,
    n_scale_sims=200,
    seed=123,
):
    rng = np.random.default_rng(seed)

    # ---- Load data ----
    print("Loading observed data...")
    infected_df, rewiring_df, degree_df = load_observed_data(
        infected_file, rewiring_file, degree_file
    )
    infected_obs, rewiring_obs, degree_obs = make_observed_arrays(
        infected_df, rewiring_df, degree_df
    )
    obs_R = infected_obs.shape[0]
    print(f"Observed replicates: {obs_R}")

    obs_summary = build_summary_S4(infected_obs, rewiring_obs, degree_obs)
    print(f"S4 observed summary: {obs_summary}")

    # ---- Estimate scale ----
    print(f"\nEstimating scale ({n_scale_sims} prior-predictive sims)...")
    scale = estimate_scale(obs_R=obs_R, n_sims=n_scale_sims, rng=rng)
    print(f"Scale: {scale}")

    # ---- Run rejection ABC (5 % tolerance) ----
    print(f"\nRunning rejection ABC ({n_draws} draws, accept_frac={accept_frac})...")
    raw_params, acc_summaries, acc_distances = run_abc(
        obs_summary=obs_summary, obs_R=obs_R, scale=scale,
        n_draws=n_draws, accept_frac=accept_frac, rng=rng,
    )
    print(f"Accepted {len(raw_params)} samples  "
          f"(distance range: {acc_distances.min():.3f} – {acc_distances.max():.3f})")

    # ---- Regression adjustment ----
    print("\nApplying local linear regression adjustment...")
    adj_params, diag = regression_adjust(
        raw_params, acc_summaries, obs_summary, acc_distances
    )

    print("\nDiagnostics:")
    for j, p in enumerate(PARAM_NAMES):
        print(f"  {p}:  R²={diag['r2'][j]:.3f}  "
              f"shrinkage={diag['shrinkage'][j]:.3f}  "
              f"std {raw_params[:,j].std():.3f} → {adj_params[:,j].std():.3f}")

    print("\nPosterior means:")
    for j, p in enumerate(PARAM_NAMES):
        print(f"  {p}:  raw={raw_params[:,j].mean():.3f}  "
              f"adj={adj_params[:,j].mean():.3f}")

    # ---- Plots ----
    print("\nGenerating plots...")
    plot_raw_vs_adjusted(raw_params, adj_params, diag)
    plot_posterior_widths(raw_params, adj_params)

    return raw_params, adj_params, diag


if __name__ == "__main__":
    run_question4a()
