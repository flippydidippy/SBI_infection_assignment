"""
Q4: ABC-MCMC  (Marjoram et al., 2003)
--------------------------------------
Reuses from Q2/Q3:
  - simulate_fast (Numba)    → simulator
  - build_summary             → S4 scalar summaries
  - weighted_distance          → normalised Euclidean distance
  - estimate_scale             → prior-predictive scale
  - calibrate_epsilon          → pilot-run tolerance

New in Q4:
  - abc_mcmc()                → the MCMC chain
  - rejection_abc()           → same-budget baseline for comparison
  - diagnostics (trace, ACF, ESS)
  - all plots for the report
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit, prange
import time

# ============================================================
# SIMULATOR  (Numba JIT — same as Q2/Q3)
# ============================================================

@njit
def simulate_fast(beta, gamma, rho, N, p_edge, n_infected0, T):
    # ---- Adjacency list (replaces N×N matrix) ----
    # Inner loops drop from O(N²) to O(N·k), where k = p_edge·N ≈ 10 for
    # default parameters — roughly a 20× algorithmic speedup.
    nbrs = np.full((N, N), np.int32(-1), dtype=np.int32)
    deg  = np.zeros(N, dtype=np.int32)

    # ---- Erdos-Renyi graph ----
    for i in range(N):
        for j in range(i + 1, N):
            if np.random.random() < p_edge:
                nbrs[i, deg[i]] = j
                deg[i] += 1
                nbrs[j, deg[j]] = i
                deg[j] += 1

    # ---- Initial infection ----
    state = np.zeros(N, dtype=np.int8)
    perm = np.random.permutation(N)
    for k in range(n_infected0):
        state[perm[k]] = 1

    infected_fraction = np.zeros(T + 1, dtype=np.float64)
    rewire_counts     = np.zeros(T + 1, dtype=np.int64)
    infected_fraction[0] = n_infected0 / N

    # Pre-allocate S-I edge buffers outside the time loop
    si_s = np.empty(N * N, dtype=np.int32)
    si_i = np.empty(N * N, dtype=np.int32)

    for t in range(1, T + 1):
        inf_count = 0
        for i in range(N):
            if state[i] == 1:
                inf_count += 1
        if inf_count == 0:
            break

        # Count infected neighbours — O(N·k) instead of O(N²)
        inf_nbr = np.zeros(N, dtype=np.int32)
        for i in range(N):
            for ki in range(deg[i]):
                if state[nbrs[i, ki]] == 1:
                    inf_nbr[i] += 1

        # Infection
        for i in range(N):
            if state[i] == 0 and inf_nbr[i] > 0:
                p_inf = 1.0 - (1.0 - beta) ** inf_nbr[i]
                if np.random.random() < p_inf:
                    state[i] = 1

        # Recovery
        for i in range(N):
            if state[i] == 1 and np.random.random() < gamma:
                state[i] = 2

        # Collect S-I edges — O(N·k)
        n_si = 0
        for i in range(N):
            if state[i] == 0:
                for ki in range(deg[i]):
                    j = nbrs[i, ki]
                    if state[j] == 1:
                        si_s[n_si] = i
                        si_i[n_si] = j
                        n_si += 1

        # Rewiring
        rewire_count = 0
        for k in range(n_si):
            if np.random.random() < rho:
                s_node = int(si_s[k])
                i_node = int(si_i[k])

                # Edge may have been removed earlier in this same time step
                edge_exists = False
                for idx in range(deg[s_node]):
                    if nbrs[s_node, idx] == i_node:
                        edge_exists = True
                        break
                if not edge_exists:
                    continue

                # Swap-remove i_node from s_node's list
                for idx in range(deg[s_node]):
                    if nbrs[s_node, idx] == i_node:
                        nbrs[s_node, idx] = nbrs[s_node, deg[s_node] - 1]
                        nbrs[s_node, deg[s_node] - 1] = -1
                        deg[s_node] -= 1
                        break
                # Swap-remove s_node from i_node's list
                for idx in range(deg[i_node]):
                    if nbrs[i_node, idx] == s_node:
                        nbrs[i_node, idx] = nbrs[i_node, deg[i_node] - 1]
                        nbrs[i_node, deg[i_node] - 1] = -1
                        deg[i_node] -= 1
                        break

                # Pick a new non-neighbour by rejection sampling.
                # For sparse graphs (k≈10, N=200) the rejection rate is ~5 %,
                # so we expect ~1 draw on average.
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

    # Final degree histogram — O(N) instead of O(N²)
    degree_histogram = np.zeros(31, dtype=np.int64)
    for i in range(N):
        d = int(min(deg[i], 30))
        degree_histogram[d] += 1

    return infected_fraction, rewire_counts, degree_histogram


@njit(parallel=True)
def _simulate_replicates_parallel(beta, gamma, rho, R, N, p_edge, n_infected0, T):
    """Run R replicates in parallel across CPU cores via Numba prange."""
    infected_arr = np.zeros((R, T + 1))
    rewiring_arr = np.zeros((R, T + 1))
    degree_arr   = np.zeros((R, 31))
    for r in prange(R):
        inf, rew, deg = simulate_fast(beta, gamma, rho, N, p_edge, n_infected0, T)
        infected_arr[r] = inf
        rewiring_arr[r] = rew
        degree_arr[r]   = deg
    return infected_arr, rewiring_arr, degree_arr


def simulate_replicates(beta, gamma, rho, R, N=200, p_edge=0.05,
                        n_infected0=5, T=200, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    np.random.seed(rng.integers(0, 2**31))
    return _simulate_replicates_parallel(beta, gamma, rho, R, N, p_edge, n_infected0, T)


# ============================================================
# DATA LOADING  (same as Q2/Q3)
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
# S4 SUMMARY STATISTICS  (same as Q3)
# ============================================================

def build_summary(infected_arr, rewiring_arr, degree_arr):
    mean_inf  = infected_arr.mean(axis=0)
    peak_inf  = mean_inf.max()
    t_peak    = mean_inf.argmax() / len(mean_inf)
    auc       = mean_inf.mean()
    mean_rew  = rewiring_arr.mean(axis=0)
    total_rew = mean_rew.sum()
    peak_rew  = mean_rew.max()
    deg_bins      = np.arange(31)
    mean_deg_dist = degree_arr.mean(axis=0)
    denom         = mean_deg_dist.sum() if mean_deg_dist.sum() > 0 else 1.0
    mean_deg      = (mean_deg_dist * deg_bins).sum() / denom
    var_deg       = (mean_deg_dist * deg_bins ** 2).sum() / denom - mean_deg ** 2
    std_deg       = np.sqrt(max(0.0, var_deg))
    return np.array([peak_inf, t_peak, auc, total_rew, peak_rew, mean_deg, std_deg])


def weighted_distance(sim_s, obs_s, scale):
    return np.sqrt(np.sum(((sim_s - obs_s) / scale) ** 2))


# ============================================================
# PRIOR  (same as Q2/Q3)
# ============================================================

BOUNDS = np.array([[0.05, 0.50],   # beta
                   [0.02, 0.20],   # gamma
                   [0.00, 0.80]])  # rho


def in_prior(theta):
    return np.all((theta >= BOUNDS[:, 0]) & (theta <= BOUNDS[:, 1]))


def sample_prior(rng):
    return rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1])


# ============================================================
# SCALE & EPSILON  (reused from Q2/Q3)
# ============================================================

def estimate_scale(obs_R, n_sims=200, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    summaries = []
    for i in range(n_sims):
        theta = sample_prior(rng)
        inf_arr, rew_arr, deg_arr = simulate_replicates(*theta, R=obs_R, rng=rng)
        summaries.append(build_summary(inf_arr, rew_arr, deg_arr))
        if (i + 1) % 50 == 0:
            print(f"    scale sim {i+1}/{n_sims}")
    return np.array(summaries).std(axis=0, ddof=1)


def calibrate_epsilon(obs_summary, obs_R, scale, n_pilot=500,
                      quantile=0.005, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    distances = []
    for i in range(n_pilot):
        theta = sample_prior(rng)
        inf_arr, rew_arr, deg_arr = simulate_replicates(*theta, R=obs_R, rng=rng)
        sim_s = build_summary(inf_arr, rew_arr, deg_arr)
        distances.append(weighted_distance(sim_s, obs_summary, scale))
        if (i + 1) % 100 == 0:
            print(f"    pilot sim {i+1}/{n_pilot}")
    return float(np.quantile(distances, quantile))


# ============================================================
# ABC-MCMC  (new for Q4)
# ============================================================

def abc_mcmc(obs_summary, obs_R, scale, epsilon,
             n_steps=20_000, burnin=5_000,
             proposal_std=None, rng=None, verbose=True):
    """
    ABC-MCMC (Marjoram et al. 2003).

    Because the proposal is symmetric and the prior is uniform,
    the MH ratio = 1 inside the prior, so acceptance is purely
    determined by the distance threshold.

    During burn-in the proposal scale is adapted every adapt_interval steps
    to keep the acceptance rate near target_acc (~20 %).  Adaptation stops
    at burn-in end so that detailed balance is not violated post burn-in.
    """
    if rng is None:
        rng = np.random.default_rng()
    if proposal_std is None:
        proposal_std = 0.15 * (BOUNDS[:, 1] - BOUNDS[:, 0])
    proposal_std = proposal_std.copy()

    # Adaptation settings
    # target_acc=0.10: accept at ~10% but with large steps → good mixing.
    # Raising target above actual acceptance would shrink steps to tiny values,
    # causing high autocorrelation (ESS collapse) even at reasonable acceptance.
    adapt_interval = 200          # re-tune every this many burn-in steps
    target_acc     = 0.10         # match observed ~10% acceptance at tight epsilon
    adapt_factor   = 1.3          # multiply/divide std by this amount
    std_lo = 0.03 * (BOUNDS[:, 1] - BOUNDS[:, 0])   # floor: prevent steps shrinking to zero
    std_hi = 0.60 * (BOUNDS[:, 1] - BOUNDS[:, 0])

    # ---- Find a valid starting point (d < epsilon) ----
    print("  Finding starting point...")
    n_tries = 0
    while True:
        theta = sample_prior(rng)
        inf_arr, rew_arr, deg_arr = simulate_replicates(*theta, R=obs_R, rng=rng)
        sim_s = build_summary(inf_arr, rew_arr, deg_arr)
        d_curr = weighted_distance(sim_s, obs_summary, scale)
        n_tries += 1
        if d_curr < epsilon:
            break
        if n_tries % 50 == 0:
            print(f"    {n_tries} tries so far...")
    theta_curr = theta.copy()
    print(f"  Start: theta={np.round(theta_curr, 3)}, d={d_curr:.4f} ({n_tries} tries)")

    # ---- Run chain ----
    chain      = np.zeros((n_steps, 3))
    distances  = np.zeros(n_steps)
    n_acc      = 0
    n_acc_post = 0
    adapt_acc  = 0    # accepts in the current adapt window
    t0 = time.time()

    for i in range(n_steps):
        theta_prop = theta_curr + rng.normal(0.0, proposal_std)

        if not in_prior(theta_prop):
            chain[i]     = theta_curr
            distances[i] = d_curr
        else:
            inf_arr, rew_arr, deg_arr = simulate_replicates(*theta_prop, R=obs_R, rng=rng)
            sim_s  = build_summary(inf_arr, rew_arr, deg_arr)
            d_prop = weighted_distance(sim_s, obs_summary, scale)

            if d_prop < epsilon:
                theta_curr = theta_prop
                d_curr     = d_prop
                n_acc     += 1
                adapt_acc += 1
                if i >= burnin:
                    n_acc_post += 1

            chain[i]     = theta_curr
            distances[i] = d_curr

        # ---- Adapt proposal during burn-in ----
        if i < burnin and (i + 1) % adapt_interval == 0:
            window_acc = adapt_acc / adapt_interval
            if window_acc > target_acc + 0.05:
                proposal_std = np.minimum(proposal_std * adapt_factor, std_hi)
            elif window_acc < target_acc - 0.05:
                proposal_std = np.maximum(proposal_std / adapt_factor, std_lo)
            adapt_acc = 0

        if verbose and (i + 1) % 2000 == 0:
            elapsed = time.time() - t0
            rate    = (i + 1) / elapsed
            eta     = (n_steps - i - 1) / rate
            print(f"  Step {i+1:>6}/{n_steps} | "
                  f"acc={n_acc/(i+1):.3f} | "
                  f"β={theta_curr[0]:.3f} γ={theta_curr[1]:.3f} ρ={theta_curr[2]:.3f} | "
                  f"proposal_std=[{', '.join(f'{s:.3f}' for s in proposal_std)}] | "
                  f"{rate:.1f} steps/s | ETA {eta/60:.1f}min")

    posterior = chain[burnin:]
    acc_rate  = n_acc_post / (n_steps - burnin) if n_steps > burnin else n_acc / n_steps
    print(f"  Done in {(time.time()-t0)/60:.1f}min | post-burnin acc rate: {acc_rate:.3f}")
    print(f"  Final proposal_std: {np.round(proposal_std, 4)}")
    return posterior, chain, distances, acc_rate


# ============================================================
# REJECTION ABC  (same budget, for comparison)
# ============================================================

def rejection_abc(obs_summary, obs_R, scale, n_draws, accept_frac=0.01, rng=None):
    """
    Rejection ABC keeping the top accept_frac of draws by distance (top-k).
    Returns accepted params, their simulated summaries, their distances, and
    the full distance array (needed for epsilon calibration).
    """
    if rng is None:
        rng = np.random.default_rng()
    params, dists, sums = [], [], []
    for i in range(n_draws):
        theta = sample_prior(rng)
        inf_arr, rew_arr, deg_arr = simulate_replicates(*theta, R=obs_R, rng=rng)
        sim_s = build_summary(inf_arr, rew_arr, deg_arr)
        params.append(theta)
        dists.append(weighted_distance(sim_s, obs_summary, scale))
        sums.append(sim_s)
        if (i + 1) % 2000 == 0:
            print(f"    rej draw {i+1}/{n_draws}")
    params = np.array(params)
    dists  = np.array(dists)
    sums   = np.array(sums)
    n_keep = max(1, int(accept_frac * n_draws))
    idx    = np.argsort(dists)[:n_keep]
    return params[idx], sums[idx], dists[idx], dists


# ============================================================
# REGRESSION ADJUSTMENT  (Beaumont et al. 2002)
# ============================================================

def _logit(x, lo, hi):
    p = np.clip((x - lo) / (hi - lo), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def _sigmoid(x, lo, hi):
    return lo + (hi - lo) / (1.0 + np.exp(-x))


def regression_adjust(params, summaries, obs_summary, distances):
    """
    Beaumont, Zhang & Balding (2002) local linear regression adjustment.
    Fits a weighted local linear model mapping summary deviations to parameters
    (in logit space), then shifts each sample to where it would land if its
    summary had been exactly obs_summary.
    """
    n     = len(params)
    delta = summaries - obs_summary[np.newaxis, :]
    h     = distances.max()
    if h < 1e-12:
        h = 1.0
    u      = distances / h
    w      = np.maximum(0.0, 0.75 * (1.0 - u ** 2))   # Epanechnikov kernel
    sqrt_w = np.sqrt(w)
    X      = np.column_stack([np.ones(n), delta])

    adj_params = np.empty_like(params, dtype=float)
    r2         = np.zeros(3)

    for j, (lo, hi) in enumerate(BOUNDS):
        phi  = _logit(params[:, j], lo, hi)
        Xw   = X   * sqrt_w[:, np.newaxis]
        phiw = phi * sqrt_w
        coef, _, _, _ = np.linalg.lstsq(Xw, phiw, rcond=None)
        beta_vec = coef[1:]
        phi_adj  = phi - delta @ beta_vec

        phi_wmean = np.average(phi, weights=w)
        ss_tot = np.sum(w * (phi - phi_wmean) ** 2)
        ss_res = np.sum(w * (phi - X @ coef) ** 2)
        r2[j]  = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        adj_params[:, j] = _sigmoid(phi_adj, lo, hi)

    raw_std   = params.std(axis=0)
    adj_std   = adj_params.std(axis=0)
    shrinkage = adj_std / np.where(raw_std > 1e-12, raw_std, 1.0)
    diagnostics = {"r2": r2, "shrinkage": shrinkage}
    return adj_params, diagnostics


# ============================================================
# DIAGNOSTICS
# ============================================================

def autocorrelation(x, max_lag=200):
    x = x - x.mean()
    n = len(x)
    var = np.var(x)
    if var < 1e-12:
        return np.zeros(max_lag + 1)
    acf = np.correlate(x, x, mode="full")[n - 1:]
    return acf[:max_lag + 1] / (var * n)


def effective_sample_size(x):
    n = len(x)
    acf = autocorrelation(x, max_lag=min(n // 2, 500))
    tau = 0.0
    for k in range(1, len(acf) - 1, 2):
        pair_sum = acf[k] + (acf[k + 1] if k + 1 < len(acf) else 0.0)
        if pair_sum < 0:
            break
        tau += pair_sum
    return n / (1.0 + 2.0 * tau)


# ============================================================
# PLOTS
# ============================================================

def plot_trace(chain, burnin, save="trace_plots.png"):
    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    names = ("β", "γ", "ρ")
    for j, name in enumerate(names):
        axes[j].plot(chain[:, j], lw=0.4, color="steelblue", alpha=0.8)
        axes[j].axvline(burnin, color="red", ls="--", lw=1, label="burn-in end")
        axes[j].set_ylabel(name)
        if j == 0:
            axes[j].legend(fontsize=8)
    axes[-1].set_xlabel("Step")
    fig.suptitle("ABC-MCMC trace plots", fontsize=12)
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save}")


def plot_distance_trace(distances, epsilon, burnin, save="distance_trace.png"):
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(distances, lw=0.4, color="darkorange", alpha=0.8)
    ax.axhline(epsilon, color="red", ls="--", lw=1, label=f"ε = {epsilon:.4f}")
    ax.axvline(burnin,  color="black", ls="--", lw=1, label="burn-in end")
    ax.set(xlabel="Step", ylabel="Distance",
           title="Distance trace (all values ≤ ε were accepted)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save}")


def plot_autocorrelation(posterior, max_lag=200, save="autocorrelation.png"):
    names = ("β", "γ", "ρ")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3))
    for j, name in enumerate(names):
        acf = autocorrelation(posterior[:, j], max_lag=max_lag)
        axes[j].bar(range(len(acf)), acf, width=1.0, color="steelblue", alpha=0.7)
        axes[j].axhline(0, color="black", lw=0.5)
        axes[j].set_xlabel("Lag")
        axes[j].set_title(f"ACF of {name}")
        if j == 0:
            axes[j].set_ylabel("Autocorrelation")
    fig.suptitle("ABC-MCMC autocorrelation", fontsize=11)
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save}")


def _simulate_pp(post, obs_R, n_pp_samples, rng):
    """Simulate posterior predictive trajectories from a posterior array."""
    idx = rng.choice(len(post), size=min(n_pp_samples, len(post)), replace=False)
    pp_params = post[idx]
    T = 200
    pp_inf = np.zeros((len(idx), T + 1))
    pp_rew = np.zeros((len(idx), T + 1))
    pp_deg = np.zeros((len(idx), 31))
    for k, (beta, gamma, rho) in enumerate(pp_params):
        inf_arr, rew_arr, deg_arr = simulate_replicates(beta, gamma, rho, R=obs_R, rng=rng)
        pp_inf[k] = inf_arr.mean(axis=0)
        pp_rew[k] = rew_arr.mean(axis=0)
        pp_deg[k] = deg_arr.mean(axis=0)
    return pp_inf, pp_rew, pp_deg


def plot_posterior_predictive(mcmc_post, rej_post, inf_obs, rew_obs, deg_obs,
                              obs_R, n_pp_samples=50, seed=0,
                              save="posterior_predictive.png"):
    """
    Two-row posterior predictive check:
      Row 1 — Rejection ABC predictive bands vs observed
      Row 2 — ABC-MCMC predictive bands vs observed
    Tighter bands in row 2 show MCMC produces better (more precise) estimates.
    """
    rng = np.random.default_rng(seed)

    print(f"  Simulating {n_pp_samples} rejection ABC posterior-predictive draws...")
    rej_inf, rej_rew, rej_deg = _simulate_pp(rej_post, obs_R, n_pp_samples, rng)

    print(f"  Simulating {n_pp_samples} ABC-MCMC posterior-predictive draws...")
    mcmc_inf, mcmc_rew, mcmc_deg = _simulate_pp(mcmc_post, obs_R, n_pp_samples, rng)

    T = inf_obs.shape[1] - 1
    t = np.arange(T + 1)
    deg_bins = np.arange(31)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    titles   = ["Infected fraction", "Rewiring time series", "Final degree distribution"]
    colors   = ["steelblue", "darkorange", "seagreen"]
    row_data = [
        (rej_inf,  rej_rew,  rej_deg,  "Rejection ABC"),
        (mcmc_inf, mcmc_rew, mcmc_deg, "ABC-MCMC"),
    ]

    for row, (pp_i, pp_r, pp_d, label) in enumerate(row_data):
        # Panel 1: infected fraction
        ax = axes[row, 0]
        for obs_row in inf_obs:
            ax.plot(t, obs_row, color="gray", lw=0.5, alpha=0.3)
        lo, hi = np.percentile(pp_i, 5, axis=0), np.percentile(pp_i, 95, axis=0)
        ax.fill_between(t, lo, hi, alpha=0.4, color=colors[0], label="PP 5–95%")
        ax.plot(t, pp_i.mean(axis=0), color=colors[0], lw=1.5, label="PP mean")
        ax.set(xlabel="Time", ylabel="Infected fraction",
               title=f"{label} — {titles[0]}")
        ax.legend(fontsize=8)

        # Panel 2: rewiring
        ax = axes[row, 1]
        for obs_row in rew_obs:
            ax.plot(t, obs_row, color="gray", lw=0.5, alpha=0.3)
        lo, hi = np.percentile(pp_r, 5, axis=0), np.percentile(pp_r, 95, axis=0)
        ax.fill_between(t, lo, hi, alpha=0.4, color=colors[1], label="PP 5–95%")
        ax.plot(t, pp_r.mean(axis=0), color=colors[1], lw=1.5, label="PP mean")
        ax.set(xlabel="Time", ylabel="Rewire count",
               title=f"{label} — {titles[1]}")
        ax.legend(fontsize=8)

        # Panel 3: degree histogram
        ax = axes[row, 2]
        obs_mean_deg = deg_obs.mean(axis=0)
        ax.bar(deg_bins, obs_mean_deg, color="gray", alpha=0.5, label="Observed mean")
        lo, hi = np.percentile(pp_d, 5, axis=0), np.percentile(pp_d, 95, axis=0)
        ax.fill_between(deg_bins - 0.5, lo, hi, alpha=0.4, color=colors[2],
                        step="post", label="PP 5–95%")
        ax.step(deg_bins - 0.5, pp_d.mean(axis=0), color=colors[2], lw=1.5,
                where="post", label="PP mean")
        ax.set(xlabel="Degree", ylabel="Node count",
               title=f"{label} — {titles[2]}")
        ax.legend(fontsize=8)

    fig.suptitle(
        "Posterior predictive check — Rejection ABC (top) vs ABC-MCMC (bottom)\n"
        "Tighter bands in bottom row show ABC-MCMC achieves more precise parameter estimates",
        fontsize=11)
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save}")


def plot_raw_vs_adjusted(raw_params, adj_params, diagnostics,
                         save="4a_raw_vs_adjusted.png"):
    """Raw ABC vs regression-adjusted posteriors with R² and shrinkage."""
    names = ["β", "γ", "ρ"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for j, (name, ax) in enumerate(zip(names, axes)):
        lo, hi = BOUNDS[j]
        ax.hist(raw_params[:, j], bins=35, density=True,
                color="steelblue", alpha=0.55, label="Raw ABC (5%)")
        ax.hist(adj_params[:, j], bins=35, density=True,
                color="darkorange", alpha=0.55, label="Adjusted")
        ax.axhline(1.0 / (hi - lo), color="gray", ls="--", lw=1, label="Prior")
        ax.set_xlim(lo, hi)
        ax.set_xlabel(name, fontsize=12)
        ax.set_ylabel("Density", fontsize=10)
        r2  = diagnostics["r2"][j]
        shr = diagnostics["shrinkage"][j]
        ax.set_title(
            f"{name}    R²={r2:.3f}    shrinkage={shr:.2f}\n"
            f"std: {raw_params[:,j].std():.3f} → {adj_params[:,j].std():.3f}",
            fontsize=9)
        if j == 0:
            ax.legend(fontsize=9)
    fig.suptitle(
        "Regression adjustment (Beaumont et al. 2002) — S4 scalar summaries\n"
        f"n_accepted={len(raw_params)}  accept_frac=5%",
        fontsize=11)
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save}")


def plot_posterior_widths(raw_params, adj_params, mcmc_post,
                          save="4a_posterior_widths.png"):
    """Bar chart: posterior std / prior std for all three methods."""
    prior_stds = np.array([(hi - lo) / np.sqrt(12) for lo, hi in BOUNDS])
    raw_norm   = raw_params.std(axis=0) / prior_stds
    adj_norm   = adj_params.std(axis=0) / prior_stds
    mcmc_norm  = mcmc_post.std(axis=0)  / prior_stds

    names  = ["β", "γ", "ρ"]
    x      = np.arange(3)
    width  = 0.25
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width, raw_norm,  width, label="Rejection ABC (5%)", color="steelblue",  alpha=0.85)
    ax.bar(x,         adj_norm,  width, label="Reg. adjusted",       color="darkorange", alpha=0.85)
    ax.bar(x + width, mcmc_norm, width, label="ABC-MCMC",            color="seagreen",   alpha=0.85)
    ax.axhline(1.0, color="black", ls="--", lw=1, label="Prior (baseline)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=12)
    ax.set_ylabel("Posterior std / Prior std  (lower = more informative)")
    ax.set_title("Information gain: all three advanced methods (S4)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save}")


def plot_beta_rho_correlation(rej_post, adj_post, mcmc_post,
                               save="beta_rho_correlation.png"):
    """β–ρ joint scatter for all three methods with correlation coefficient."""
    methods = [
        ("Rejection ABC (1%)",   rej_post,  "darkorange"),
        ("Reg. adjustment (5%)", adj_post,  "seagreen"),
        ("ABC-MCMC",             mcmc_post, "steelblue"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (label, post, color) in zip(axes, methods):
        ax.scatter(post[:, 0], post[:, 2], s=8, alpha=0.4, color=color)
        corr = np.corrcoef(post[:, 0], post[:, 2])[0, 1]
        ax.set_xlim(0.05, 0.50)
        ax.set_ylim(0.00, 0.80)
        ax.set_xlabel("β", fontsize=12)
        ax.set_ylabel("ρ", fontsize=12)
        ax.set_title(f"{label}\nr(β, ρ) = {corr:.3f}")
    fig.suptitle("Joint β–ρ posterior — all three advanced methods", fontsize=11)
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save}")


def plot_comparison(rej_post, adj_post, mcmc_post,
                    save="mcmc_vs_rejection.png"):
    """
    Single-row KDE comparison of all three advanced methods:
      - Rejection ABC (top 1%)
      - Regression adjustment (top 5%, Beaumont 2002)
      - ABC-MCMC (tighter epsilon, Marjoram 2003)
    """
    from scipy.stats import gaussian_kde

    names  = ("β", "γ", "ρ")
    bounds = ((0.05, 0.50), (0.02, 0.20), (0.0, 0.8))
    methods = [
        ("Rejection ABC (1%)",  rej_post,  "darkorange"),
        ("Reg. adjustment (5%)", adj_post, "seagreen"),
        ("ABC-MCMC",            mcmc_post, "steelblue"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for j, (name, (lo, hi)) in enumerate(zip(names, bounds)):
        ax = axes[j]
        xs = np.linspace(lo, hi, 300)
        ax.axhline(1.0 / (hi - lo), color="gray", ls=":", lw=1, label="Prior")

        for label, post, color in methods:
            if len(post) > 1:
                kde = gaussian_kde(post[:, j], bw_method="scott")
                ys  = kde(xs)
                ax.plot(xs, ys, lw=2, color=color, label=label)
                ax.fill_between(xs, ys, alpha=0.15, color=color)

        ax.set_xlim(lo, hi)
        ax.set_xlabel(name, fontsize=12)
        ax.set_ylabel("Density")
        ax.set_title(f"Posterior of {name}")
        if j == 0:
            ax.legend(fontsize=8)

    fig.suptitle("Advanced methods comparison — Rejection ABC, Regression adjustment, ABC-MCMC",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save}")


# ============================================================
# MAIN
# ============================================================

def run(
    infected_file="data/infected_timeseries.csv",
    rewiring_file="data/rewiring_timeseries.csv",
    degree_file="data/final_degree_histograms.csv",
    # -- Tuneable parameters --
    n_scale_sims=200,           # for estimating normalisation scale
    n_pilot=500,               # for calibrating epsilon
    epsilon_quantile=0.002,    # 0.2th percentile for MCMC — much tighter than rejection ABC's 1%
    n_steps=20_000,            # ABC-MCMC chain length
    burnin=5_000,
    n_rej_draws=5_000,         # rejection ABC budget (matches Q3)
    n_rej_keep=5_000,          # keep top 1% = 50 samples
    seed=42,
):
    rng = np.random.default_rng(seed)

    # ---- 0. JIT warmup ----
    print("Compiling simulator (one-time Numba JIT)...")
    np.random.seed(0)
    _ = simulate_fast(0.2, 0.08, 0.3, 50, 0.05, 3, 10)
    _ = _simulate_replicates_parallel(0.2, 0.08, 0.3, 2, 50, 0.05, 3, 10)
    print("  Done.\n")

    # ---- 1. Load data ----
    print("Loading observed data...")
    inf_df, rew_df, deg_df = load_observed_data(infected_file, rewiring_file, degree_file)
    inf_obs, rew_obs, deg_obs = make_observed_arrays(inf_df, rew_df, deg_df)
    obs_R = inf_obs.shape[0]
    obs_summary = build_summary(inf_obs, rew_obs, deg_obs)
    print(f"  R={obs_R} replicates, summary={np.round(obs_summary, 4)}\n")

    # ---- 2. Estimate scale ----
    print(f"Estimating scale ({n_scale_sims} prior-predictive sims)...")
    scale = estimate_scale(obs_R, n_sims=n_scale_sims, rng=rng)
    scale[scale < 1e-8] = 1.0
    print(f"  scale={np.round(scale, 4)}\n")

    # ---- 3a. Rejection ABC top 1% (Q3 baseline + epsilon for MCMC) ----
    print(f"Running rejection ABC ({n_rej_draws} draws, top 1%)...")
    rej_post, _, rej_dists_acc, _ = rejection_abc(
        obs_summary, obs_R, scale, n_draws=n_rej_draws, accept_frac=0.01, rng=rng)
    epsilon_rej  = float(rej_dists_acc[-1])
    epsilon_mcmc = epsilon_rej / 2.0
    print(f"  {len(rej_post)} samples  (epsilon_rej={epsilon_rej:.4f})")
    print(f"  epsilon_mcmc = {epsilon_mcmc:.4f}  (half of epsilon_rej)\n")

    # ---- 3b. Rejection ABC top 5% + regression adjustment (Q4a, separate draws) ----
    print(f"Running rejection ABC ({n_rej_draws} draws, top 5%) for regression adjustment...")
    raw5_post, raw5_sums, raw5_dists, _ = rejection_abc(
        obs_summary, obs_R, scale, n_draws=n_rej_draws, accept_frac=0.05, rng=rng)
    print(f"  Applying regression adjustment ({len(raw5_post)} raw samples)...")
    adj_post, reg_diag = regression_adjust(raw5_post, raw5_sums, obs_summary, raw5_dists)
    print(f"  R²={np.round(reg_diag['r2'], 3)}  shrinkage={np.round(reg_diag['shrinkage'], 3)}\n")

    # ---- 4. ABC-MCMC (tight epsilon) ----
    print(f"Running ABC-MCMC ({n_steps} steps, burn-in={burnin}, epsilon={epsilon_mcmc:.4f})...")
    mcmc_post, chain, distances, mcmc_acc = abc_mcmc(
        obs_summary, obs_R, scale, epsilon_mcmc,
        n_steps=n_steps, burnin=burnin, rng=rng)

    n_rej = n_rej_draws
    print(f"  Rejection ABC: {len(rej_post)} samples (top 1%)\n")

    # ---- 6. Print results ----
    names = ("β", "γ", "ρ")
    mcmc_ess = [effective_sample_size(mcmc_post[:, j]) for j in range(3)]
    print("=" * 75)
    print(f"  ABC-MCMC posterior  (post burn-in, {len(mcmc_post)} steps)")
    print("=" * 75)
    for j, name in enumerate(names):
        col = mcmc_post[:, j]
        ess = mcmc_ess[j]
        print(f"  {name}:  mean={col.mean():.4f}  std={col.std():.4f}  "
              f"95%CI=[{np.percentile(col,2.5):.4f}, {np.percentile(col,97.5):.4f}]  "
              f"ESS={ess:.0f}  (ESS/n={ess/len(col):.3f})")

    if len(rej_post) > 0:
        print()
        print("=" * 75)
        print(f"  Rejection ABC posterior  ({len(rej_post)} accepted / {n_steps} draws"
              f" = {len(rej_post)/n_steps:.3f} acceptance rate)")
        print("=" * 75)
        for j, name in enumerate(names):
            col = rej_post[:, j]
            # Rejection samples are independent so ESS = n
            print(f"  {name}:  mean={col.mean():.4f}  std={col.std():.4f}  "
                  f"95%CI=[{np.percentile(col,2.5):.4f}, {np.percentile(col,97.5):.4f}]  "
                  f"ESS={len(col)}  (ESS/n=1.000, independent)")
        print()
        total_mcmc_ess = sum(mcmc_ess) / 3
        print(f"  >> Mean ESS:  ABC-MCMC={total_mcmc_ess:.0f}  "
              f"Rejection={len(rej_post):.0f}  "
              f"(MCMC advantage = {total_mcmc_ess/len(rej_post):.2f}×)")
    else:
        print("  Rejection ABC: 0 accepted samples.")

    # ---- 7. Plots ----
    print("\nGenerating plots...")
    plot_autocorrelation(mcmc_post)
    plot_raw_vs_adjusted(raw5_post, adj_post, reg_diag)
    plot_posterior_widths(raw5_post, adj_post, mcmc_post)
    plot_comparison(rej_post, adj_post, mcmc_post)
    plot_beta_rho_correlation(rej_post, adj_post, mcmc_post)
    plot_posterior_predictive(mcmc_post, rej_post, inf_obs, rew_obs, deg_obs, obs_R)
    print("\nAll done.")

    return dict(mcmc_posterior=mcmc_post, rej_posterior=rej_post,
                adj_posterior=adj_post, raw5_posterior=raw5_post,
                chain=chain, distances=distances,
                epsilon_mcmc=epsilon_mcmc, epsilon_rej=epsilon_rej,
                scale=scale, acc_rate=mcmc_acc)


if __name__ == "__main__":
    results = run()