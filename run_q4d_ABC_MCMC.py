"""
4d. Advanced Methods — Synthetic Likelihood MCMC (Wood 2010)

Key idea: instead of comparing summary statistics through a distance function
and an epsilon threshold, assume that the summary statistics follow a
multivariate normal distribution conditional on the parameters, and estimate
the mean and covariance from repeated simulations. This gives a smooth
approximate likelihood that can be plugged into standard MCMC.

At each MH step:
  1. Propose theta' = theta + Normal(0, sigma^2)       [symmetric random walk]
  2. If theta' outside prior bounds -> reject
  3. Run n_sim groups of obs_R replicates at theta' -> n_sim S4 vectors
  4. Fit MVN: mu_hat, Sigma_hat  from those n_sim vectors
  5. log p_SL(s_obs | theta') = log N(s_obs; mu_hat, Sigma_hat)
  6. Accept theta' with probability min(1, exp(log_SL_proposed - log_SL_current))

No epsilon is needed. The "tolerance" is implicit in n_sim: more groups -> more
accurate covariance estimate -> smoother likelihood surface.

Key difference from 4_ABC_MCMC.py: acceptance is decided by a likelihood ratio,
not a hard distance threshold, so the chain can make large steps and still
accept — leading to better mixing.

Summary set: S4 (7 scalar statistics). The 7x7 covariance matrix is accurately
estimated from n_sim >= 20 groups, making this setting ideal for synthetic
likelihood. Using S3 (73 dimensions) would require n_sim >> 73 groups per step.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
import os


# ============================================================
# SIMULATOR  (identical to 4_ABC_MCMC.py)
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
                s_rew    = s_idx[sub_rs[sel]]
                i_rew    = i_idx[sub_cs[sel]]
                A[s_rew, i_rew] = False
                A[i_rew, s_rew] = False
                for s_node in s_rew:
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


def simulate_replicates(beta, gamma, rho, R, N=200, p_edge=0.05,
                        n_infected0=5, T=200, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    results      = [simulate_fast(beta, gamma, rho, N=N, p_edge=p_edge,
                                  n_infected0=n_infected0, T=T, rng=rng)
                    for _ in range(R)]
    infected_arr = np.array([r[0] for r in results])
    rewiring_arr = np.array([r[1] for r in results])
    degree_arr   = np.array([r[2] for r in results])
    return infected_arr, rewiring_arr, degree_arr


# ============================================================
# DATA LOADING
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
# SUMMARY STATISTICS  (S4 — 7 scalar summaries)
# ============================================================

def build_summary(infected_arr, rewiring_arr, degree_arr):
    """
    7 scalar summaries (S4 from Q3):
      peak infection, time-to-peak (normalised), epidemic AUC,
      total rewirings, peak rewiring rate, mean and std of final degree.
    """
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


# ============================================================
# PRIOR
# ============================================================

BOUNDS = np.array([[0.05, 0.50],   # beta
                   [0.02, 0.20],   # gamma
                   [0.00, 0.80]])  # rho


def in_prior(theta):
    return np.all((theta >= BOUNDS[:, 0]) & (theta <= BOUNDS[:, 1]))


def sample_prior(rng):
    return rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1])


# ============================================================
# SYNTHETIC LIKELIHOOD HELPERS
# ============================================================

def _mvn_logpdf(x, mu, cov):
    """
    Multivariate normal log-pdf via Cholesky decomposition.
    Returns -inf if the covariance matrix is not positive-definite.
    """
    d    = len(x)
    diff = x - mu
    try:
        L       = np.linalg.cholesky(cov)
        log_det = 2.0 * np.sum(np.log(np.diag(L)))
        v       = np.linalg.solve(L, diff)          # L^{-1} (x - mu)
        return -0.5 * (d * np.log(2.0 * np.pi) + log_det + np.dot(v, v))
    except np.linalg.LinAlgError:
        return -np.inf


def _sl_sim_worker(args):
    """
    Top-level worker (must be importable for pickling).
    Run one group of obs_R replicates at theta and return its S4 vector.
    """
    beta, gamma, rho, obs_R, seed = args
    rng = np.random.default_rng(seed)
    inf_arr, rew_arr, deg_arr = simulate_replicates(beta, gamma, rho, R=obs_R, rng=rng)
    return build_summary(inf_arr, rew_arr, deg_arr)


def _eval_sl(executor, theta, obs_summary, obs_R, n_sim, rng, n_workers):
    """
    Estimate the synthetic log-likelihood at theta.

    Runs n_sim groups of obs_R replicates in parallel, fits a MVN to the
    resulting S4 vectors, and evaluates log N(s_obs; mu_hat, Sigma_hat).

    The diagonal of Sigma is regularised by 1e-4 * mean(diag) to guard
    against near-singular covariances when n_sim is small relative to d.
    """
    beta, gamma, rho = theta
    seeds    = rng.integers(0, 2**31, size=n_sim)
    args     = [(beta, gamma, rho, obs_R, int(s)) for s in seeds]
    summaries = np.array(list(executor.map(_sl_sim_worker, args, chunksize=5)))
    # summaries: (n_sim, 7)

    mu  = summaries.mean(axis=0)
    cov = np.cov(summaries, rowvar=False)            # (7, 7)

    # Tikhonov regularisation: add a small fraction of the mean variance
    # along the diagonal so Sigma is always positive-definite
    cov += 1e-4 * np.mean(np.diag(cov)) * np.eye(len(obs_summary))

    return _mvn_logpdf(obs_summary, mu, cov)


# ============================================================
# SYNTHETIC LIKELIHOOD MCMC
# ============================================================

def sl_mcmc(obs_summary, obs_R, n_steps=1000, n_burnin=200,
            n_sim=30, proposal_std=None, seed=42,
            n_workers=None, verbose=True):
    """
    Synthetic likelihood Metropolis-Hastings chain.

    Parameters
    ----------
    obs_summary  : (7,)   observed S4 summary
    obs_R        : int    observed replicates (used for each sim group)
    n_steps      : int    total chain length (including burn-in)
    n_burnin     : int    steps to discard as burn-in
    n_sim        : int    simulation groups per likelihood evaluation
                          (more -> better covariance estimate, more compute)
    proposal_std : (3,) or None  Gaussian random walk std per parameter;
                          defaults to 15% of each prior range
    seed         : int    RNG seed

    Returns
    -------
    posterior    : (n_steps - n_burnin, 3)  post-burnin samples
    chain        : (n_steps, 3)             full chain
    log_liks     : (n_steps,)               cached log-likelihood at each step
    acc_rate     : float
    """
    rng = np.random.default_rng(seed)
    if n_workers is None:
        n_workers = os.cpu_count()

    if proposal_std is None:
        # 4% of each prior range. SL posteriors are much sharper than the
        # ABC-MCMC hard-threshold posterior, so the 15% default causes proposals
        # to jump off the likelihood peak almost every time -> chain sticks.
        # Target acceptance rate ~0.20-0.35.
        proposal_std = 0.04 * (BOUNDS[:, 1] - BOUNDS[:, 0])

    # ---- Initialise from prior ----
    print("  Initialising chain...")
    theta_curr    = sample_prior(rng)

    chain    = np.zeros((n_steps, 3))
    log_liks = np.zeros(n_steps)
    n_accepted = 0

    # Keep one persistent process pool for the whole chain.
    # Re-creating a pool every step is expensive (~0.1s spawn overhead each time).
    with ProcessPoolExecutor(max_workers=n_workers) as executor:

        log_lik_curr = _eval_sl(executor, theta_curr, obs_summary, obs_R, n_sim, rng, n_workers)
        print(f"  Initial log-lik: {log_lik_curr:.4f}  theta: {theta_curr.round(3)}")

        for i in range(n_steps):
            theta_prop = theta_curr + rng.normal(0.0, proposal_std)

            if in_prior(theta_prop):
                log_lik_prop = _eval_sl(executor, theta_prop, obs_summary,
                                        obs_R, n_sim, rng, n_workers)
                # MH ratio: prior ratio = 1 (uniform), proposal ratio = 1 (symmetric)
                log_alpha = log_lik_prop - log_lik_curr
                if np.log(rng.random()) < log_alpha:
                    theta_curr   = theta_prop
                    log_lik_curr = log_lik_prop
                    n_accepted  += 1

            chain[i]    = theta_curr
            log_liks[i] = log_lik_curr

            if verbose and (i + 1) % 100 == 0:
                rate = n_accepted / (i + 1)
                print(f"  Step {i+1:>5}/{n_steps} | acc={rate:.3f} | "
                      f"log_lik={log_lik_curr:.3f} | "
                      f"theta=({theta_curr[0]:.3f}, {theta_curr[1]:.3f}, {theta_curr[2]:.3f})")

    acc_rate  = n_accepted / n_steps
    posterior = chain[n_burnin:]
    return posterior, chain, log_liks, acc_rate


# ============================================================
# REJECTION ABC  (for comparison, same budget)
# ============================================================

def _rej_worker(args):
    obs_summary, obs_R, scale, seed = args
    rng = np.random.default_rng(seed)
    theta = sample_prior(rng)
    inf_arr, rew_arr, deg_arr = simulate_replicates(*theta, R=obs_R, rng=rng)
    sim_s = build_summary(inf_arr, rew_arr, deg_arr)
    d = np.sqrt(np.sum(((sim_s - obs_summary) / scale) ** 2))
    return theta, d


def _scale_worker(args):
    obs_R, seed = args
    rng   = np.random.default_rng(seed)
    theta = sample_prior(rng)
    inf_arr, rew_arr, deg_arr = simulate_replicates(*theta, R=obs_R, rng=rng)
    return build_summary(inf_arr, rew_arr, deg_arr)


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


def rejection_abc(obs_summary, obs_R, scale, accept_frac=0.01,
                  n_draws=2000, rng=None, n_workers=None):
    if rng is None:
        rng = np.random.default_rng()
    if n_workers is None:
        n_workers = os.cpu_count()
    seeds = rng.integers(0, 2**31, size=n_draws)
    args  = [(obs_summary, obs_R, scale, int(s)) for s in seeds]
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        results = list(ex.map(_rej_worker, args, chunksize=50))
    params = np.array([r[0] for r in results])
    dists  = np.array([r[1] for r in results])
    n_keep = max(1, int(accept_frac * n_draws))
    idx    = np.argsort(dists)[:n_keep]
    return params[idx]


# ============================================================
# PLOTS
# ============================================================

def plot_trace(chain, n_burnin, log_liks, param_names=("β", "γ", "ρ")):
    fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

    for j, name in enumerate(param_names):
        axes[j].plot(chain[:, j], lw=0.4, color="steelblue", alpha=0.8)
        axes[j].axvline(n_burnin, color="red", linestyle="--", lw=1,
                        label="burn-in end" if j == 0 else None)
        axes[j].set_ylabel(name)

    axes[3].plot(log_liks, lw=0.4, color="darkorange", alpha=0.8)
    axes[3].axvline(n_burnin, color="red", linestyle="--", lw=1)
    axes[3].set_ylabel("log SL")
    axes[3].set_xlabel("Step")

    axes[0].legend(fontsize=8)
    fig.suptitle("Synthetic Likelihood MCMC — trace plots", fontsize=12)
    plt.tight_layout()
    plt.savefig("4d_trace.png", dpi=150, bbox_inches="tight")
    print("Saved: 4d_trace.png")
    plt.show()


def plot_comparison(sl_post, rej_post, param_names=("β", "γ", "ρ")):
    prior_bounds = [(0.05, 0.50), (0.02, 0.20), (0.0, 0.8)]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for j, (name, (lo, hi)) in enumerate(zip(param_names, prior_bounds)):
        ax = axes[j]
        ax.hist(sl_post[:, j],  bins=30, density=True, alpha=0.55,
                color="steelblue",   label=f"SL-MCMC (n={len(sl_post)})")
        ax.hist(rej_post[:, j], bins=30, density=True, alpha=0.55,
                color="darkorange",  label=f"Rejection ABC (n={len(rej_post)})")
        ax.axhline(1.0 / (hi - lo), color="gray", linestyle="--", lw=1, label="prior")
        ax.set_xlabel(name, fontsize=12)
        ax.set_xlim(lo, hi)
        if j == 0:
            ax.set_ylabel("Density")
        ax.set_title(f"Posterior of {name}")
        ax.legend(fontsize=7)

    fig.suptitle("Synthetic Likelihood MCMC vs Rejection ABC (S4 summaries)", fontsize=11)
    plt.tight_layout()
    plt.savefig("4d_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved: 4d_comparison.png")
    plt.show()


def print_summary(label, samples, param_names=("β", "γ", "ρ")):
    print(f"\n{'='*55}")
    print(f"  {label}  (n = {len(samples)} samples)")
    print(f"{'='*55}")
    for j, name in enumerate(param_names):
        col = samples[:, j]
        print(f"  {name}:  mean={col.mean():.4f}  std={col.std():.4f}  "
              f"95% CI=[{np.percentile(col, 2.5):.4f}, {np.percentile(col, 97.5):.4f}]")


# ============================================================
# MAIN DRIVER
# ============================================================

def run(
    infected_file="data/infected_timeseries.csv",
    rewiring_file="data/rewiring_timeseries.csv",
    degree_file="data/final_degree_histograms.csv",
    n_steps=1000,
    n_burnin=200,
    n_sim=30,
    n_scale_sims=200,
    n_rej_draws=2000,
    seed=42,
):
    """
    n_sim    : simulation groups per likelihood evaluation.
               Each group is obs_R=40 replicate epidemics, so total simulator
               calls per MCMC step = n_sim * obs_R.
               n_sim=30 -> 30*40=1200 calls/step, 1000 steps -> 1.2M calls total.
    """
    rng = np.random.default_rng(seed)

    # ---- Load data ----
    print("Loading observed data...")
    infected_df, rewiring_df, degree_df = load_observed_data(
        infected_file, rewiring_file, degree_file
    )
    infected_obs, rewiring_obs, degree_obs = make_observed_arrays(
        infected_df, rewiring_df, degree_df
    )
    obs_R       = infected_obs.shape[0]
    obs_summary = build_summary(infected_obs, rewiring_obs, degree_obs)
    print(f"  Observed replicates: {obs_R}")
    print(f"  Observed S4 summary: {obs_summary.round(4)}")

    # ---- SL-MCMC ----
    print(f"\nRunning SL-MCMC ({n_steps} steps, burn-in={n_burnin}, n_sim={n_sim})...")
    print(f"  Simulator calls per step: {n_sim} groups × {obs_R} replicates = {n_sim*obs_R}")
    print(f"  Total simulator calls:    ~{n_steps * n_sim * obs_R:,}")

    sl_post, chain, log_liks, acc_rate = sl_mcmc(
        obs_summary, obs_R,
        n_steps=n_steps, n_burnin=n_burnin, n_sim=n_sim,
        seed=seed,
    )
    print(f"\n  SL-MCMC acceptance rate: {acc_rate:.3f}")

    # ---- Rejection ABC (comparison baseline) ----
    print(f"\nRunning rejection ABC ({n_rej_draws} draws) for comparison...")
    scale   = estimate_scale(obs_R, n_sims=n_scale_sims, rng=rng)
    rej_post = rejection_abc(obs_summary, obs_R, scale,
                             accept_frac=0.01, n_draws=n_rej_draws, rng=rng)
    print(f"  Accepted {len(rej_post)} samples")

    # ---- Summaries ----
    print_summary("SL-MCMC (post burn-in)", sl_post)
    print_summary("Rejection ABC", rej_post)

    # ---- Plots ----
    print("\nGenerating plots...")
    plot_trace(chain, n_burnin, log_liks)
    plot_comparison(sl_post, rej_post)

    return {
        "sl_posterior":  sl_post,
        "rej_posterior": rej_post,
        "chain":         chain,
        "log_liks":      log_liks,
        "acc_rate":      acc_rate,
        "scale":         scale,
    }


if __name__ == "__main__":
    results = run()
