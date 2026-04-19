"""
4. Advanced Methods — ABC-MCMC
Following Marjoram et al. (2003).

Key idea: instead of drawing proposals blindly from the prior (rejection ABC),
run a Markov chain whose stationary distribution is the ABC posterior.

At each step:
  1. Propose theta' = theta + Normal(0, sigma^2)   [symmetric Gaussian random walk]
  2. If theta' is outside prior bounds -> reject (stay at theta)
  3. Simulate under theta', compute distance d'
  4. If d' < epsilon -> accept (move to theta')
  5. Else           -> reject (stay at theta)

Because the proposal is symmetric and the prior is uniform, the MH ratio
simplifies to 1 whenever theta' is in the prior support, so only the
distance threshold matters for acceptance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# ============================================================
# SIMULATOR
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
            A_sub = A[np.ix_(s_idx, i_idx)]
            sub_rs, sub_cs = np.where(A_sub)
            if len(sub_rs) > 0:
                sel = rng.random(len(sub_rs)) < rho
                s_rew = s_idx[sub_rs[sel]]
                i_rew = i_idx[sub_cs[sel]]
                A[s_rew, i_rew] = False
                A[i_rew, s_rew] = False
                for s_node in s_rew:
                    cand_mask = ~A[s_node]
                    cand_mask[s_node] = False
                    candidates = all_nodes[cand_mask]
                    if candidates.size > 0:
                        new_partner = rng.choice(candidates)
                        A[s_node, new_partner] = True
                        A[new_partner, s_node] = True
                        rewire_count += 1

        infected_fraction[t] = (state == 1).sum() / N
        rewire_counts[t]      = rewire_count

    degrees = A.sum(axis=1)
    degree_histogram = np.bincount(np.minimum(degrees, 30), minlength=31)
    return infected_fraction, rewire_counts, degree_histogram


def simulate_replicates(beta, gamma, rho, R, N=200, p_edge=0.05, n_infected0=5, T=200, rng=None):
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
# SUMMARY STATISTICS  (S4 — best from Q3)
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


def weighted_distance(sim_s, obs_s, scale):
    z = (sim_s - obs_s) / scale
    return np.sqrt(np.sum(z ** 2))


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
# STEP 1: ESTIMATE SCALE
# ============================================================

def estimate_scale(obs_R, n_sims=200, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    summaries = []
    for _ in range(n_sims):
        theta = sample_prior(rng)
        inf_arr, rew_arr, deg_arr = simulate_replicates(*theta, R=obs_R, rng=rng)
        summaries.append(build_summary(inf_arr, rew_arr, deg_arr))
    summaries = np.array(summaries)
    scale = summaries.std(axis=0, ddof=1)
    scale[scale < 1e-8] = 1.0
    return scale


# ============================================================
# STEP 2: CALIBRATE EPSILON via pilot rejection ABC
# ============================================================

def calibrate_epsilon(obs_summary, obs_R, scale, n_pilot=500,
                      quantile=0.10, rng=None):
    """
    Run a small pilot rejection ABC and set epsilon to the
    `quantile`-th percentile of observed distances.
    A quantile of 0.10 means we target ~10% acceptance in the pilot,
    which is tighter than the 1% in Q2 but reachable by the chain.
    """
    if rng is None:
        rng = np.random.default_rng()
    distances = []
    for _ in range(n_pilot):
        theta = sample_prior(rng)
        inf_arr, rew_arr, deg_arr = simulate_replicates(*theta, R=obs_R, rng=rng)
        sim_s = build_summary(inf_arr, rew_arr, deg_arr)
        distances.append(weighted_distance(sim_s, obs_summary, scale))
    return float(np.quantile(distances, quantile))


# ============================================================
# STEP 3: ABC-MCMC
# ============================================================

def abc_mcmc(obs_summary, obs_R, scale, epsilon,
             n_steps=10_000, burnin=1_000,
             theta_init=None, proposal_std=None, rng=None, verbose=True):
    """
    ABC-MCMC chain (Marjoram et al. 2003).

    Returns
    -------
    posterior : ndarray (n_steps - burnin, 3)   post-burnin samples
    chain     : ndarray (n_steps, 3)             full chain
    distances : ndarray (n_steps,)               distance at each step
    acc_rate  : float                            overall acceptance rate
    """
    if rng is None:
        rng = np.random.default_rng()

    if proposal_std is None:
        # 4% of each prior range — small steps to avoid sticking given the
        # sharp SL-like posterior. 15% caused near-zero acceptance rates.
        proposal_std = 0.08 * (BOUNDS[:, 1] - BOUNDS[:, 0])

    # ---- Initialise ----
    if theta_init is not None:
        # Warm start from a provided point (e.g. S4 posterior mean from Q3).
        # Still simulate once to get the initial distance.
        print(f"  Starting from provided theta_init: {theta_init.round(3)}")
        theta_curr = np.array(theta_init, dtype=float)
        inf_arr, rew_arr, deg_arr = simulate_replicates(*theta_curr, R=obs_R, rng=rng)
        sim_s  = build_summary(inf_arr, rew_arr, deg_arr)
        d_curr = weighted_distance(sim_s, obs_summary, scale)
        print(f"  Initial distance: {d_curr:.4f}  (epsilon={epsilon:.4f})")
        if d_curr >= epsilon:
            print("  Warning: theta_init has d >= epsilon, chain will not move until "
                  "a proposal lands inside. Consider a tighter theta_init or larger epsilon.")
    else:
        print("  Searching for valid starting point...")
        n_init_tries = 0
        while True:
            theta = sample_prior(rng)
            inf_arr, rew_arr, deg_arr = simulate_replicates(*theta, R=obs_R, rng=rng)
            sim_s  = build_summary(inf_arr, rew_arr, deg_arr)
            d_curr = weighted_distance(sim_s, obs_summary, scale)
            n_init_tries += 1
            if d_curr < epsilon:
                break
            if n_init_tries % 50 == 0:
                print(f"    ... {n_init_tries} tries, best distance so far: {d_curr:.4f}")
        theta_curr = theta.copy()
    print(f"  Started at theta={theta_curr}, d={d_curr:.4f}")

    chain     = np.zeros((n_steps, 3))
    distances = np.zeros(n_steps)
    n_accepted = 0

    for i in range(n_steps):
        # Propose
        theta_prop = theta_curr + rng.normal(0.0, proposal_std)

        if not in_prior(theta_prop):
            # Reflect off prior boundary (keeps chain from getting stuck at edges)
            chain[i]     = theta_curr
            distances[i] = d_curr
            continue

        # Simulate
        inf_arr, rew_arr, deg_arr = simulate_replicates(*theta_prop, R=obs_R, rng=rng)
        sim_s  = build_summary(inf_arr, rew_arr, deg_arr)
        d_prop = weighted_distance(sim_s, obs_summary, scale)

        # Accept / reject
        if d_prop < epsilon:
            theta_curr = theta_prop
            d_curr     = d_prop
            n_accepted += 1

        chain[i]     = theta_curr
        distances[i] = d_curr

        if verbose and (i + 1) % 2_000 == 0:
            print(f"  Step {i+1:>6}/{n_steps} | "
                  f"acc rate: {n_accepted/(i+1):.3f} | "
                  f"d_curr: {d_curr:.4f} | "
                  f"theta: beta={theta_curr[0]:.3f} "
                  f"gamma={theta_curr[1]:.3f} "
                  f"rho={theta_curr[2]:.3f}")

    acc_rate  = n_accepted / n_steps
    posterior = chain[burnin:]
    return posterior, chain, distances, acc_rate


# ============================================================
# STEP 4: REJECTION ABC  (same budget, for comparison)
# ============================================================

def rejection_abc(obs_summary, obs_R, scale, n_draws, accept_frac=0.01, rng=None):
    """
    Rejection ABC keeping the top accept_frac of draws by distance.
    Uses the same n_draws as the MCMC chain for a fair budget comparison.
    """
    if rng is None:
        rng = np.random.default_rng()

    params, dists = [], []
    for _ in range(n_draws):
        theta = sample_prior(rng)
        inf_arr, rew_arr, deg_arr = simulate_replicates(*theta, R=obs_R, rng=rng)
        sim_s = build_summary(inf_arr, rew_arr, deg_arr)
        d = weighted_distance(sim_s, obs_summary, scale)
        params.append(theta)
        dists.append(d)

    params = np.array(params)
    dists  = np.array(dists)
    n_keep = max(1, int(accept_frac * n_draws))
    idx    = np.argsort(dists)[:n_keep]
    return params[idx], n_keep / n_draws


# ============================================================
# PLOTS
# ============================================================

def plot_posterior(samples, param_names=("β", "γ", "ρ")):
    prior_bounds = [(0.05, 0.50), (0.02, 0.20), (0.0, 0.8)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for j, (name, (lo, hi)) in enumerate(zip(param_names, prior_bounds)):
        axes[j].hist(samples[:, j], bins=30, density=True,
                     color="steelblue", alpha=0.75)
        axes[j].axhline(1.0 / (hi - lo), color="gray", linestyle="--",
                        lw=1, label="prior")
        axes[j].set_xlim(lo, hi)
        axes[j].set_xlabel(name, fontsize=12)
        axes[j].set_ylabel("Density")
        axes[j].set_title(f"Posterior of {name}")
        axes[j].legend(fontsize=8)
    fig.suptitle(f"ABC-MCMC posterior (n={len(samples)} post-burnin samples)", fontsize=11)
    plt.tight_layout()
    plt.savefig("mcmc_posterior.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_trace(chain, burnin, param_names=("β", "γ", "ρ")):
    fig, axes = plt.subplots(3, 1, figsize=(12, 6), sharex=True)
    for j, name in enumerate(param_names):
        axes[j].plot(chain[:, j], lw=0.4, color="steelblue", alpha=0.8)
        axes[j].axvline(burnin, color="red", linestyle="--", lw=1, label="burn-in end")
        axes[j].set_ylabel(name)
        if j == 0:
            axes[j].legend(fontsize=8)
    axes[-1].set_xlabel("Step")
    fig.suptitle("ABC-MCMC trace plots", fontsize=12)
    plt.tight_layout()
    plt.savefig("trace_plots.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_distance_trace(distances, epsilon, burnin):
    _, ax = plt.subplots(figsize=(12, 3))
    ax.plot(distances, lw=0.4, color="darkorange", alpha=0.8)
    ax.axhline(epsilon, color="red", linestyle="--", lw=1, label=f"ε = {epsilon:.4f}")
    ax.axvline(burnin,  color="black", linestyle="--", lw=1, label="burn-in end")
    ax.set_xlabel("Step")
    ax.set_ylabel("Distance")
    ax.set_title("Distance trace (all values ≤ ε were accepted)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("distance_trace.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_comparison(mcmc_post, rej_post, param_names=("β", "γ", "ρ"),
                    prior_bounds=((0.05, 0.50), (0.02, 0.20), (0.0, 0.8))):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for j, (name, (lo, hi)) in enumerate(zip(param_names, prior_bounds)):
        ax = axes[j]
        prior_density = 1.0 / (hi - lo)

        for label, samples, color in [
            ("ABC-MCMC",     mcmc_post, "steelblue"),
            ("Rejection ABC", rej_post, "darkorange"),
        ]:
            if len(samples) > 0:
                ax.hist(samples[:, j], bins=30, density=True,
                        alpha=0.55, color=color, label=f"{label} (n={len(samples)})")

        ax.axhline(prior_density, color="gray", linestyle="--", lw=1, label="prior")
        ax.set_xlabel(name)
        ax.set_xlim(lo, hi)
        if j == 0:
            ax.set_ylabel("Density")
        ax.set_title(f"Posterior of {name}")
        ax.legend(fontsize=7)

    fig.suptitle("ABC-MCMC vs Rejection ABC (same ε, same simulation budget)", fontsize=11)
    plt.tight_layout()
    plt.savefig("mcmc_vs_rejection.png", dpi=150, bbox_inches="tight")
    plt.show()


def print_summary(label, samples, param_names=("β", "γ", "ρ")):
    print(f"\n{'='*50}")
    print(f"  {label}  (n = {len(samples)} samples)")
    print(f"{'='*50}")
    if len(samples) == 0:
        print("  No accepted samples.")
        return
    for j, name in enumerate(param_names):
        col = samples[:, j]
        print(f"  {name}:  mean={col.mean():.4f}  std={col.std():.4f}  "
              f"95% CI=[{np.percentile(col,2.5):.4f}, {np.percentile(col,97.5):.4f}]")


# ============================================================
# MAIN DRIVER
# ============================================================

def run(
    infected_file="data/infected_timeseries.csv",
    rewiring_file="data/rewiring_timeseries.csv",
    degree_file="data/final_degree_histograms.csv",
    n_scale_sims=200,
    n_pilot=500,
    epsilon_quantile=0.10,
    epsilon=None,           # set this after first run to skip pilot calibration
    n_steps=10_000,
    burnin=1_000,
    seed=42,
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
    obs_summary = build_summary(infected_obs, rewiring_obs, degree_obs)
    print(f"  Observed replicates: {obs_R}")
    print(f"  Observed summary:    {obs_summary.round(4)}")

    # ---- Estimate scale ----
    print(f"\nEstimating scale ({n_scale_sims} prior-predictive sims)...")
    scale = estimate_scale(obs_R, n_sims=n_scale_sims, rng=rng)
    print(f"  Scale: {scale.round(4)}")

    # ---- Calibrate epsilon (skip if hardcoded) ----
    if epsilon is None:
        print(f"\nCalibrating epsilon ({n_pilot} pilot draws, {epsilon_quantile*100:.0f}th percentile)...")
        epsilon = calibrate_epsilon(obs_summary, obs_R, scale,
                                    n_pilot=n_pilot, quantile=epsilon_quantile, rng=rng)
        print(f"  epsilon = {epsilon:.4f}  <-- hardcode this in run() to skip next time")
    else:
        print(f"\nUsing hardcoded epsilon = {epsilon:.4f}")

    # ---- ABC-MCMC ----
    # Warm start from S4 posterior mean (Q3 result): β≈0.20, γ≈0.08, ρ≈0.30.
    # This is principled — it comes from a separate ABC run, not from the chain
    # itself — and avoids wasting steps searching for a valid starting point.
    theta_init = np.array([0.20, 0.08, 0.30])
    print(f"\nRunning ABC-MCMC ({n_steps} steps, burn-in={burnin}, "
          f"warm start from S4 posterior mean)...")
    mcmc_post, chain, distances, mcmc_acc_rate = abc_mcmc(
        obs_summary, obs_R, scale, epsilon,
        n_steps=n_steps, burnin=burnin,
        theta_init=theta_init, rng=rng,
    )
    print(f"\n  ABC-MCMC acceptance rate: {mcmc_acc_rate:.3f}")

    # ---- Summaries ----
    print_summary("ABC-MCMC (post burn-in)", mcmc_post)

    # ---- Plots ----
    print("\nGenerating plots...")
    plot_trace(chain, burnin)
    plot_distance_trace(distances, epsilon, burnin)
    plot_posterior(mcmc_post)

    return {
        "mcmc_posterior":  mcmc_post,
        "chain":           chain,
        "distances":       distances,
        "epsilon":         epsilon,
        "mcmc_acc_rate":   mcmc_acc_rate,
        "scale":           scale,
    }


if __name__ == "__main__":
    results = run(epsilon=1.7962, n_steps=500, burnin=100)
