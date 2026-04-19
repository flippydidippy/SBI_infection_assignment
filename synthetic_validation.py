"""
Synthetic truth validation
--------------------------
Generates synthetic observed data from known parameters, then runs all three
inference methods and checks whether the true values are recovered:

  1. Rejection ABC          (top 1%, 5k draws)
  2. Regression adjustment  (Beaumont et al. 2002, top 5% then adjust)
  3. ABC-MCMC               (Marjoram et al. 2003, epsilon = half of rej epsilon)

For each method:
  - KDE of marginal posteriors with true value marked
  - 90% credible interval shaded
  - Coverage table printed to console
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from numba import njit, prange
import time

# ============================================================
# TRUE PARAMETERS
# ============================================================

TRUTH        = np.array([0.25, 0.10, 0.40])   # β*, γ*, ρ*
PARAM_NAMES  = ["β", "γ", "ρ"]
BOUNDS       = np.array([[0.05, 0.50],
                          [0.02, 0.20],
                          [0.00, 0.80]])

# Simulation settings
N         = 200
P_EDGE    = 0.05
N_INF0    = 5
T         = 200
OBS_R     = 40

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


def simulate_replicates(beta, gamma, rho, R, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    np.random.seed(rng.integers(0, 2**31))
    return _simulate_replicates_parallel(beta, gamma, rho, R, N, P_EDGE, N_INF0, T)


# ============================================================
# SUMMARY STATISTICS  (S4 — 7 scalars)
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
# PRIOR
# ============================================================

def in_prior(theta):
    return np.all((theta >= BOUNDS[:, 0]) & (theta <= BOUNDS[:, 1]))


def sample_prior(rng):
    return rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1])


# ============================================================
# SCALE ESTIMATION
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
    scale = np.array(summaries).std(axis=0, ddof=1)
    scale[scale < 1e-8] = 1.0
    return scale


# ============================================================
# METHOD 1 & 2: REJECTION ABC + REGRESSION ADJUSTMENT
# ============================================================

def rejection_abc(obs_summary, obs_R, scale, n_draws=5000, accept_frac=0.01, rng=None):
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
        if (i + 1) % 1000 == 0:
            print(f"    draw {i+1}/{n_draws}")
    params = np.array(params)
    dists  = np.array(dists)
    sums   = np.array(sums)
    n_keep = max(1, int(accept_frac * n_draws))
    idx    = np.argsort(dists)[:n_keep]
    return params[idx], sums[idx], dists[idx], dists


def _logit(x, lo, hi):
    p = np.clip((x - lo) / (hi - lo), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def _sigmoid(x, lo, hi):
    return lo + (hi - lo) / (1.0 + np.exp(-x))


def regression_adjust(params, summaries, obs_summary, distances):
    n  = len(params)
    h  = distances.max()
    u  = distances / h
    w  = np.maximum(0.0, 0.75 * (1.0 - u ** 2))
    ds = summaries - obs_summary
    X  = np.column_stack([np.ones(n), ds])
    adj = np.zeros_like(params)
    for j, (lo, hi) in enumerate(BOUNDS):
        phi    = _logit(params[:, j], lo, hi)
        W      = np.diag(w)
        XtW    = X.T @ W
        beta_j = np.linalg.lstsq(XtW @ X, XtW @ phi, rcond=None)[0]
        adj[:, j] = _sigmoid(phi - ds @ beta_j[1:], lo, hi)
    return adj


# ============================================================
# METHOD 3: ABC-MCMC
# ============================================================

def abc_mcmc(obs_summary, obs_R, scale, epsilon,
             n_steps=20_000, burnin=5_000,
             start_theta=None, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    theta_curr = start_theta.copy() if start_theta is not None else sample_prior(rng)
    inf_arr, rew_arr, deg_arr = simulate_replicates(*theta_curr, R=obs_R, rng=rng)
    d_curr = weighted_distance(build_summary(inf_arr, rew_arr, deg_arr), obs_summary, scale)
    tries = 0
    while d_curr > epsilon:
        theta_curr = sample_prior(rng)
        inf_arr, rew_arr, deg_arr = simulate_replicates(*theta_curr, R=obs_R, rng=rng)
        d_curr = weighted_distance(build_summary(inf_arr, rew_arr, deg_arr), obs_summary, scale)
        tries += 1
    print(f"    Found valid start in {tries} tries (d={d_curr:.4f})")

    proposal_std   = 0.15 * (BOUNDS[:, 1] - BOUNDS[:, 0])
    adapt_interval = 200
    target_acc     = 0.10
    adapt_factor   = 1.3
    std_lo = 0.03 * (BOUNDS[:, 1] - BOUNDS[:, 0])
    std_hi = 0.60 * (BOUNDS[:, 1] - BOUNDS[:, 0])

    chain      = np.zeros((n_steps, 3))
    n_acc      = 0
    n_acc_post = 0
    adapt_acc  = 0
    t0 = time.time()

    for i in range(n_steps):
        theta_prop = theta_curr + rng.normal(0.0, proposal_std)
        if in_prior(theta_prop):
            inf_arr, rew_arr, deg_arr = simulate_replicates(*theta_prop, R=obs_R, rng=rng)
            d_prop = weighted_distance(build_summary(inf_arr, rew_arr, deg_arr), obs_summary, scale)
            if d_prop <= epsilon:
                theta_curr = theta_prop
                n_acc += 1
                adapt_acc += 1
                if i >= burnin:
                    n_acc_post += 1

        chain[i] = theta_curr

        if i < burnin and (i + 1) % adapt_interval == 0:
            window_acc = adapt_acc / adapt_interval
            if window_acc > target_acc + 0.05:
                proposal_std = np.minimum(proposal_std * adapt_factor, std_hi)
            elif window_acc < target_acc - 0.05:
                proposal_std = np.maximum(proposal_std / adapt_factor, std_lo)
            adapt_acc = 0

        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta  = (n_steps - i - 1) / rate
            print(f"    Step {i+1}/{n_steps} | acc={n_acc/(i+1):.3f} | ETA {eta/60:.1f}min")

    posterior = chain[burnin:]
    acc_rate  = n_acc_post / (n_steps - burnin)
    print(f"    Done | post-burnin acc rate: {acc_rate:.3f}")
    return posterior, acc_rate


# ============================================================
# COVERAGE TABLE
# ============================================================

def coverage_table(samples_dict):
    print(f"\n{'='*70}")
    print(f"  Synthetic truth recovery  "
          f"(truth: β={TRUTH[0]}, γ={TRUTH[1]}, ρ={TRUTH[2]})")
    print(f"{'='*70}")
    print(f"  {'Method':<25} {'Param':<5} {'Mean':>7} {'5%':>7} {'95%':>7} {'Cover':>6}")
    print(f"  {'-'*58}")
    for method, samples in samples_dict.items():
        for j, p in enumerate(PARAM_NAMES):
            col = samples[:, j]
            mean = col.mean()
            lo   = np.percentile(col, 5)
            hi   = np.percentile(col, 95)
            covered = lo <= TRUTH[j] <= hi
            print(f"  {method:<25} {p:<5} {mean:>7.3f} {lo:>7.3f} {hi:>7.3f} "
                  f"  {'YES' if covered else 'NO ':>4}")
        print()


# ============================================================
# PLOT
# ============================================================

def plot_recovery(samples_dict, save="synthetic_validation.png"):
    colors = {"Rejection ABC":     "darkorange",
               "Reg. adjustment":  "seagreen",
               "ABC-MCMC":         "steelblue"}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for col, (name, (lo, hi)) in enumerate(zip(PARAM_NAMES, BOUNDS)):
        ax = axes[col]
        for label, post in samples_dict.items():
            x  = post[:, col]
            xs = np.linspace(lo, hi, 300)
            try:
                kde = gaussian_kde(x, bw_method="scott")
                ys  = kde(xs)
                ax.plot(xs, ys, lw=2, color=colors[label], label=label)
                p5, p95 = np.percentile(x, 5), np.percentile(x, 95)
                ax.fill_between(xs, ys,
                                where=(xs >= p5) & (xs <= p95),
                                alpha=0.15, color=colors[label])
            except Exception:
                pass

        ax.axvline(TRUTH[col], color="red", lw=2, ls="--", label="True value")
        ax.axhline(1.0 / (hi - lo), color="gray", lw=1, ls=":", label="Prior")
        ax.set_xlim(lo, hi)
        ax.set_xlabel(name, fontsize=12)
        ax.set_ylabel("Density")
        ax.set_title(f"Recovery of {name}")
        if col == 0:
            ax.legend(fontsize=8)

    fig.suptitle(
        f"Synthetic truth recovery  (true: β={TRUTH[0]}, γ={TRUTH[1]}, ρ={TRUTH[2]})\n"
        "Shaded = 90% CI   Red dashed = true value",
        fontsize=11)
    plt.tight_layout()
    plt.savefig(save, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save}")


# ============================================================
# MAIN
# ============================================================

def run(seed=42):
    rng = np.random.default_rng(seed)

    # ---- JIT warmup ----
    print("Compiling Numba simulator...")
    _ = simulate_replicates(TRUTH[0], TRUTH[1], TRUTH[2], R=1, rng=rng)
    print("  Done.\n")

    # ---- Generate synthetic observed data ----
    print(f"Generating synthetic data  (β={TRUTH[0]}, γ={TRUTH[1]}, ρ={TRUTH[2]}, R={OBS_R})...")
    inf_obs, rew_obs, deg_obs = simulate_replicates(*TRUTH, R=OBS_R, rng=rng)
    obs_summary = build_summary(inf_obs, rew_obs, deg_obs)
    print(f"  obs_summary = {np.round(obs_summary, 4)}\n")

    # ---- Scale ----
    print("Estimating normalisation scale (200 sims)...")
    scale = estimate_scale(OBS_R, n_sims=200, rng=rng)
    print(f"  scale = {np.round(scale, 4)}\n")

    # ---- Method 1: Rejection ABC (5k draws, top 1%) ----
    print("Method 1 — Rejection ABC (5000 draws, top 1%)...")
    rej_post, _, rej_dists, _ = rejection_abc(
        obs_summary, OBS_R, scale, n_draws=20000, accept_frac=0.01, rng=rng)
    epsilon_rej  = float(rej_dists[-1])
    epsilon_mcmc = epsilon_rej / 2.0
    print(f"  {len(rej_post)} samples  (epsilon_rej={epsilon_rej:.4f})\n")

    # ---- Method 2: Regression adjustment (5k draws, top 5%) ----
    print("Method 2 — Regression adjustment (5000 draws, top 5%)...")
    rej5_post, rej5_sums, rej5_dists, _ = rejection_abc(
        obs_summary, OBS_R, scale, n_draws=20000, accept_frac=0.05, rng=rng)
    adj_post = regression_adjust(rej5_post, rej5_sums, obs_summary, rej5_dists)
    print(f"  {len(adj_post)} samples adjusted.\n")

    # ---- Method 3: ABC-MCMC (epsilon = half of rej epsilon) ----
    print(f"Method 3 — ABC-MCMC (20k steps, epsilon={epsilon_mcmc:.4f})...")
    warm_start = rej_post.mean(axis=0)
    mcmc_post, acc_rate = abc_mcmc(
        obs_summary, OBS_R, scale, epsilon_mcmc,
        n_steps=20_000, burnin=5_000,
        start_theta=warm_start, rng=rng)
    print(f"  {len(mcmc_post)} samples, acc rate={acc_rate:.3f}\n")

    # ---- Coverage + plot ----
    samples_dict = {
        "Rejection ABC":    rej_post,
        "Reg. adjustment":  adj_post,
        "ABC-MCMC":         mcmc_post,
    }
    coverage_table(samples_dict)
    plot_recovery(samples_dict)


if __name__ == "__main__":
    run()
