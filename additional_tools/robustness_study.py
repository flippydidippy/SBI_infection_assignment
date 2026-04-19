"""
Robustness Study

This script runs a robustness study for the ABC inference procedures used in the
report. It evaluates how posterior conclusions change under different tuning
choices, including rejection-ABC tolerance, ABC-MCMC seed and proposal scale,
and alternative distance metrics such as Euclidean, Mahalanobis, and reweighted
Euclidean distance.

Main steps
----------
1. Load an external ABC module containing the simulator, prior, summary
   statistic, and support checks.
2. Load the observed data and construct the observed summary vector.
3. Build a prior-predictive summary bank to calibrate distance scaling and
   covariance-based distance metrics.
4. Re-run rejection ABC across multiple tolerances and distance choices.
5. Re-run ABC-MCMC across multiple seeds, proposal scales, and distance choices.
6. Summarise posterior movement relative to baseline settings using means,
   widths, and Wasserstein-style comparisons.
7. Save posterior samples and robustness summaries to disk for later analysis.

The goal is to assess whether the main inferential conclusions are stable to
reasonable methodological choices, rather than artefacts of a single tuning
configuration.
"""

import argparse
import importlib.util
import itertools
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd


LOGGER = logging.getLogger("robustness")



# ============================================================
# Utilities
# ============================================================

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def parse_csv_list(text, cast=float):
    if text is None or text == "":
        return []
    return [cast(x.strip()) for x in text.split(",") if x.strip() != ""]


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def save_npz(path, **kwargs):
    ensure_dir(Path(path).parent)
    np.savez(path, **kwargs)


def load_module_from_path(module_path):
    module_path = Path(module_path).resolve()
    if not module_path.exists():
        raise FileNotFoundError(f"ABC module not found: {module_path}")

    spec = importlib.util.spec_from_file_location("abc_module", str(module_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def posterior_summary(samples, bounds):
    means = samples.mean(axis=0)
    medians = np.median(samples, axis=0)
    lower = np.percentile(samples, 2.5, axis=0)
    upper = np.percentile(samples, 97.5, axis=0)
    std = samples.std(axis=0, ddof=1)

    prior_std = (bounds[:, 1] - bounds[:, 0]) / np.sqrt(12.0)
    width_norm = std / prior_std

    out = {}
    names = ["beta", "gamma", "rho"]
    for j, name in enumerate(names):
        out[f"{name}_mean"] = means[j]
        out[f"{name}_median"] = medians[j]
        out[f"{name}_std"] = std[j]
        out[f"{name}_q025"] = lower[j]
        out[f"{name}_q975"] = upper[j]
        out[f"{name}_width_norm"] = width_norm[j]
    return out


def mean_abs_shift(samples_a, samples_b):
    return np.abs(samples_a.mean(axis=0) - samples_b.mean(axis=0))


def std_ratio(samples_a, samples_b):
    std_a = samples_a.std(axis=0, ddof=1)
    std_b = samples_b.std(axis=0, ddof=1)
    out = np.full(3, np.nan)
    mask = std_b > 1e-12
    out[mask] = std_a[mask] / std_b[mask]
    return out


def wasserstein_1d(x, y):
    x = np.sort(np.asarray(x))
    y = np.sort(np.asarray(y))
    n = min(len(x), len(y))
    if n == 0:
        return np.nan
    q = np.linspace(0.0, 1.0, n, endpoint=False) + 0.5 / n
    xq = np.quantile(x, q)
    yq = np.quantile(y, q)
    return np.mean(np.abs(xq - yq))


def posterior_move_metrics(samples, baseline):
    mean_shift = mean_abs_shift(samples, baseline)
    width_change = std_ratio(samples, baseline)
    w1 = np.array([wasserstein_1d(samples[:, j], baseline[:, j]) for j in range(3)])

    return {
        "beta_mean_shift": mean_shift[0],
        "gamma_mean_shift": mean_shift[1],
        "rho_mean_shift": mean_shift[2],
        "beta_std_ratio_vs_baseline": width_change[0],
        "gamma_std_ratio_vs_baseline": width_change[1],
        "rho_std_ratio_vs_baseline": width_change[2],
        "beta_w1_vs_baseline": w1[0],
        "gamma_w1_vs_baseline": w1[1],
        "rho_w1_vs_baseline": w1[2],
    }


# ============================================================
# Distance metrics
# ============================================================

def build_prior_predictive_bank(abc, obs_R, n_bank, rng):
    summaries = []
    thetas = []
    for i in range(n_bank):
        theta = abc.sample_prior(rng)
        inf_arr, rew_arr, deg_arr = abc.simulate_replicates(*theta, R=obs_R, rng=rng)
        s = abc.build_summary(inf_arr, rew_arr, deg_arr)
        summaries.append(s)
        thetas.append(theta)
        if (i + 1) % 50 == 0:
            LOGGER.info("Prior-predictive bank: %d/%d", i + 1, n_bank)

    return np.asarray(thetas), np.asarray(summaries)


def build_distance_context(bank_summaries):
    scale = bank_summaries.std(axis=0, ddof=1)
    scale[scale < 1e-10] = 1.0

    cov = np.cov(bank_summaries, rowvar=False)
    ridge = 1e-8 * np.eye(cov.shape[0])
    inv_cov = np.linalg.pinv(cov + ridge)

    return {
        "scale": scale,
        "inv_cov": inv_cov,
    }


def distance_euclidean(sim_s, obs_s, ctx):
    z = (sim_s - obs_s) / ctx["scale"]
    return np.sqrt(np.sum(z ** 2))


def distance_mahalanobis(sim_s, obs_s, ctx):
    z = sim_s - obs_s
    return float(np.sqrt(z @ ctx["inv_cov"] @ z))


def distance_reweighted(sim_s, obs_s, ctx, weights=None):
    if weights is None:
        weights = 1.0 / ctx["scale"]
    z = (sim_s - obs_s) * weights
    return np.sqrt(np.sum(z ** 2))


def get_distance_fn(name, ctx):
    name = name.lower()
    if name == "euclidean":
        return lambda sim_s, obs_s: distance_euclidean(sim_s, obs_s, ctx)
    if name == "mahalanobis":
        return lambda sim_s, obs_s: distance_mahalanobis(sim_s, obs_s, ctx)
    if name == "reweighted":
        return lambda sim_s, obs_s: distance_reweighted(sim_s, obs_s, ctx)
    raise ValueError(f"Unknown distance metric: {name}")


# ============================================================
# Robust rejection ABC
# ============================================================

def run_rejection_abc_custom(abc, obs_summary, obs_R, n_draws, accept_frac, distance_name, rng, ctx):
    distance_fn = get_distance_fn(distance_name, ctx)

    params = []
    summaries = []
    distances = []

    for i in range(n_draws):
        theta = abc.sample_prior(rng)
        inf_arr, rew_arr, deg_arr = abc.simulate_replicates(*theta, R=obs_R, rng=rng)
        sim_s = abc.build_summary(inf_arr, rew_arr, deg_arr)
        d = distance_fn(sim_s, obs_summary)

        params.append(theta)
        summaries.append(sim_s)
        distances.append(d)

        if (i + 1) % 100 == 0 or i == n_draws - 1:
            LOGGER.info(
                "[Rejection | %s | tol=%.4f] %d/%d draws",
                distance_name, accept_frac, i + 1, n_draws
            )

    params = np.asarray(params)
    summaries = np.asarray(summaries)
    distances = np.asarray(distances)

    n_keep = max(1, int(round(accept_frac * n_draws)))
    keep_idx = np.argsort(distances)[:n_keep]

    return {
        "accepted_params": params[keep_idx],
        "accepted_summaries": summaries[keep_idx],
        "accepted_distances": distances[keep_idx],
        "all_distances": distances,
        "n_keep": n_keep,
    }


# ============================================================
# Robust ABC-MCMC
# ============================================================

def run_abc_mcmc_custom(
    abc,
    obs_summary,
    obs_R,
    epsilon,
    n_steps,
    burnin,
    proposal_scale,
    distance_name,
    rng,
    ctx,
):
    distance_fn = get_distance_fn(distance_name, ctx)
    proposal_std = proposal_scale * (abc.BOUNDS[:, 1] - abc.BOUNDS[:, 0])

    LOGGER.info("Finding valid ABC-MCMC start for %s ...", distance_name)
    n_tries = 0
    while True:
        theta = abc.sample_prior(rng)
        inf_arr, rew_arr, deg_arr = abc.simulate_replicates(*theta, R=obs_R, rng=rng)
        sim_s = abc.build_summary(inf_arr, rew_arr, deg_arr)
        d = distance_fn(sim_s, obs_summary)
        n_tries += 1
        if d < epsilon:
            theta_curr = theta.copy()
            d_curr = d
            break
        if n_tries % 50 == 0:
            LOGGER.info("Still searching for start: %d tries", n_tries)

    LOGGER.info("MCMC start found after %d tries: theta=%s d=%.4f", n_tries, np.round(theta_curr, 4), d_curr)

    chain = np.zeros((n_steps, 3))
    distances = np.zeros(n_steps)
    accepted = 0

    for i in range(n_steps):
        theta_prop = theta_curr + rng.normal(0.0, proposal_std, size=3)

        if abc.in_prior(theta_prop):
            inf_arr, rew_arr, deg_arr = abc.simulate_replicates(*theta_prop, R=obs_R, rng=rng)
            sim_s = abc.build_summary(inf_arr, rew_arr, deg_arr)
            d_prop = distance_fn(sim_s, obs_summary)

            if d_prop < epsilon:
                theta_curr = theta_prop
                d_curr = d_prop
                accepted += 1

        chain[i] = theta_curr
        distances[i] = d_curr

        if (i + 1) % 500 == 0 or i == n_steps - 1:
            LOGGER.info(
                "[MCMC | %s | scale=%.3f] step %d/%d | acc=%.3f",
                distance_name, proposal_scale, i + 1, n_steps, accepted / (i + 1)
            )

    post = chain[burnin:]
    return {
        "posterior": post,
        "chain": chain,
        "distances": distances,
        "acc_rate": accepted / n_steps,
    }


# ============================================================
# Epsilon calibration for custom distances
# ============================================================

def calibrate_epsilon_custom(abc, obs_summary, obs_R, n_pilot, quantile, distance_name, rng, ctx):
    distance_fn = get_distance_fn(distance_name, ctx)
    distances = []

    for i in range(n_pilot):
        theta = abc.sample_prior(rng)
        inf_arr, rew_arr, deg_arr = abc.simulate_replicates(*theta, R=obs_R, rng=rng)
        sim_s = abc.build_summary(inf_arr, rew_arr, deg_arr)
        distances.append(distance_fn(sim_s, obs_summary))
        if (i + 1) % 100 == 0 or i == n_pilot - 1:
            LOGGER.info(
                "Pilot epsilon [%s]: %d/%d",
                distance_name, i + 1, n_pilot
            )

    distances = np.asarray(distances)
    return float(np.quantile(distances, quantile)), distances


# ============================================================
# Main study
# ============================================================

def load_observed_summary(abc, infected_file, rewiring_file, degree_file):
    infected_df, rewiring_df, degree_df = abc.load_observed_data(
        infected_file=infected_file,
        rewiring_file=rewiring_file,
        degree_file=degree_file,
    )
    inf_obs, rew_obs, deg_obs = abc.make_observed_arrays(infected_df, rewiring_df, degree_df)
    obs_summary = abc.build_summary(inf_obs, rew_obs, deg_obs)
    return obs_summary, inf_obs.shape[0]


def write_run_metadata(output_dir, args):
    with open(Path(output_dir) / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Robustness study for rejection ABC and ABC-MCMC")
    parser.add_argument(
        "--abc_module",
        type=str,
        default="4_Advanced_Methods/4_ABC_MCMC.py",
        help="Path to the Python file containing simulate_replicates/build_summary/etc.",
    )
    parser.add_argument("--infected_file", type=str, default="data/infected_timeseries.csv")
    parser.add_argument("--rewiring_file", type=str, default="data/rewiring_timeseries.csv")
    parser.add_argument("--degree_file", type=str, default="data/final_degree_histograms.csv")

    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--scales", type=str, default="0.10,0.15,0.20")
    parser.add_argument("--tolerances", type=str, default="0.01,0.02,0.05")
    parser.add_argument("--distance_metrics", type=str, default="euclidean,mahalanobis,reweighted")

    parser.add_argument("--Nsim", type=int, default=5000, help="Rejection ABC prior draws")
    parser.add_argument("--n_iter", type=int, default=20000, help="ABC-MCMC total steps")
    parser.add_argument("--burnin", type=int, default=5000)
    parser.add_argument("--pilot_sims", type=int, default=500, help="Pilot sims for epsilon calibration")
    parser.add_argument("--pilot_quantile", type=float, default=0.005, help="Quantile for epsilon")
    parser.add_argument("--bank_sims", type=int, default=200, help="Prior-predictive bank size for distances")

    parser.add_argument("--output_dir", type=str, default="results/robustness")
    parser.add_argument("--skip_rejection", action="store_true")
    parser.add_argument("--skip_mcmc", action="store_true")

    args = parser.parse_args()

    seeds = parse_csv_list(args.seeds, int)
    scales = parse_csv_list(args.scales, float)
    tolerances = parse_csv_list(args.tolerances, float)
    distances = [x.strip() for x in args.distance_metrics.split(",") if x.strip()]

    ensure_dir(args.output_dir)
    write_run_metadata(args.output_dir, args)

    LOGGER.info("Loading ABC code from %s", args.abc_module)
    abc = load_module_from_path(args.abc_module)

    required = [
        "simulate_replicates",
        "build_summary",
        "sample_prior",
        "in_prior",
        "BOUNDS",
        "load_observed_data",
        "make_observed_arrays",
    ]
    missing = [name for name in required if not hasattr(abc, name)]
    if missing:
        raise AttributeError(f"ABC module is missing required names: {missing}")

    LOGGER.info("Loading observed data and summary")
    obs_summary, obs_R = load_observed_summary(
        abc, args.infected_file, args.rewiring_file, args.degree_file
    )
    LOGGER.info("Observed summary: %s", np.round(obs_summary, 4))
    LOGGER.info("Observed replicate count R=%d", obs_R)

    LOGGER.info("Building prior-predictive summary bank")
    rng_bank = np.random.default_rng(12345)
    _, bank_summaries = build_prior_predictive_bank(abc, obs_R, args.bank_sims, rng_bank)
    dist_ctx = build_distance_context(bank_summaries)

    results_rows = []
    saved_posteriors = {}

    # ----------------------------
    # Rejection ABC robustness
    # ----------------------------
    if not args.skip_rejection:
        rej_dir = Path(args.output_dir) / "rejection"
        ensure_dir(rej_dir)

        baseline_key = None

        for tol, dist_name in itertools.product(tolerances, distances):
            rng = np.random.default_rng(1000 + int(10000 * tol) + abs(hash(dist_name)) % 1000)
            LOGGER.info("Running rejection study: tol=%.4f | distance=%s", tol, dist_name)

            out = run_rejection_abc_custom(
                abc=abc,
                obs_summary=obs_summary,
                obs_R=obs_R,
                n_draws=args.Nsim,
                accept_frac=tol,
                distance_name=dist_name,
                rng=rng,
                ctx=dist_ctx,
            )

            samples = out["accepted_params"]
            key = f"rejection_tol{tol:g}_{dist_name}"
            saved_posteriors[key] = samples

            save_npz(
                rej_dir / f"{key}.npz",
                samples=samples,
                summaries=out["accepted_summaries"],
                distances=out["accepted_distances"],
                all_distances=out["all_distances"],
            )

            row = {
                "method": "rejection",
                "distance": dist_name,
                "seed": np.nan,
                "proposal_scale": np.nan,
                "tolerance": tol,
                "n_samples": len(samples),
            }
            row.update(posterior_summary(samples, abc.BOUNDS))
            results_rows.append(row)

            if baseline_key is None and dist_name == "euclidean":
                baseline_key = key

        # posterior movement vs baseline Euclidean at first tolerance
        if baseline_key is not None:
            baseline = saved_posteriors[baseline_key]
            for row in results_rows:
                if row["method"] != "rejection":
                    continue
                key = f"rejection_tol{row['tolerance']:g}_{row['distance']}"
                move = posterior_move_metrics(saved_posteriors[key], baseline)
                row.update(move)

    # ----------------------------
    # ABC-MCMC robustness
    # ----------------------------
    if not args.skip_mcmc:
        mcmc_dir = Path(args.output_dir) / "mcmc"
        ensure_dir(mcmc_dir)

        baseline_key = None

        epsilon_by_distance = {}
        for dist_name in distances:
            rng_eps = np.random.default_rng(7000 + abs(hash(dist_name)) % 1000)
            eps, pilot_d = calibrate_epsilon_custom(
                abc=abc,
                obs_summary=obs_summary,
                obs_R=obs_R,
                n_pilot=args.pilot_sims,
                quantile=args.pilot_quantile,
                distance_name=dist_name,
                rng=rng_eps,
                ctx=dist_ctx,
            )
            epsilon_by_distance[dist_name] = eps
            save_npz(mcmc_dir / f"pilot_{dist_name}.npz", distances=pilot_d, epsilon=eps)
            LOGGER.info("Calibrated epsilon for %s: %.6f", dist_name, eps)

        mcmc_rows_start = len(results_rows)

        for seed, scale, dist_name in itertools.product(seeds, scales, distances):
            rng = np.random.default_rng(seed)
            eps = epsilon_by_distance[dist_name]

            LOGGER.info(
                "Running ABC-MCMC: seed=%d | scale=%.3f | distance=%s | epsilon=%.6f",
                seed, scale, dist_name, eps
            )

            out = run_abc_mcmc_custom(
                abc=abc,
                obs_summary=obs_summary,
                obs_R=obs_R,
                epsilon=eps,
                n_steps=args.n_iter,
                burnin=args.burnin,
                proposal_scale=scale,
                distance_name=dist_name,
                rng=rng,
                ctx=dist_ctx,
            )

            samples = out["posterior"]
            key = f"mcmc_seed{seed}_scale{scale:g}_{dist_name}"
            saved_posteriors[key] = samples

            save_npz(
                mcmc_dir / f"{key}.npz",
                samples=samples,
                chain=out["chain"],
                distances=out["distances"],
                epsilon=eps,
                acc_rate=out["acc_rate"],
            )

            row = {
                "method": "mcmc",
                "distance": dist_name,
                "seed": seed,
                "proposal_scale": scale,
                "tolerance": np.nan,
                "epsilon": eps,
                "n_samples": len(samples),
                "acc_rate": out["acc_rate"],
            }
            row.update(posterior_summary(samples, abc.BOUNDS))
            results_rows.append(row)

            if baseline_key is None and seed == seeds[0] and scale == scales[0] and dist_name == "euclidean":
                baseline_key = key

        if baseline_key is not None:
            baseline = saved_posteriors[baseline_key]
            for row in results_rows[mcmc_rows_start:]:
                key = f"mcmc_seed{int(row['seed'])}_scale{row['proposal_scale']:g}_{row['distance']}"
                move = posterior_move_metrics(saved_posteriors[key], baseline)
                row.update(move)

    # ----------------------------
    # Save summaries
    # ----------------------------
    df = pd.DataFrame(results_rows)
    df.to_csv(Path(args.output_dir) / "robustness_summary.csv", index=False)

    move_cols = [
        "method", "distance", "seed", "proposal_scale", "tolerance",
        "beta_mean_shift", "gamma_mean_shift", "rho_mean_shift",
        "beta_std_ratio_vs_baseline", "gamma_std_ratio_vs_baseline", "rho_std_ratio_vs_baseline",
        "beta_w1_vs_baseline", "gamma_w1_vs_baseline", "rho_w1_vs_baseline",
    ]
    move_cols = [c for c in move_cols if c in df.columns]
    df[move_cols].to_csv(Path(args.output_dir) / "posterior_movement.csv", index=False)

    LOGGER.info("Done. Wrote:")
    LOGGER.info("  %s", Path(args.output_dir) / "robustness_summary.csv")
    LOGGER.info("  %s", Path(args.output_dir) / "posterior_movement.csv")


if __name__ == "__main__":
    main()