import numpy as np
from archive.fast_simulator_test.org_simulator import simulate
from archive.fast_simulator_test.fast_sim import simulate_fast

def compare_many(sim1, sim2, *,
                 n_runs=500,
                 beta=0.25, gamma=0.08, rho=0.2,
                 N=200, p_edge=0.05, n_infected0=5, T=200):
    peak1 = []
    peak2 = []
    final1 = []
    final2 = []
    total_rew1 = []
    total_rew2 = []
    deg1 = []
    deg2 = []

    for seed in range(n_runs):
        rng1 = np.random.default_rng(seed+1000)
        rng2 = np.random.default_rng(seed+1000)

        inf1, rew1, hist1 = sim1(beta, gamma, rho, N=N, p_edge=p_edge,
                                 n_infected0=n_infected0, T=T, rng=rng1)
        inf2, rew2, hist2 = sim2(beta, gamma, rho, N=N, p_edge=p_edge,
                                 n_infected0=n_infected0, T=T, rng=rng2)

        peak1.append(np.max(inf1))
        peak2.append(np.max(inf2))

        final1.append(inf1[-1])
        final2.append(inf2[-1])

        total_rew1.append(np.sum(rew1))
        total_rew2.append(np.sum(rew2))

        deg1.append(hist1)
        deg2.append(hist2)

    peak1 = np.array(peak1)
    peak2 = np.array(peak2)
    final1 = np.array(final1)
    final2 = np.array(final2)
    total_rew1 = np.array(total_rew1)
    total_rew2 = np.array(total_rew2)
    deg1 = np.array(deg1)
    deg2 = np.array(deg2)

    print("Peak infected fraction:")
    print("  original mean:", peak1.mean(), "fast mean:", peak2.mean(),
          "diff:", peak2.mean() - peak1.mean())

    print("Final infected fraction:")
    print("  original mean:", final1.mean(), "fast mean:", final2.mean(),
          "diff:", final2.mean() - final1.mean())

    print("Total rewires:")
    print("  original mean:", total_rew1.mean(), "fast mean:", total_rew2.mean(),
          "diff:", total_rew2.mean() - total_rew1.mean())

    print("Final degree histogram mean absolute difference per bin:")
    print(np.mean(np.abs(deg1.mean(axis=0) - deg2.mean(axis=0))))

    return {
        "peak1": peak1, "peak2": peak2,
        "final1": final1, "final2": final2,
        "total_rew1": total_rew1, "total_rew2": total_rew2,
        "deg1": deg1, "deg2": deg2,
    }

results = compare_many(simulate, simulate_fast, n_runs=300)
print(results)



def compare_mean_trajectories(sim1, sim2, *,
                              n_runs=300,
                              beta=0.25, gamma=0.08, rho=0.2,
                              N=200, p_edge=0.05, n_infected0=5, T=200):
    curves1 = []
    curves2 = []

    for seed in range(n_runs):
        rng1 = np.random.default_rng(seed)
        rng2 = np.random.default_rng(seed)

        inf1, rew1, hist1 = sim1(beta, gamma, rho, N=N, p_edge=p_edge,
                                 n_infected0=n_infected0, T=T, rng=rng1)
        inf2, rew2, hist2 = sim2(beta, gamma, rho, N=N, p_edge=p_edge,
                                 n_infected0=n_infected0, T=T, rng=rng2)

        curves1.append(inf1)
        curves2.append(inf2)

    curves1 = np.array(curves1)
    curves2 = np.array(curves2)

    mean1 = curves1.mean(axis=0)
    mean2 = curves2.mean(axis=0)

    max_abs_diff = np.max(np.abs(mean1 - mean2))
    print("Max absolute difference in mean infection trajectory:", max_abs_diff)

    return mean1, mean2

# print("Comparing mean trajectories...")
# mean_traj1, mean_traj2 = compare_mean_trajectories(simulate, simulate_fast, n_runs=300)
# print("Done.")
# print("Mean trajectory original:", mean_traj1)
# print("Mean trajectory fast:", mean_traj2)

def sanity_checks(sim):
    tests = [
        dict(beta=0.0, gamma=0.1, rho=0.2, p_edge=0.05),
        dict(beta=0.2, gamma=0.0, rho=0.2, p_edge=0.05),
        dict(beta=0.2, gamma=0.1, rho=0.0, p_edge=0.05),
        dict(beta=0.2, gamma=0.1, rho=0.2, p_edge=0.0),
    ]

    for k, params in enumerate(tests, 1):
        inf, rew, hist = sim(**params, N=100, n_infected0=5, T=50, rng=np.random.default_rng(0))
        print(f"Test {k}: {params}")
        print("  final infected fraction:", inf[-1])
        print("  total rewires:", rew.sum())
        print("  degree histogram sum:", hist.sum())

print("Running sanity checks on original simulator...")
sanity_checks(simulate)
print("Running sanity checks on fast simulator...")
sanity_checks(simulate_fast)