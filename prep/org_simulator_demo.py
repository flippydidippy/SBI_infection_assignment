import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ============================================================
# ORIGINAL-STYLE SIMULATOR + VISUALIZATION
# ============================================================

def simulate_with_history(
    beta=0.25,
    gamma=0.08,
    rho=0.20,
    N=60,
    p_edge=0.06,
    n_infected0=3,
    T=80,
    rng=None,
    verbose=False,
):
    """
    Run one replicate of the adaptive-network SIR model and store
    enough history to visualize it nicely later.

    Model logic matches the original version:
      Phase 1: infection
      Phase 2: recovery
      Phase 3: rewiring

    Parameters
    ----------
    beta : float in [0,1]
        Per-edge infection probability per time step.
    gamma : float in [0,1]
        Recovery probability per infected node per time step.
    rho : float in [0,1]
        Rewiring probability per S-I edge per time step.
    N : int
        Number of nodes.
    p_edge : float
        Initial Erdos-Renyi edge probability.
    n_infected0 : int
        Number of initially infected nodes.
    T : int
        Number of time steps.
    rng : np.random.Generator or None
        Random number generator.
    verbose : bool
        If True, print a readable summary each time step.

    Returns
    -------
    result : dict
        Contains time series, graph history, and node state history.
    """
    if rng is None:
        rng = np.random.default_rng()

    # ------------------------------------------------------------
    # Build the initial Erdos-Renyi graph using adjacency-list sets
    # ------------------------------------------------------------
    neighbors = [set() for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            if rng.random() < p_edge:
                neighbors[i].add(j)
                neighbors[j].add(i)

    # ------------------------------------------------------------
    # States: 0=S, 1=I, 2=R
    # ------------------------------------------------------------
    state = np.zeros(N, dtype=np.int8)
    initial_infected = rng.choice(N, size=n_infected0, replace=False)
    state[initial_infected] = 1

    infected_fraction = np.zeros(T + 1)
    rewire_counts = np.zeros(T + 1, dtype=np.int64)

    # Extra diagnostics for user visibility
    susceptible_counts = np.zeros(T + 1, dtype=np.int64)
    infected_counts = np.zeros(T + 1, dtype=np.int64)
    recovered_counts = np.zeros(T + 1, dtype=np.int64)
    new_infection_counts = np.zeros(T + 1, dtype=np.int64)
    recovery_counts = np.zeros(T + 1, dtype=np.int64)

    # Store history for visualization
    state_history = [state.copy()]
    edge_history = [extract_edges(neighbors)]

    infected_fraction[0] = np.sum(state == 1) / N
    susceptible_counts[0] = np.sum(state == 0)
    infected_counts[0] = np.sum(state == 1)
    recovered_counts[0] = np.sum(state == 2)

    if verbose:
        print("=" * 72)
        print("Adaptive-network SIR simulation")
        print(f"N={N}, p_edge={p_edge}, T={T}")
        print(f"beta={beta}, gamma={gamma}, rho={rho}")
        print(f"Initial infected nodes: {sorted(initial_infected.tolist())}")
        print(f"t=0 | S={susceptible_counts[0]:3d}  I={infected_counts[0]:3d}  "
              f"R={recovered_counts[0]:3d}  infected_fraction={infected_fraction[0]:.3f}")
        print("=" * 72)

    # ------------------------------------------------------------
    # Main simulation loop
    # ------------------------------------------------------------
    for t in range(1, T + 1):
        # -------------------------
        # Phase 1: infection
        # -------------------------
        new_infections = set()
        infected_nodes = np.where(state == 1)[0]

        for i in infected_nodes:
            for j in neighbors[i]:
                if state[j] == 0:
                    if rng.random() < beta:
                        new_infections.add(j)

        for j in new_infections:
            state[j] = 1

        # -------------------------
        # Phase 2: recovery
        # -------------------------
        recoveries_this_step = 0
        infected_nodes = np.where(state == 1)[0]
        for i in infected_nodes:
            if rng.random() < gamma:
                state[i] = 2
                recoveries_this_step += 1

        # -------------------------
        # Phase 3: rewiring
        # -------------------------
        rewire_count = 0
        si_edges = []

        for i in range(N):
            if state[i] == 0:
                for j in neighbors[i]:
                    if state[j] == 1:
                        si_edges.append((i, j))

        for s_node, i_node in si_edges:
            if rng.random() < rho:
                if i_node not in neighbors[s_node]:
                    continue

                neighbors[s_node].discard(i_node)
                neighbors[i_node].discard(s_node)

                candidates = []
                for k in range(N):
                    if k != s_node and k not in neighbors[s_node]:
                        candidates.append(k)

                if candidates:
                    new_partner = rng.choice(candidates)
                    neighbors[s_node].add(new_partner)
                    neighbors[new_partner].add(s_node)
                    rewire_count += 1

        # -------------------------
        # Record stats
        # -------------------------
        infected_fraction[t] = np.sum(state == 1) / N
        rewire_counts[t] = rewire_count
        susceptible_counts[t] = np.sum(state == 0)
        infected_counts[t] = np.sum(state == 1)
        recovered_counts[t] = np.sum(state == 2)
        new_infection_counts[t] = len(new_infections)
        recovery_counts[t] = recoveries_this_step

        state_history.append(state.copy())
        edge_history.append(extract_edges(neighbors))

        if verbose:
            print(
                f"t={t:2d} | "
                f"S={susceptible_counts[t]:3d}  "
                f"I={infected_counts[t]:3d}  "
                f"R={recovered_counts[t]:3d}  "
                f"new_inf={new_infection_counts[t]:3d}  "
                f"recov={recovery_counts[t]:3d}  "
                f"rewires={rewire_counts[t]:3d}  "
                f"infected_fraction={infected_fraction[t]:.3f}"
            )

    # ------------------------------------------------------------
    # Final degree histogram
    # ------------------------------------------------------------
    degree_histogram = np.zeros(31, dtype=np.int64)
    for i in range(N):
        deg = min(len(neighbors[i]), 30)
        degree_histogram[deg] += 1

    return {
        "params": {
            "beta": beta,
            "gamma": gamma,
            "rho": rho,
            "N": N,
            "p_edge": p_edge,
            "n_infected0": n_infected0,
            "T": T,
        },
        "infected_fraction": infected_fraction,
        "rewire_counts": rewire_counts,
        "susceptible_counts": susceptible_counts,
        "infected_counts": infected_counts,
        "recovered_counts": recovered_counts,
        "new_infection_counts": new_infection_counts,
        "recovery_counts": recovery_counts,
        "degree_histogram": degree_histogram,
        "state_history": state_history,
        "edge_history": edge_history,
    }


def extract_edges(neighbors):
    """Return undirected edge list as list of (i, j) with i < j."""
    edges = []
    for i, nbrs in enumerate(neighbors):
        for j in nbrs:
            if i < j:
                edges.append((i, j))
    return edges


def spring_layout_simple(N, edges, rng=None, iterations=100, k=None):
    """
    Simple force-directed layout for prettier network plots.
    No external libraries needed.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    pos = rng.random((N, 2)) - 0.5

    if k is None:
        k = 1.0 / np.sqrt(max(N, 1))

    for _ in range(iterations):
        disp = np.zeros((N, 2), dtype=float)

        # Repulsion
        for i in range(N):
            for j in range(i + 1, N):
                delta = pos[i] - pos[j]
                dist = np.linalg.norm(delta) + 1e-6
                force = (k * k) / dist
                direction = delta / dist
                disp[i] += direction * force
                disp[j] -= direction * force

        # Attraction
        for i, j in edges:
            delta = pos[i] - pos[j]
            dist = np.linalg.norm(delta) + 1e-6
            force = (dist * dist) / k
            direction = delta / dist
            disp[i] -= direction * force
            disp[j] += direction * force

        pos += 0.01 * disp
        pos = np.clip(pos, -1.5, 1.5)

    return pos


def draw_frame(ax_net, ax_ts, result, pos, t, show_legend=True):
    """
    Draw one frame:
      - left: network at time t
      - right: time series up to t
    """
    state = result["state_history"][t]
    edges = result["edge_history"][t]
    infected_fraction = result["infected_fraction"]
    rewire_counts = result["rewire_counts"]
    S = result["susceptible_counts"]
    I = result["infected_counts"]
    R = result["recovered_counts"]
    params = result["params"]

    ax_net.clear()
    ax_ts.clear()

    # -------------------------
    # Network plot
    # -------------------------
    for i, j in edges:
        ax_net.plot(
            [pos[i, 0], pos[j, 0]],
            [pos[i, 1], pos[j, 1]],
            linewidth=0.8,
            alpha=0.25,
        )

    susceptible = np.where(state == 0)[0]
    infected = np.where(state == 1)[0]
    recovered = np.where(state == 2)[0]

    if len(susceptible) > 0:
        ax_net.scatter(
            pos[susceptible, 0], pos[susceptible, 1],
            s=60, label="Susceptible", alpha=0.95
        )
    if len(infected) > 0:
        ax_net.scatter(
            pos[infected, 0], pos[infected, 1],
            s=80, label="Infected", alpha=0.95, marker="o"
        )
    if len(recovered) > 0:
        ax_net.scatter(
            pos[recovered, 0], pos[recovered, 1],
            s=60, label="Recovered", alpha=0.95, marker="s"
        )

    ax_net.set_title(
        f"Adaptive-network SIR\n"
        f"t={t} | S={S[t]}  I={I[t]}  R={R[t]}  rewires={rewire_counts[t]}"
    )
    ax_net.set_xticks([])
    ax_net.set_yticks([])
    ax_net.set_aspect("equal")

    if show_legend:
        ax_net.legend(loc="upper right", frameon=True)

    subtitle = (
        f"beta={params['beta']}, gamma={params['gamma']}, rho={params['rho']}\n"
        f"N={params['N']}, p_edge={params['p_edge']}, initial infected={params['n_infected0']}"
    )
    ax_net.text(
        0.02, 0.02, subtitle,
        transform=ax_net.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    # -------------------------
    # Time series panel
    # -------------------------
    times = np.arange(len(infected_fraction))
    ax_ts.plot(times[:t + 1], infected_fraction[:t + 1], linewidth=2, label="Infected fraction")
    ax_ts.plot(times[:t + 1], rewire_counts[:t + 1] / max(1, params["N"]), linewidth=2,
               label="Rewires per step / N")
    ax_ts.set_xlim(0, params["T"])
    ax_ts.set_ylim(0, max(1.0, np.max(rewire_counts / max(1, params["N"])) + 0.05))
    ax_ts.set_title("Epidemic and rewiring over time")
    ax_ts.set_xlabel("Time step")
    ax_ts.set_ylabel("Value")
    ax_ts.grid(True, alpha=0.3)
    ax_ts.legend(frameon=True)


def animate_simulation(
    result,
    layout_seed=0,
    interval=400,
    figsize=(13, 6),
    repeat=False,
):
    """
    Animate the stored simulation history.
    """
    N = result["params"]["N"]
    initial_edges = result["edge_history"][0]
    pos = spring_layout_simple(N, initial_edges, rng=np.random.default_rng(layout_seed))

    fig, (ax_net, ax_ts) = plt.subplots(1, 2, figsize=figsize)

    def update(frame):
        draw_frame(ax_net, ax_ts, result, pos, frame)

    anim = FuncAnimation(
        fig,
        update,
        frames=len(result["state_history"]),
        interval=interval,
        repeat=repeat,
    )
    plt.tight_layout()
    plt.show()
    return anim


def plot_summary(result, layout_seed=0, figsize=(13, 10)):
    """
    Static summary:
      1. final network
      2. S/I/R counts over time
      3. infected fraction + rewires
      4. final degree histogram
    """
    N = result["params"]["N"]
    final_t = result["params"]["T"]
    final_edges = result["edge_history"][-1]
    pos = spring_layout_simple(N, result["edge_history"][0], rng=np.random.default_rng(layout_seed))

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax_net, ax_counts, ax_frac, ax_deg = axes.flatten()

    # Final network
    state = result["state_history"][-1]
    for i, j in final_edges:
        ax_net.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], linewidth=0.8, alpha=0.25)

    susceptible = np.where(state == 0)[0]
    infected = np.where(state == 1)[0]
    recovered = np.where(state == 2)[0]

    if len(susceptible) > 0:
        ax_net.scatter(pos[susceptible, 0], pos[susceptible, 1], s=60, label="Susceptible")
    if len(infected) > 0:
        ax_net.scatter(pos[infected, 0], pos[infected, 1], s=80, label="Infected")
    if len(recovered) > 0:
        ax_net.scatter(pos[recovered, 0], pos[recovered, 1], s=60, marker="s", label="Recovered")

    ax_net.set_title(f"Final network at t={final_t}")
    ax_net.set_xticks([])
    ax_net.set_yticks([])
    ax_net.set_aspect("equal")
    ax_net.legend(frameon=True)

    # S/I/R counts
    t = np.arange(final_t + 1)
    ax_counts.plot(t, result["susceptible_counts"], label="Susceptible")
    ax_counts.plot(t, result["infected_counts"], label="Infected")
    ax_counts.plot(t, result["recovered_counts"], label="Recovered")
    ax_counts.set_title("Population counts over time")
    ax_counts.set_xlabel("Time step")
    ax_counts.set_ylabel("Number of nodes")
    ax_counts.grid(True, alpha=0.3)
    ax_counts.legend(frameon=True)

    # Infected fraction + rewires
    ax_frac.plot(t, result["infected_fraction"], linewidth=2, label="Infected fraction")
    ax_frac.plot(t, result["rewire_counts"], linewidth=2, label="Rewires per step")
    ax_frac.set_title("Epidemic and rewiring")
    ax_frac.set_xlabel("Time step")
    ax_frac.set_ylabel("Value")
    ax_frac.grid(True, alpha=0.3)
    ax_frac.legend(frameon=True)

    # Degree histogram
    deg = result["degree_histogram"]
    ax_deg.bar(np.arange(len(deg)), deg)
    ax_deg.set_title("Final degree histogram")
    ax_deg.set_xlabel("Degree (30 means 30+)")
    ax_deg.set_ylabel("Number of nodes")
    ax_deg.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def run_demo(
    beta=0.25,
    gamma=0.08,
    rho=0.20,
    N=60,
    p_edge=0.06,
    n_infected0=3,
    T=80,
    seed=7,
    verbose=True,
    animate=True,
    show_summary=True,
    interval=350,
):
    """
    Convenience function with lots of visible output.
    """
    rng = np.random.default_rng(seed)

    result = simulate_with_history(
        beta=beta,
        gamma=gamma,
        rho=rho,
        N=N,
        p_edge=p_edge,
        n_infected0=n_infected0,
        T=T,
        rng=rng,
        verbose=verbose,
    )

    print("\nFinal summary")
    print("-" * 72)
    print(f"Final susceptible count : {result['susceptible_counts'][-1]}")
    print(f"Final infected count    : {result['infected_counts'][-1]}")
    print(f"Final recovered count   : {result['recovered_counts'][-1]}")
    print(f"Peak infected count     : {np.max(result['infected_counts'])}")
    print(f"Peak infected fraction  : {np.max(result['infected_fraction']):.3f}")
    print(f"Total rewires           : {np.sum(result['rewire_counts'])}")
    print("-" * 72)

    if show_summary:
        plot_summary(result)

    if animate:
        animate_simulation(result, interval=interval)

    return result


# ============================================================
# DEFAULT COMMAND
# ============================================================
if __name__ == "__main__":
    run_demo()