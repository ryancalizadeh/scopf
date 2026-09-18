"""
Runs the centralized dynamic OPF on an 8-bus config and plots the solution.

    python test_centralized.py

Checks the returned trajectory against every constraint the solver imposes, then
writes three separate figures to results/:
  - power schedule of every bus, coloured by device type
  - thermal bus temperatures with ambient and the comfort deadband
  - battery state of charge with its limits
"""
import os
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Config import Config
from algorithms.centralized import solve

RESULTS_DIR = "results"

# Categorical slots 1/2/3/7 of the reference palette. Validated together
# (all pairs, light mode): CVD dE 9.2, normal-vision dE 16.3, both above floor.
TYPE_COLORS = {
    "gen": "#2a78d6",
    "load": "#eb6834",
    "thermal": "#1baf7a",
    "battery": "#4a3aa7",
}
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#d9d8d4"


def bus_types(config: Config) -> list[str]:
    """Device type of each bus, in the [gens | loads | thermal | batteries] layout."""
    return (["gen"] * config.n_gens + ["load"] * config.n_loads
            + ["thermal"] * config.n_thermal + ["battery"] * config.n_battery)


def check_solution(config: Config, traj) -> None:
    """Asserts the trajectory satisfies the constraints; raises AssertionError if not."""
    tol = 1e-6
    N, dt = config.N, config.dt
    n_gens, n_loads = config.n_gens, config.n_loads
    n_thermal = config.n_thermal

    p = traj["p"]          # (N, n_buses) net injection
    theta = traj["theta"]
    soc = traj["soc"]
    temp = traj["temp"]

    # Lossless DC flow: injections sum to zero at every step.
    imbalance = np.abs(p.sum(axis=1)).max()
    assert imbalance < 1e-6, f"power imbalance {imbalance:.2e}"

    # Generators: capacity and ramping.
    p_gen = p[:, :n_gens]
    assert (p_gen >= config.gen_P_min - tol).all(), "generator below P_min"
    assert (p_gen <= config.gen_P_max + tol).all(), "generator above P_max"
    ramp = np.diff(p_gen, axis=0)
    assert (ramp >= config.gen_R_min - tol).all(), "generator ramps down too fast"
    assert (ramp <= config.gen_R_max + tol).all(), "generator ramps up too fast"

    # Fixed loads are served exactly.
    served = -p[:, n_gens:n_gens + n_loads]
    assert np.abs(served - config.load_P.T).max() < tol, "fixed load not served"

    # Thermal: comfort band, power box, and the flipped dynamics (p > 0 heats).
    p_thermal = -p[:, n_gens + n_loads:n_gens + n_loads + n_thermal]
    assert (temp >= config.thermal_T_min.T - 1e-5).all(), "temperature below deadband"
    assert (temp <= config.thermal_T_max.T + 1e-5).all(), "temperature above deadband"
    assert (p_thermal >= config.thermal_p_min - tol).all(), "thermal below p_min"
    assert (p_thermal <= config.thermal_p_max + tol).all(), "thermal above p_max"
    # temp[t] is the state at step t, so the step from t to t+1 uses the control
    # at t+1 and the ambient at t+1; temp[0] follows from the initial condition.
    decay = 1.0 - config.thermal_mu / config.thermal_c
    gain = config.thermal_eta / config.thermal_c
    ambient = config.thermal_mu / config.thermal_c
    predicted = (decay * temp[:-1] + gain * p_thermal[1:] + ambient * config.thermal_T_amb.T[1:])
    assert np.abs(predicted - temp[1:]).max() < 1e-5, "thermal dynamics violated"
    first = (decay * config.thermal_T0 + gain * p_thermal[0] + ambient * config.thermal_T_amb.T[0])
    assert np.abs(first - temp[0]).max() < 1e-5, "thermal initial step violated"
    assert p_thermal.mean() > 0, "expected net heating; the thermal sign may be flipped"

    # Battery: SOC bounds, dynamics, and the initial condition.
    # soc[t] is the state at step t, reached by applying p_batt[t] to soc[t-1];
    # soc[0] follows from battery_q0 and soc[-1] is the terminal state.
    p_batt = p[:, config.n_gens + n_loads + n_thermal:]
    assert (soc >= config.battery_q_min - tol).all(), "SOC below q_min"
    assert (soc <= config.battery_q_max + tol).all(), "SOC above q_max"
    predicted_soc = soc[:-1] - dt * p_batt[1:]
    assert np.abs(predicted_soc - soc[1:]).max() < 1e-6, "SOC dynamics violated"
    assert np.abs((config.battery_q0 - dt * p_batt[0]) - soc[0]).max() < 1e-6, "SOC initial step"
    assert np.abs(soc[-1] - config.battery_qT).max() < 1e-5, "SOC terminal condition"

    # Line flows within limits.
    worst = 0.0
    for i, j in config.lines:
        flow = -config.B[i, j] * (theta[:, i] - theta[:, j])
        limit = config.line_flow_limits[i, j]
        assert np.abs(flow).max() <= limit + 1e-6, f"line ({i},{j}) over limit"
        worst = max(worst, np.abs(flow).max())
    print(f"  max |line flow| {worst:.3f} (limit {config.line_flow_limits[0, 1]:.1f})")
    print(f"  mean thermal power {p_thermal.mean():+.3f} p.u. (positive = net heating)")


def style(ax, ylabel: str) -> None:
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=10)
    ax.grid(True, color=GRID, linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)


DASHES = [(None, None), (5, 2), (1.5, 1.5), (7, 2, 1.5, 2)]


def _save(fig, name: str, timestamp: str) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"centralized_8bus_{name}_{timestamp}.png")
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def plot_power(config: Config, traj, obj: float, runtime: float, timestamp: str) -> str:
    """Power schedule of every bus, one line per bus, coloured by device type."""
    hours = np.arange(config.N) * config.dt
    types = bus_types(config)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor("#fcfcfb")

    # Buses of the same type share a hue and are separated by line style, so
    # identity never rests on colour alone.
    for bus, kind in enumerate(types):
        within = [b for b, k in enumerate(types) if k == kind].index(bus)
        ax.plot(hours, traj["p"][:, bus], color=TYPE_COLORS[kind], linewidth=2,
                dashes=DASHES[within % len(DASHES)], label=f"bus {bus} ({kind})")
    ax.axhline(0, color=TEXT_SECONDARY, linewidth=1, alpha=0.5)
    ax.set_title(
        f"Power schedule by bus  —  positive = injection, negative = consumption\n"
        f"{config.n_buses} buses, cost {obj:.1f}, solved in {runtime:.2f}s",
        color=TEXT_PRIMARY, fontsize=11, loc="left", pad=10)
    style(ax, "net injection (p.u.)")
    ax.set_xlabel("hour of day", color=TEXT_SECONDARY, fontsize=10)
    ax.set_xlim(0, config.T)
    ax.set_xticks(np.arange(0, config.T + 1, 3))
    # Legend sits below the axes so it never covers the traces.
    ax.legend(ncol=4, fontsize=8, frameon=False, labelcolor=TEXT_SECONDARY,
              loc="upper center", bbox_to_anchor=(0.5, -0.16))

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, "power", timestamp)


def plot_thermal(config: Config, traj, timestamp: str) -> str:
    """Thermal bus temperatures, ambient temperature, and the comfort deadband."""
    hours = np.arange(config.N) * config.dt

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor("#fcfcfb")

    # Config tiles one profile across thermal buses, so shade the band once
    # (shading per bus would double the alpha wherever the bands overlap). If
    # the bands ever differ per bus, fall back to one shaded band each.
    shared_band = (np.ptp(config.thermal_T_min, axis=0).max() == 0
                   and np.ptp(config.thermal_T_max, axis=0).max() == 0)
    for k in range(1 if shared_band else config.n_thermal):
        ax.fill_between(hours, config.thermal_T_min[k], config.thermal_T_max[k],
                        color=TYPE_COLORS["thermal"], alpha=0.10,
                        label="comfort deadband" if k == 0 else None)
    for k in range(config.n_thermal):
        bus = config.n_gens + config.n_loads + k
        ax.plot(hours, traj["temp"][:, k], color=TYPE_COLORS["thermal"], linewidth=2,
                dashes=DASHES[k % len(DASHES)], label=f"bus {bus} temperature")
        ax.plot(hours, config.thermal_T_amb[k], color=TEXT_SECONDARY, linewidth=1.5,
                dashes=(4, 3), alpha=0.7, label="ambient" if k == 0 else None)
    ax.set_title("Thermal bus temperatures, ambient, and comfort deadband",
                 color=TEXT_PRIMARY, fontsize=11, loc="left", pad=10)
    style(ax, "temperature (°C)")
    ax.set_xlabel("hour of day", color=TEXT_SECONDARY, fontsize=10)
    ax.set_xlim(0, config.T)
    ax.set_xticks(np.arange(0, config.T + 1, 3))
    ax.legend(ncol=4, fontsize=8, frameon=False, labelcolor=TEXT_SECONDARY,
              loc="upper center", bbox_to_anchor=(0.5, -0.16))

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, "thermal", timestamp)


def plot_battery(config: Config, traj, timestamp: str) -> str:
    """Battery state of charge with its lower/upper limits."""
    hours = np.arange(config.N) * config.dt

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor("#fcfcfb")

    for k in range(config.n_battery):
        bus = config.n_gens + config.n_loads + config.n_thermal + k
        ax.plot(hours, traj["soc"][:, k], color=TYPE_COLORS["battery"], linewidth=2,
                dashes=DASHES[k % len(DASHES)], label=f"bus {bus} SOC")
        for limit in (config.battery_q_min[k], config.battery_q_max[k]):
            ax.axhline(limit, color=TYPE_COLORS["battery"], linewidth=1,
                       dashes=(2, 3), alpha=0.55)
    ax.plot([], [], color=TYPE_COLORS["battery"], linewidth=1, dashes=(2, 3),
            alpha=0.55, label="SOC limits")
    ax.set_title("Battery state of charge", color=TEXT_PRIMARY, fontsize=11, loc="left", pad=10)
    style(ax, "SOC (p.u.)")
    ax.set_xlabel("hour of day", color=TEXT_SECONDARY, fontsize=10)
    ax.set_xlim(0, config.T)
    ax.set_xticks(np.arange(0, config.T + 1, 3))
    ax.legend(ncol=3, fontsize=8, frameon=False, labelcolor=TEXT_SECONDARY,
              loc="upper center", bbox_to_anchor=(0.5, -0.16))

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, "battery", timestamp)


def main() -> None:
    config = Config(n_buses=8)
    print(f"8-bus config: {config.n_gens} gen, {config.n_loads} load, "
          f"{config.n_thermal} thermal, {config.n_battery} battery, N={config.N} steps")

    result = solve(config)
    print(f"solved: objective {result.obj:.2f}, runtime {result.runtime:.3f}s")

    check_solution(config, result.trajectory)
    print("  all constraint checks passed")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for path in (
        plot_power(config, result.trajectory, result.obj, result.runtime, timestamp),
        plot_thermal(config, result.trajectory, timestamp),
        plot_battery(config, result.trajectory, timestamp),
    ):
        print(f"figure written to {path}")


if __name__ == "__main__":
    main()
