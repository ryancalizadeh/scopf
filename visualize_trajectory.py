"""
Time-domain trajectory comparison across algorithms.

Given the raw per-algorithm results of one experiment_runner() call (as saved,
nested under a sweep key, by main.py._run_sweep), plots the first run's
trajectory from each algorithm against the others to check that they agree
and respect the problem's constraints:

  - plot_power_comparison:       net injection at every bus, all algorithms overlaid
  - plot_temperature_comparison: temperature at every thermal bus, with the
                                  time-varying comfort deadband
  - plot_battery_comparison:     SOC at every battery bus, with its bounds

Trajectories from centralized.solve and admm_vanilla.solve/admm_parallel.solve
disagree on layout (see algorithms/centralized.py vs algorithms/common.py):
centralized's "temp"/"soc" are narrow (width n_thermal/n_battery) and
absolute, while ADMM's are full width (n_buses), zero off their device
columns, and "temp" is shifted by thermal_T0. _thermal_series/_battery_series
below normalize both to an absolute per-device series.

Use save_trajectory_comparison() to write all three to disk; see
plot_trajectory.py for a command-line entry point over a results/*.pkl file.
"""
import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from Config import Config
from ExperimentResult import ExperimentResult
from Trajectory import Trajectory

# Chart chrome, matching test_centralized.py's plots.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_COLOR = "#d9d8d4"
THERMAL_COLOR = "#1baf7a"

# Categorical slots 1/2/3 of the reference palette (validated all-pairs,
# light mode, CVD dE 9.2 / normal-vision 24.0). Algorithms outside the known
# three fall back to the next slots, in sorted order, so the mapping stays
# deterministic.
_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
_PREFERRED_ORDER = ["centralized", "admm"]

# admm_parallel only swaps the bus-projection executor (see
# algorithms/admm_parallel.py); it solves the identical problem to
# admm_vanilla bit for bit, so only one is worth plotting.
_ADMM_VARIANTS = ("admm_vanilla", "admm_parallel")

BOUND_DASH = (0, (3, 2))


def _algorithm_colors(algorithms) -> Dict[str, str]:
    ordered = [a for a in _PREFERRED_ORDER if a in algorithms]
    ordered += sorted(a for a in algorithms if a not in _PREFERRED_ORDER)
    if len(ordered) > len(_PALETTE):
        raise ValueError(f"only {len(_PALETTE)} categorical colors available for {len(ordered)} algorithms")
    return dict(zip(ordered, _PALETTE))


def _first_by_algorithm(results: Dict[str, List[ExperimentResult]]) -> Dict[str, ExperimentResult]:
    if not results:
        raise ValueError("results is empty; nothing to plot")
    first = {}
    for algo, runs in results.items():
        if not runs:
            raise ValueError(f"no runs for algorithm {algo!r}")
        er = runs[0]
        if not isinstance(er, ExperimentResult):
            raise TypeError(
                f"expected an ExperimentResult, got {type(er).__name__} for algorithm {algo!r}; "
                "did you pass an aggregated results pickle instead of a raw one?"
            )
        first[algo] = er
    return _canonicalize_admm(first)


def _canonicalize_admm(by_algo: Dict[str, ExperimentResult]) -> Dict[str, ExperimentResult]:
    """Drops admm_parallel if admm_vanilla is also present (or vice versa)
    and relabels whichever survives as "admm", since they solve identically."""
    present = [a for a in _ADMM_VARIANTS if a in by_algo]
    if not present:
        return by_algo
    keep = present[0]
    merged = {algo: er for algo, er in by_algo.items() if algo not in _ADMM_VARIANTS}
    merged["admm"] = by_algo[keep]
    return merged


def _reference_config(by_algo: Dict[str, ExperimentResult]) -> Config:
    """The first algorithm's config, after checking every algorithm ran the
    same problem (experiment_runner solves the same configs for each)."""
    configs = {algo: er.config for algo, er in by_algo.items()}
    reference_algo, reference = next(iter(configs.items()))
    fields = ("n_buses", "n_gens", "n_loads", "n_thermal", "n_battery", "N", "dt")
    for algo, config in configs.items():
        mismatched = [f for f in fields if getattr(config, f) != getattr(reference, f)]
        if mismatched:
            raise ValueError(
                f"config mismatch on {mismatched}: {algo!r} does not match "
                f"the reference config from {reference_algo!r}"
            )
    return reference


def _bus_layout(config: Config) -> List[Tuple[str, int]]:
    """(device kind, device index) for every bus, in the
    [gens | loads | thermal | batteries] layout Config uses throughout."""
    layout: List[Tuple[str, int]] = []
    layout += [("gen", i) for i in range(config.n_gens)]
    layout += [("load", i) for i in range(config.n_loads)]
    layout += [("thermal", i) for i in range(config.n_thermal)]
    layout += [("battery", i) for i in range(config.n_battery)]
    return layout


def _power_bounds(config: Config, kind: str, idx: int) -> Tuple[float, float]:
    if kind == "gen":
        return float(config.gen_P_min[idx]), float(config.gen_P_max[idx])
    if kind == "thermal":
        # bus convention is the negative of device convention (heating draws power)
        return float(-config.thermal_p_max[idx]), float(-config.thermal_p_min[idx])
    if kind == "battery":
        return float(config.battery_p_min[idx]), float(config.battery_p_max[idx])
    raise ValueError(kind)


def _thermal_series(traj: Trajectory, config: Config, k: int) -> np.ndarray:
    """Actual temperature (deg C) for thermal device k, normalizing the two
    trajectory layouts (see module docstring)."""
    temp = traj["temp"]
    if temp.shape[1] == config.n_buses:
        bus = config.n_gens + config.n_loads + k
        return temp[:, bus] + config.thermal_T0[k]
    return temp[:, k]


def _battery_series(traj: Trajectory, config: Config, k: int) -> np.ndarray:
    """SOC for battery device k, normalizing the two trajectory layouts."""
    soc = traj["soc"]
    if soc.shape[1] == config.n_buses:
        bus = config.n_gens + config.n_loads + config.n_thermal + k
        return soc[:, bus]
    return soc[:, k]


def _grid_figure(n_panels: int, panel_size: Tuple[float, float] = (3.6, 2.3), min_width: float = 6.8):
    ncols = max(1, min(6, int(np.ceil(np.sqrt(n_panels)))))
    nrows = int(np.ceil(n_panels / ncols))
    width = max(min_width, panel_size[0] * ncols)
    height = panel_size[1] * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), squeeze=False, sharex=True)
    fig.patch.set_facecolor(SURFACE)
    flat = list(axes.flatten())
    panels, spare = flat[:n_panels], flat[n_panels:]
    for ax in spare:
        ax.axis("off")
    for ax in panels:
        ax.set_facecolor(SURFACE)
    return fig, panels, ncols


def _style_panel(ax, title: str) -> None:
    ax.set_title(title, color=TEXT_SECONDARY, fontsize=8.5, loc="left", pad=3)
    ax.grid(True, color=GRID_COLOR, linewidth=0.5, alpha=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID_COLOR)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=7)


def _plural(n: int, singular: str, plural: str) -> str:
    return f"{n} {singular if n == 1 else plural}"


def _empty_figure(message: str) -> Figure:
    fig, ax = plt.subplots(figsize=(6, 2))
    fig.patch.set_facecolor(SURFACE)
    ax.axis("off")
    ax.text(0.5, 0.5, message, ha="center", va="center", color=TEXT_SECONDARY, fontsize=11)
    return fig


def _plot_algorithms(ax, hours: np.ndarray, colors: Dict[str, str], series: Dict[str, np.ndarray]) -> None:
    """
    Overlaid line per algorithm, drawn widest-and-lowest-zorder first and
    thinnest-and-highest-zorder last. When the lines coincide (the expected
    case here) only the top, thinnest one reads; if any algorithm actually
    disagrees, the wider line(s) underneath show as a halo instead of being
    silently hidden by draw order.
    """
    widths = np.linspace(4.0, 1.4, len(colors)) if len(colors) > 1 else [1.8]
    for zorder, ((algo, color), lw) in enumerate(zip(colors.items(), widths)):
        ax.plot(hours, series[algo], color=color, linewidth=lw, zorder=zorder + 2)


def _finish_grid(fig: Figure, panels, ncols: int, handles, labels,
                  title: str, xlabel: str, ylabel: str, subtitle_lines: int = 0) -> None:
    fig_w, fig_h = fig.get_size_inches()
    top_in = 0.48 + 0.32 * subtitle_lines
    bottom_in = 0.95
    left_in = 0.62
    right_in = 0.12
    fig.subplots_adjust(
        top=1 - top_in / fig_h, bottom=bottom_in / fig_h,
        left=left_in / fig_w, right=1 - right_in / fig_w,
        hspace=0.65, wspace=0.4,
    )

    n_panels = len(panels)
    for i, ax in enumerate(panels):
        if i + ncols < n_panels:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=8)

    fig.suptitle(title, color=TEXT_PRIMARY, fontsize=11, x=0.02, y=1 - 0.05 / fig_h, ha="left", va="top")
    fig.supylabel(ylabel, color=TEXT_SECONDARY, fontsize=10)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               fontsize=8.5, frameon=False, labelcolor=TEXT_SECONDARY,
               bbox_to_anchor=(0.5, 0.22 / fig_h))


def plot_power_comparison(results: Dict[str, List[ExperimentResult]]) -> Figure:
    """Net injection at every bus, all algorithms overlaid, against each
    device's power bounds (or, for loads, the exact demand they must serve)."""
    by_algo = _first_by_algorithm(results)
    config = _reference_config(by_algo)
    colors = _algorithm_colors(by_algo.keys())
    hours = np.arange(config.N) * config.dt
    layout = _bus_layout(config)

    fig, axes, ncols = _grid_figure(config.n_buses)

    for bus, (kind, idx) in enumerate(layout):
        ax = axes[bus]
        series = {algo: by_algo[algo].trajectory["p"][:, bus] for algo in colors}
        if kind == "load":
            ax.plot(hours, -config.load_P[idx], color=TEXT_PRIMARY, linewidth=1,
                     linestyle=(0, (1, 1)), alpha=0.5, zorder=1)
        else:
            lo, hi = _power_bounds(config, kind, idx)
            ax.axhline(lo, color=TEXT_SECONDARY, linewidth=1, linestyle=BOUND_DASH, alpha=0.6, zorder=1)
            ax.axhline(hi, color=TEXT_SECONDARY, linewidth=1, linestyle=BOUND_DASH, alpha=0.6, zorder=1)
        ax.axhline(0, color=TEXT_SECONDARY, linewidth=0.6, alpha=0.3, zorder=1)
        _plot_algorithms(ax, hours, colors, series)
        _style_panel(ax, f"bus {bus} ({kind})")

    handles = [Line2D([0], [0], color=c, linewidth=2.5) for c in colors.values()]
    labels = list(colors.keys())
    handles.append(Line2D([0], [0], color=TEXT_SECONDARY, linewidth=1, linestyle=BOUND_DASH, alpha=0.6))
    labels.append("power limit")
    if config.n_loads:
        handles.append(Line2D([0], [0], color=TEXT_PRIMARY, linewidth=1, linestyle=(0, (1, 1)), alpha=0.5))
        labels.append("load target")

    subtitle = "  |  ".join(f"{a}: obj {r.obj:.1f}, {r.runtime:.2f}s" for a, r in by_algo.items())
    _finish_grid(fig, axes, ncols, handles, labels,
                 title=f"Power schedule by bus — {config.n_buses} buses\n{subtitle}",
                 xlabel="hour of day", ylabel="net injection (p.u.)", subtitle_lines=1)
    return fig


def plot_temperature_comparison(results: Dict[str, List[ExperimentResult]]) -> Figure:
    """Temperature at every thermal bus, all algorithms overlaid, against the
    time-varying comfort deadband."""
    by_algo = _first_by_algorithm(results)
    config = _reference_config(by_algo)
    if config.n_thermal == 0:
        return _empty_figure("No thermal buses in this configuration.")
    colors = _algorithm_colors(by_algo.keys())
    hours = np.arange(config.N) * config.dt

    fig, axes, ncols = _grid_figure(config.n_thermal)

    for k in range(config.n_thermal):
        bus = config.n_gens + config.n_loads + k
        ax = axes[k]
        ax.fill_between(hours, config.thermal_T_min[k], config.thermal_T_max[k],
                         color=THERMAL_COLOR, alpha=0.10, linewidth=0, zorder=1)
        series = {algo: _thermal_series(by_algo[algo].trajectory, config, k) for algo in colors}
        _plot_algorithms(ax, hours, colors, series)
        _style_panel(ax, f"bus {bus} (thermal)")

    handles = [Line2D([0], [0], color=c, linewidth=2.5) for c in colors.values()]
    labels = list(colors.keys())
    handles.append(Patch(facecolor=THERMAL_COLOR, alpha=0.10)) # type: ignore
    labels.append("comfort deadband")

    _finish_grid(fig, axes, ncols, handles, labels,
                 title=f"Thermal bus temperatures — {_plural(config.n_thermal, 'thermal bus', 'thermal buses')}",
                 xlabel="hour of day", ylabel="temperature (°C)")
    return fig


def plot_battery_comparison(results: Dict[str, List[ExperimentResult]]) -> Figure:
    """SOC at every battery bus, all algorithms overlaid, against its bounds."""
    by_algo = _first_by_algorithm(results)
    config = _reference_config(by_algo)
    if config.n_battery == 0:
        return _empty_figure("No battery buses in this configuration.")
    colors = _algorithm_colors(by_algo.keys())
    hours = np.arange(config.N) * config.dt

    fig, axes, ncols = _grid_figure(config.n_battery)

    for k in range(config.n_battery):
        bus = config.n_gens + config.n_loads + config.n_thermal + k
        ax = axes[k]
        ax.axhline(config.battery_q_min[k], color=TEXT_SECONDARY, linewidth=1,
                   linestyle=BOUND_DASH, alpha=0.6, zorder=1)
        ax.axhline(config.battery_q_max[k], color=TEXT_SECONDARY, linewidth=1,
                   linestyle=BOUND_DASH, alpha=0.6, zorder=1)
        series = {algo: _battery_series(by_algo[algo].trajectory, config, k) for algo in colors}
        _plot_algorithms(ax, hours, colors, series)
        _style_panel(ax, f"bus {bus} (battery)")

    handles = [Line2D([0], [0], color=c, linewidth=2.5) for c in colors.values()]
    labels = list(colors.keys())
    handles.append(Line2D([0], [0], color=TEXT_SECONDARY, linewidth=1, linestyle=BOUND_DASH, alpha=0.6))
    labels.append("SOC limit")

    _finish_grid(fig, axes, ncols, handles, labels,
                 title=f"Battery state of charge — {_plural(config.n_battery, 'battery bus', 'battery buses')}",
                 xlabel="hour of day", ylabel="SOC (p.u.)")
    return fig


def save_trajectory_comparison(results: Dict[str, List[ExperimentResult]],
                                results_dir: str = "results",
                                run_name: str = "trajectory") -> Tuple[str, str, str]:
    """Writes the power/temperature/battery comparison figures to results_dir,
    named f"{run_name}_{kind}_{timestamp}.png". Returns the three paths."""
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    paths = []
    for kind, fig in (
        ("power", plot_power_comparison(results)),
        ("temperature", plot_temperature_comparison(results)),
        ("battery", plot_battery_comparison(results)),
    ):
        path = os.path.join(results_dir, f"{run_name}_{kind}_{timestamp}.png")
        fig.savefig(path, dpi=140, facecolor=fig.get_facecolor())
        plt.close(fig)
        paths.append(path)
    return tuple(paths)
