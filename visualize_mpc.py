"""
Closed-loop diagnostics for receding-horizon runs (mpc.ClosedLoopResult).

plot_closed_loop_diagnostics draws four stacked panels against hour of day,
one line per closed-loop run:

  - stage cost of the realized dispatch, with the day-ahead plan's for reference
  - solver runtime per MPC step
  - worst bound excess per step (max over mpc.violations_by_step)
  - one-step prediction error per step (plan's first step vs realized)

Colors follow the labels of visualize_trajectory ("<algo> closed loop",
"day-ahead"), so a run has the same color in every figure.
"""
from typing import Dict, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from algorithms.base import SolveResult
from mpc import ClosedLoopResult
from visualize_trajectory import (SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, BOUND_DASH,
                                  _algorithm_colors, _style_panel)

# Log panels plot exact zeros at this floor, well below solver precision.
LOG_FLOOR = 1e-12
VIOLATION_TOL = 1e-6


def closed_loop_label(algorithm: str) -> str:
    return f"{algorithm} closed loop"


DAY_AHEAD_LABEL = "day-ahead"


def _floored(values: np.ndarray) -> np.ndarray:
    return np.maximum(values, LOG_FLOOR)


def plot_closed_loop_diagnostics(results: Dict[str, ClosedLoopResult],
                                  day_ahead: Optional[SolveResult] = None) -> Figure:
    if not results:
        raise ValueError("results is empty; nothing to plot")
    config = next(iter(results.values())).config
    labels = [closed_loop_label(a) for a in results] + ([DAY_AHEAD_LABEL] if day_ahead is not None else [])
    colors = _algorithm_colors(labels)
    dt = config.dt

    fig, axes = plt.subplots(4, 1, figsize=(9.0, 9.6), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
    ax_cost, ax_time, ax_viol, ax_pred = axes

    if day_ahead is not None:
        p_gen = day_ahead.trajectory["p"][:, :config.n_gens]
        cost = (config.gen_cost_alpha * p_gen ** 2 + config.gen_cost_beta * p_gen).sum(axis=1)
        ax_cost.plot(np.arange(len(cost)) * dt, cost, color=colors[DAY_AHEAD_LABEL],
                     linewidth=3.2, alpha=0.55, zorder=2)

    for algo, res in results.items():
        color = colors[closed_loop_label(algo)]
        hours = np.arange(res.n_steps) * dt
        ax_cost.plot(hours, res.stage_cost, color=color, linewidth=1.6, zorder=3)
        ax_time.plot(hours, res.solve_runtime, color=color, linewidth=1.6, zorder=3)
        worst = np.max(np.vstack(list(res.violations.values())), axis=0) if res.n_steps else np.zeros(0)
        ax_viol.plot(hours, _floored(worst), color=color, linewidth=1.6, zorder=3)
        pred = np.max(np.vstack(list(res.prediction_error.values())), axis=0) if res.n_steps else np.zeros(0)
        ax_pred.plot(hours, _floored(pred), color=color, linewidth=1.6, zorder=3)
        if res.failed_step is not None:
            for ax in axes:
                ax.axvline(res.failed_step * dt, color=color, linewidth=1, linestyle=BOUND_DASH, zorder=1)

    ax_viol.axhline(VIOLATION_TOL, color=TEXT_SECONDARY, linewidth=1, linestyle=BOUND_DASH, alpha=0.6, zorder=1)
    for ax in (ax_time, ax_viol, ax_pred):
        ax.set_yscale("log")
        ax.tick_params(which="minor", colors=TEXT_SECONDARY, labelsize=7)
    ax_viol.set_ylim(LOG_FLOOR / 3, None)
    ax_pred.set_ylim(LOG_FLOOR / 3, None)

    _style_panel(ax_cost, "stage cost of the realized dispatch")
    _style_panel(ax_time, "solve time per MPC step (s)")
    _style_panel(ax_viol, f"worst bound excess per step (p.u. / °C; zero drawn at {LOG_FLOOR:.0e})")
    _style_panel(ax_pred, "one-step prediction error, plan vs plant (max over p, soc, temp)")
    ax_pred.set_xlabel("hour of day", color=TEXT_SECONDARY, fontsize=8)
    ax_pred.set_xlim(0, config.N * dt)
    ax_pred.set_xticks(np.arange(0, config.N * dt + 1, 3))

    handles = [Line2D([0], [0], color=c, linewidth=2.5) for c in colors.values()]
    legend_labels = list(colors.keys())
    handles.append(Line2D([0], [0], color=TEXT_SECONDARY, linewidth=1, linestyle=BOUND_DASH, alpha=0.6))
    legend_labels.append(f"violation tolerance {VIOLATION_TOL:.0e}")

    horizons = {res.horizon for res in results.values()}
    horizon = "shrinking to midnight" if horizons == {None} else \
        ", ".join(f"{h * dt:g} h" for h in sorted(h for h in horizons if h is not None))
    fig.suptitle(f"Closed-loop diagnostics — {config.n_buses} buses, seed {config.seed}, window {horizon}",
                 color=TEXT_PRIMARY, fontsize=11, x=0.02, ha="left")
    fig.legend(handles, legend_labels, loc="lower center", ncol=len(legend_labels), fontsize=8.5,
               frameon=False, labelcolor=TEXT_SECONDARY)
    fig.subplots_adjust(top=0.93, bottom=0.1, left=0.1, right=0.97, hspace=0.35)
    return fig
