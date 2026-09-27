"""
ADMM iteration count against the number of binding line limits, on one small
dynamic-OPF case.

    python line_limit_sweep.py                              # n=20, seed=42, default grid
    python line_limit_sweep.py --limits 2.5 2.0 1.5         # custom uniform limits (p.u.)
    python line_limit_sweep.py --n-buses 15 --seed 3
    python line_limit_sweep.py --replot results/line_limit_sweep_<...>.json

Every line gets the same limit L (overriding Config's 1.5 * max(gen_P_max)),
swept from the config default, where nothing binds, down towards the smallest
feasible L. At each L:

  1. the centralized QP gives the reference optimum and which line limits
     bind: a (line, step) pair binds when its flow is at the limit and its
     dual is nonzero; a line binds when any of its steps does;
  2. admm_parallel solves the same config (admm_vanilla with the executor
     admm_parallel picks, at a raised iteration cap so slow points are not
     censored at the default 1000), and its iteration count, final residuals
     and distance from the centralized optimum are recorded.

Writes to --results-dir:
  line_limit_sweep_<tag>_<timestamp>.json   one record per L, with r/s histories
  line_limit_sweep_<tag>_<timestamp>.png    iterations vs binding lines, pairs, and L
"""
import argparse
import copy
import json
import logging
import os
import sys
from datetime import datetime

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from Config import Config
from algorithms import admm_vanilla
from algorithms.admm_parallel import executor_for
from algorithms.centralized import _build_constraints

logger = logging.getLogger(__name__)

# A (line, step) pair binds when its flow is within _AT_LIMIT of the limit and
# its dual exceeds _DUAL_TOL. On n=20 seed=42 the two tests agree at every L of
# the default grid (no degenerate active constraints), so either alone would do.
_AT_LIMIT = 1e-5
_DUAL_TOL = 1e-4


def with_uniform_limit(config: Config, limit: float) -> Config:
    """A copy of config with every line limit set to limit (p.u.)."""
    c = copy.copy(config)
    c.line_flow_limits = float(limit) * np.ones_like(config.B)
    return c


def centralized_binding(config: Config) -> dict:
    """
    Solves the centralized dynamic OPF (same constraints and cost as
    centralized.solve) and reports which line limits bind at the optimum.
    """
    constraints, v = _build_constraints(config)
    p_gen = v["p_gen"]
    cost = cp.sum(cp.multiply(config.gen_cost_alpha[:, None], cp.square(p_gen))
                  + cp.multiply(config.gen_cost_beta[:, None], p_gen))
    problem = cp.Problem(cp.Minimize(cost), constraints)
    problem.solve(solver=cp.CLARABEL)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"centralized OPF status {problem.status!r}")

    # _build_constraints appends the line limits last, one per line.
    line_constraints = constraints[-len(config.lines):]
    theta = np.asarray(v["theta"].value)
    flows = np.array([-config.B[i, j] * (theta[i] - theta[j]) for i, j in config.lines])  # (lines, N)
    limits = np.array([config.line_flow_limits[i, j] for i, j in config.lines])
    duals = np.array([np.abs(np.asarray(c.dual_value)) for c in line_constraints])        # (lines, N)
    binding = (np.abs(flows) >= limits[:, None] - _AT_LIMIT) & (duals > _DUAL_TOL)

    return {
        "obj": float(problem.value),
        "p": np.asarray(v["P"].value).T,                                      # (N, n_buses)
        "binding": binding.T,                                                 # (N, lines)
        "binding_lines": [[int(i), int(j)] for (i, j), b in zip(config.lines, binding.any(axis=1)) if b],
        "n_binding_lines": int(binding.any(axis=1).sum()),
        "n_binding_pairs": int(binding.sum()),
        "max_line_dual": float(duals.max()) if duals.size else 0.0,
        "sum_line_dual": float(duals[binding].sum()),
    }


class ActiveSetTracker:
    """
    ADMM callback recording which (line, step) limits the Network projection x
    holds at the limit, and the last iteration at which that set changed. After
    it, ADMM has identified its active set and runs on fixed faces of the two
    constraint sets, where convergence is linear, so the iteration count splits
    into identification + linear tail.
    """
    def __init__(self, config: Config):
        self.A = np.zeros((len(config.lines), config.n_buses))
        self.F = np.zeros(len(config.lines))
        for l, (i, j) in enumerate(config.lines):
            self.A[l, i], self.A[l, j] = -config.B[i, j], config.B[i, j]
            self.F[l] = config.line_flow_limits[i, j]
        self.active = None
        self.last_change = 0
        self.n_changes = 0
        self.n_active: list[int] = []

    def __call__(self, iteration, x, z, u, r, s):
        active = np.abs(x["theta"] @ self.A.T) >= self.F - 1e-7                 # (N, lines)
        if self.active is None or not np.array_equal(active, self.active):
            self.last_change = iteration
            self.n_changes += 1
        self.active = active
        self.n_active.append(int(active.sum()))


def tail_rate(r: list, start: int) -> float:
    """Per-iteration contraction of r from iteration start on (least-squares slope of log r)."""
    tail = np.log(np.asarray(r[start:]))
    if tail.size < 10:
        return float("nan")
    return float(np.exp(np.polyfit(np.arange(tail.size), tail, 1)[0]))


def run_point(config: Config, limit: float, max_iterations: int) -> dict:
    c = with_uniform_limit(config, limit)
    ref = centralized_binding(c)
    tracker = ActiveSetTracker(c)

    sol = admm_vanilla.solve(c, parallel=executor_for(c), max_iterations=max_iterations, callback=tracker)
    r, s = sol.convergence.r, sol.convergence.s
    threshold = 1e-3 * np.sqrt(c.n_buses / 8)   # admm_vanilla.solve's stopping threshold
    gens = slice(0, c.n_gens)
    return {
        "limit": float(limit),
        "centralized_obj": ref["obj"],
        "binding_lines": ref["binding_lines"],
        "n_binding_lines": ref["n_binding_lines"],
        "n_binding_pairs": ref["n_binding_pairs"],
        "max_line_dual": ref["max_line_dual"],
        "sum_line_dual": ref["sum_line_dual"],
        "iterations": len(r),
        "converged": bool(r[-1] < threshold and s[-1] < threshold),
        "r_final": float(r[-1]),
        "s_final": float(s[-1]),
        "admm_obj": float(sol.obj),
        "obj_rel_gap": float((sol.obj - ref["obj"]) / abs(ref["obj"])),
        # Generator output is unique (strictly convex cost); thermal and battery
        # schedules need not be, so dispatch_err can be large at a zero gap.
        "gen_dispatch_err": float(np.abs(sol.trajectory["p"][:, gens] - ref["p"][:, gens]).max()),
        "dispatch_err": float(np.abs(sol.trajectory["p"] - ref["p"]).max()),
        "line_violation": float(sol.checks["line_limit"]),
        "runtime": float(sol.runtime),
        "identified_at": tracker.last_change + 1,
        "active_set_changes": tracker.n_changes,
        "active_set_mismatch": int((tracker.active != ref["binding"]).sum()),
        "tail_rate": tail_rate(r, tracker.last_change + 1),
        "r": [float(x) for x in r],
        "s": [float(x) for x in s],
        "n_active": tracker.n_active,
    }


def plot_sweep(records: list, title: str):
    """
    Iterations against binding lines and binding (line, step) pairs, then both
    against the limit itself, with the iteration at which ADMM's active set
    stopped changing and bands marking the number of binding lines.
    """
    ink, muted, grid, band = "#0b0b0b", "#52514e", "#e4e3df", "#f1f0ec"
    total_color, ident_color = "#2a78d6", "#eb6834"
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True,
                             gridspec_kw={"width_ratios": [1, 1, 1.6]})

    def style(ax, xlabel):
        ax.set_xlabel(xlabel, color=muted)
        ax.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(muted)
        ax.tick_params(colors=muted)

    def dots(ax, x, y, color, filled):
        ax.plot(x, y, "o", ms=8, color=color, markerfacecolor=color if filled else "none",
                markeredgecolor="white" if filled else color, markeredgewidth=1.5 if filled else 2)

    for ax, key, xlabel in [(axes[0], "n_binding_lines", "binding lines (centralized optimum)"),
                            (axes[1], "n_binding_pairs", "binding (line, step) pairs")]:
        for rec in records:
            dots(ax, rec[key], rec["iterations"], total_color, rec["converged"])
        style(ax, xlabel)
    axes[0].set_ylabel("ADMM iterations", color=muted)

    # Iterations against L. The config default (far above every flow) would
    # squash the axis, so points above 3 p.u. are left off and noted instead.
    ax = axes[2]
    shown = sorted((rec for rec in records if rec["limit"] <= 3.0), key=lambda rec: -rec["limit"])
    L = np.array([rec["limit"] for rec in shown])
    if len(shown) > 1:
        # Bands of constant binding-line count, split halfway between grid points.
        edges = np.concatenate([[L[0] + 0.5 * (L[0] - L[1])], 0.5 * (L[1:] + L[:-1]), [L[-1] - 0.5 * (L[-2] - L[-1])]])
        counts = [rec["n_binding_lines"] for rec in shown]
        start, segment = 0, 0
        for k in range(1, len(shown) + 1):
            if k == len(shown) or counts[k] != counts[start]:
                if segment % 2 == 0:
                    ax.axvspan(edges[k], edges[start], color=band, zorder=0)
                ax.text(0.5 * (edges[start] + edges[k]), 1.0, str(counts[start]), transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", color=muted, fontsize=9)
                start, segment = k, segment + 1
    ax.plot(L, [rec["iterations"] for rec in shown], "-", color=total_color, linewidth=2, label="total iterations")
    ax.plot(L, [rec["identified_at"] for rec in shown], "-", color=ident_color, linewidth=2,
            label="active set last changes")
    for rec in shown:
        dots(ax, rec["limit"], rec["iterations"], total_color, rec["converged"])
        dots(ax, rec["limit"], rec["identified_at"], ident_color, True)
    style(ax, "uniform line limit L (p.u.); numbers above: binding lines")
    ax.invert_xaxis()   # tighter limits to the right, matching the other panels
    hidden = [rec for rec in records if rec["limit"] > 3.0]
    note = "; ".join(f"L = {rec['limit']:.2f} (default): {rec['iterations']} its" for rec in hidden)
    ax.legend(frameon=False, labelcolor=muted, loc="upper left", title=note or None, title_fontsize=9)

    if not all(rec["converged"] for rec in records):
        axes[0].plot([], [], "o", ms=8, color=total_color, markerfacecolor="none", markeredgewidth=2,
                     label="hit the iteration cap")
        axes[0].legend(frameon=False, labelcolor=muted)
    fig.suptitle(title, color=ink, y=1.02)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-buses", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limits", type=float, nargs="+", default=None,
                        help="uniform line limits to sweep (p.u.); default: the config's own limit, "
                             "then 2.80 down to 1.30 in steps of 0.05, then 1.28")
    parser.add_argument("--max-iterations", type=int, default=5000)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--replot", metavar="JSON", default=None,
                        help="redraw the figure of a saved sweep (written next to it) instead of solving")
    args = parser.parse_args()

    if args.replot:
        with open(args.replot) as f:
            saved = json.load(f)
        fig = plot_sweep(saved["records"], f"admm_parallel iterations vs binding line limits "
                                           f"(n={saved['n_buses']}, seed={saved['seed']}, {saved['n_lines']} lines)")
        fig.savefig(os.path.splitext(args.replot)[0] + ".png", dpi=150, bbox_inches="tight")
        return

    config = Config(n_buses=args.n_buses, seed=args.seed)
    default_limit = float(config.line_flow_limits.flat[0])
    limits = args.limits or [default_limit, *np.round(np.arange(2.80, 1.299, -0.05), 2), 1.28]

    records = []
    for k, limit in enumerate(limits):
        rec = run_point(config, limit, args.max_iterations)
        records.append(rec)
        logger.info(
            f"[{k + 1}/{len(limits)}] L={limit:.3f}: {rec['n_binding_lines']} binding lines, "
            f"{rec['n_binding_pairs']} pairs -> {rec['iterations']} its "
            f"({'converged' if rec['converged'] else 'CAP'}; active set fixed from it {rec['identified_at']} "
            f"after {rec['active_set_changes']} changes, tail rate {rec['tail_rate']:.4f}, "
            f"{rec['active_set_mismatch']} pairs off centralized), obj gap {rec['obj_rel_gap']:.2e}, "
            f"gen dispatch err {rec['gen_dispatch_err']:.2e}, {rec['runtime']:.1f} s"
        )

    os.makedirs(args.results_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"n{args.n_buses}_seed{args.seed}"
    base = os.path.join(args.results_dir, f"line_limit_sweep_{tag}_{stamp}")
    with open(base + ".json", "w") as f:
        json.dump({"n_buses": args.n_buses, "seed": args.seed, "max_iterations": args.max_iterations,
                   "n_lines": len(config.lines), "records": records}, f)
    fig = plot_sweep(records, f"admm_parallel iterations vs binding line limits "
                              f"(n={args.n_buses}, seed={args.seed}, {len(config.lines)} lines)")
    fig.savefig(base + ".png", dpi=150, bbox_inches="tight")
    logger.info(f"wrote {base}.json and {base}.png")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
    # admm_vanilla logs one INFO line per solve; the sweep's own line carries the same numbers.
    logging.getLogger("algorithms").setLevel(logging.WARNING)
    main()
