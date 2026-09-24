"""
Where does parallel-ADMM runtime go: the f prox (Network) or the g prox (buses)?

The two proxes are the whole body of the ADMM iteration (admm.admm), so wrapping
each in a timer accounts for essentially all of the solve. f is one dense
Cholesky solve over all N timesteps plus a QP for each timestep whose line
limits the analytic projection misses; g is n_buses independent device
projections, fanned out over the executor admm_parallel.executor_for picks.

The interesting quantity is the *share*, not the absolute time. Reported per
iteration as well as per solve, because the iteration count itself moves with n
and would otherwise confound the comparison.

What the sweep actually shows is a regime switch rather than a smooth trend, and
it is driven by the QP fallback. While the analytic projection satisfies the line
limits, f is one Cholesky solve over N right-hand sides: sub-millisecond, ~1% of
runtime, and g (n_buses device QPs) dominates completely. Once the network runs
close enough to its line limits that the analytic projection misses them, f
re-solves those timesteps one at a time with cvxpy and its cost jumps by two
orders of magnitude to ~(QP/iter) * (ms per QP), overtaking g. The switch is a
property of congestion, not of n directly -- bigger networks in this generator
sit closer to their limits, so they cross over -- which is why n_qp_solves is
reported alongside the timings and why two seeds at one size can land on
opposite sides.
"""
import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict

import numpy as np

from Config import Config
from Proxable import Proxable
from Trajectory import Trajectory
from admm import admm
from algorithms.admm_parallel import executor_for
from algorithms.common import Network, make_bus_behaviours, rho_fixed, check_solution

logger = logging.getLogger(__name__)


class TimedProx(Proxable):
    """
    Wraps a Proxable and records the wall-clock duration of every prox call.
    perf_counter around the call is the honest measure for both sides: f is
    numpy/cvxpy in this process, and g's cost as seen by the algorithm includes
    the scatter/gather and (for the process executor) the pickling, which is
    real runtime the caller cannot overlap with anything else.
    """

    def __init__(self, inner: Proxable, name: str):
        self.inner = inner
        self.name = name
        self.times: list[float] = []

    def prox(self, z: Trajectory, rho: float = 1.0) -> Trajectory:
        start = time.perf_counter()
        try:
            return self.inner.prox(z, rho)
        finally:
            self.times.append(time.perf_counter() - start)

    def close(self) -> None:
        close = getattr(self.inner, "close", None)
        if close is not None:
            close()


@dataclass
class ProfileRun:
    n_buses: int
    n_timesteps: int
    seed: int
    executor: str
    n_workers: int
    iterations: int
    converged: bool
    runtime: float
    f_total: float
    g_total: float
    other_total: float
    objective: float
    p_residual: float
    network_residual: float
    n_qp_solves: int
    f_times: list[float] = field(default_factory=list)
    g_times: list[float] = field(default_factory=list)

    @property
    def f_share(self) -> float:
        return self.f_total / self.runtime

    @property
    def g_share(self) -> float:
        return self.g_total / self.runtime

    @property
    def f_per_iter(self) -> float:
        return self.f_total / self.iterations

    @property
    def g_per_iter(self) -> float:
        return self.g_total / self.iterations

    # The first call to each prox pays one-off costs that do not recur: cvxpy
    # canonicalises every device problem, and the process executor's first
    # round-trip follows the worker spawn. At n=500 that is seconds against a
    # ~0.5 s steady-state call, so a short run (--max-iterations 10) would be
    # mostly warmup unless it is excluded. Over a converged run the difference
    # is negligible, so these are the figures to compare across run lengths.
    @property
    def f_per_iter_steady(self) -> float:
        return _mean(self.f_times[1:]) if len(self.f_times) > 1 else self.f_per_iter

    @property
    def g_per_iter_steady(self) -> float:
        return _mean(self.g_times[1:]) if len(self.g_times) > 1 else self.g_per_iter

    @property
    def f_share_steady(self) -> float:
        total = self.f_per_iter_steady + self.g_per_iter_steady
        return self.f_per_iter_steady / total if total else float("nan")


def profile_one(config: Config, max_iterations: int = 1000,
                n_workers: "int | None" = None,
                executor: "str | None" = None) -> ProfileRun:
    """
    One instrumented solve, mirroring admm_vanilla.solve (same rho schedule,
    same threshold, same executor choice) with f and g wrapped in timers.
    executor overrides the choice admm_parallel would make, so the same size
    can be measured on both executors.
    """
    executor = executor or executor_for(config)
    f = TimedProx(Network(config), "f")
    g = TimedProx(make_bus_behaviours(config, parallel=executor, n_workers=n_workers), "g")

    z0 = config.make_base_trajectory()
    threshold = 1e-3 * np.sqrt(config.n_buses / 8)

    start = time.perf_counter()
    try:
        _, zs, _, rs, ss, _ = admm(
            f, g, z0, rho=rho_fixed(2.0), threshold=threshold,
            max_iterations=max_iterations,
        )
    finally:
        g.close()
    runtime = time.perf_counter() - start

    checks = check_solution(zs[-1], config)
    iterations = len(rs)
    f_total, g_total = sum(f.times), sum(g.times)
    inner = getattr(g.inner, "n_workers", None) or getattr(g.inner, "max_workers", 1)

    return ProfileRun(
        n_buses=config.n_buses,
        n_timesteps=config.N,
        seed=config.seed,
        executor=executor,
        n_workers=inner,
        iterations=iterations,
        converged=bool(rs[-1] < threshold and ss[-1] < threshold),
        runtime=runtime,
        f_total=f_total,
        g_total=g_total,
        other_total=runtime - f_total - g_total,
        objective=checks["objective"],
        p_residual=rs[-1],
        network_residual=checks["network_residual"],
        n_qp_solves=f.inner.n_qp_solves,
        f_times=f.times,
        g_times=g.times,
    )


def sweep(n_buses_list: "list[int]", seeds: "list[int]", max_iterations: int = 1000,
          n_workers: "int | None" = None, executor: "str | None" = None,
          **config_kwargs) -> "list[ProfileRun]":
    runs: "list[ProfileRun]" = []
    for n_buses in n_buses_list:
        for seed in seeds:
            config = Config(seed=seed, n_buses=n_buses, **config_kwargs)
            logger.info(f"profiling n_buses={n_buses} seed={seed} ...")
            run = profile_one(config, max_iterations=max_iterations,
                              n_workers=n_workers, executor=executor)
            logger.info(
                f"  {run.iterations} iters in {run.runtime:.2f}s: "
                f"f={run.f_share:.1%} g={run.g_share:.1%} "
                f"(f {run.f_per_iter_steady * 1e3:.2f} ms/iter, g {run.g_per_iter_steady * 1e3:.2f} ms/iter)"
                + ("" if run.converged else
                   "  [truncated]" if run.iterations >= max_iterations - 1 else "  [NOT CONVERGED]")
            )
            runs.append(run)
    return runs


def _mean(values):
    return float(np.mean(values)) if len(values) else float("nan")


def summarize(runs: "list[ProfileRun]") -> "list[dict]":
    """Mean over seeds at each size, in the order the sizes were first seen."""
    by_size: "dict[int, list[ProfileRun]]" = {}
    for run in runs:
        by_size.setdefault(run.n_buses, []).append(run)

    rows = []
    for n_buses, group in by_size.items():
        rows.append({
            "n_buses": n_buses,
            "n_runs": len(group),
            "executor": group[0].executor,
            "n_workers": group[0].n_workers,
            "iterations": _mean([r.iterations for r in group]),
            "converged": sum(r.converged for r in group),
            "runtime": _mean([r.runtime for r in group]),
            "f_total": _mean([r.f_total for r in group]),
            "g_total": _mean([r.g_total for r in group]),
            "f_share": _mean([r.f_share for r in group]),
            "g_share": _mean([r.g_share for r in group]),
            "other_share": _mean([r.other_total / r.runtime for r in group]),
            "f_per_iter": _mean([r.f_per_iter for r in group]),
            "g_per_iter": _mean([r.g_per_iter for r in group]),
            "f_per_iter_steady": _mean([r.f_per_iter_steady for r in group]),
            "g_per_iter_steady": _mean([r.g_per_iter_steady for r in group]),
            "f_share_steady": _mean([r.f_share_steady for r in group]),
            "f_over_g": _mean([r.f_total / r.g_total for r in group]),
            "n_qp_solves": _mean([r.n_qp_solves for r in group]),
            # QPs per iteration against N is the diagnostic: 0 means the analytic
            # projection sufficed throughout, ~N means nearly every timestep had
            # to be re-solved and f has entered its expensive regime.
            "qp_per_iter": _mean([r.n_qp_solves / r.iterations for r in group]),
            "qp_saturation": _mean([r.n_qp_solves / r.iterations / r.n_timesteps for r in group]),
        })
    return rows


def format_table(rows: "list[dict]") -> str:
    header = (
        f"{'n':>5} {'exec':>9} {'iters':>7} {'runtime':>9} "
        f"{'f ms/it':>9} {'g ms/it':>9} {'f shr*':>7} {'f share':>8} {'g share':>8} {'other':>7} "
        f"{'f/g':>7} {'QP/it':>7} {'QP sat':>7}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        lines.append(
            f"{row['n_buses']:>5} {row['executor']:>9} {row['iterations']:>7.0f} "
            f"{row['runtime']:>8.2f}s {row['f_per_iter_steady'] * 1e3:>9.2f} "
            f"{row['g_per_iter_steady'] * 1e3:>9.2f} {row['f_share_steady']:>6.1%} "
            f"{row['f_share']:>7.1%} {row['g_share']:>7.1%} {row['other_share']:>6.1%} "
            f"{row['f_over_g']:>7.2f} {row['qp_per_iter']:>7.1f} {row['qp_saturation']:>6.0%}"
        )
    lines.append("")
    lines.append("ms/it and 'f shr*' exclude the first (warmup) call to each prox; "
                 "'f share'/'g share' are of total solve wall time.")
    return "\n".join(lines)


def plot(rows: "list[dict]", path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = [r["n_buses"] for r in rows]
    fig, (ax_share, ax_time, ax_qp) = plt.subplots(1, 3, figsize=(17, 4.5))

    # Left: the shares, the claim under test.
    # Steady-state shares, so a truncated run (--max-iterations 10) is not
    # dominated by the one-off canonicalisation/spawn in the first call.
    ax_share.stackplot(
        n,
        [r["f_share_steady"] * 100 for r in rows],
        [(1.0 - r["f_share_steady"]) * 100 for r in rows],
        labels=["f prox (network)", "g prox (buses)"],
        colors=["#4c72b0", "#dd8452"],
        alpha=0.9,
    )
    ax_share.set_xscale("log")
    ax_share.set_xticks(n)
    ax_share.set_xticklabels([str(v) for v in n])
    ax_share.set_xlim(min(n), max(n))
    ax_share.set_ylim(0, 100)
    ax_share.set_xlabel("n_buses")
    ax_share.set_ylabel("share of per-iteration prox time (%)")
    ax_share.set_title("Runtime split between the two proxes")
    ax_share.legend(loc="center left")

    # Right: absolute cost per iteration, which is what actually drives the split.
    ax_time.plot(n, [r["f_per_iter_steady"] * 1e3 for r in rows], "o-", color="#4c72b0", label="f prox (network)")
    ax_time.plot(n, [r["g_per_iter_steady"] * 1e3 for r in rows], "s-", color="#dd8452", label="g prox (buses)")
    ax_time.set_xscale("log")
    ax_time.set_yscale("log")
    ax_time.set_xticks(n)
    ax_time.set_xticklabels([str(v) for v in n])
    ax_time.set_xlabel("n_buses")
    ax_time.set_ylabel("time per ADMM iteration (ms)")
    ax_time.set_title("Per-iteration cost of each prox")
    ax_time.grid(True, which="both", alpha=0.3)
    ax_time.legend()

    # Right: the explanatory variable. f's jump tracks the fraction of timesteps
    # the analytic projection fails to keep inside the line limits.
    ax_qp.plot(n, [r["qp_saturation"] * 100 for r in rows], "D-", color="#55a868")
    ax_qp.set_xscale("log")
    ax_qp.set_xticks(n)
    ax_qp.set_xticklabels([str(v) for v in n])
    ax_qp.set_ylim(0, 100)
    ax_qp.set_xlabel("n_buses")
    ax_qp.set_ylabel("timesteps needing the line-limit QP (%)")
    ax_qp.set_title("Why f jumps: the QP fallback")
    ax_qp.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    logger.info(f"wrote {path}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-buses", type=int, nargs="+",
                        default=[8, 16, 32, 64, 128, 256])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--n-workers", type=int, default=None)
    parser.add_argument("--executor", choices=["sequential", "threads", "processes"], default=None,
                        help="override the executor admm_parallel would pick for the size")
    parser.add_argument("--avg-degree", type=float, default=2.3)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])

    runs = sweep(args.n_buses, args.seeds, max_iterations=args.max_iterations,
                 n_workers=args.n_workers, executor=args.executor,
                 avg_degree=args.avg_degree)
    rows = summarize(runs)

    print()
    print(format_table(rows))
    print()

    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.tag or time.strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(args.out_dir, f"prox_profile_{tag}.json")
    with open(json_path, "w") as fh:
        json.dump({"summary": rows, "runs": [asdict(r) for r in runs]}, fh, indent=2)
    logger.info(f"wrote {json_path}")

    if not args.no_plot:
        plot(rows, os.path.join(args.out_dir, f"prox_profile_{tag}.png"))


if __name__ == "__main__":
    main()
