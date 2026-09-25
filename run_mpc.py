"""
Runs dynamic-OPF solvers as receding-horizon (MPC) controllers in closed loop
with a plant, over one day, and saves the trajectories for inspection.

    python run_mpc.py                                          # centralized, 8 buses, 24 h window
    python run_mpc.py --algorithms centralized admm_vanilla --n-buses 15 --horizon-hours 12
    python run_mpc.py --shrinking                              # window shrinks to midnight

At every 15-minute step each algorithm solves the dynamic OPF over the next
--horizon-hours from the plant's measured state (battery SOC at the end of
each window pinned to battery_qT), applies the first step to the plant, and
repeats. The plant is plant.Plant: the MPC's own linear device models, with
generator participation factors balancing any mismatch.

The day-ahead optimum (centralized.solve over the day) is the reference. With
a nominal plant it is also the perfect-information optimum; a rolling window
need not end the day at battery_qT, so compare the cost gap together with the
midnight SOC.

Writes to --results-dir:
  mpc_raw_<timestamp>.pkl                {"config", "day_ahead", "closed_loop": {algo: ClosedLoopResult}}
  mpc_<tag>_{power,temperature,battery}_<timestamp>.png   closed loop vs day-ahead, with bounds
  mpc_<tag>_diagnostics_<timestamp>.png  cost, solve time, violations, prediction error per step
"""
import argparse
import logging
import os
import pickle
import sys
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from Config import Config
from ExperimentResult import ExperimentResult
from Trajectory import Trajectory
from algorithms import ALGORITHMS, centralized
from algorithms.centralized import check_feasible
from mpc import ClosedLoopResult, run_closed_loop
from visualize_mpc import DAY_AHEAD_LABEL, closed_loop_label, plot_closed_loop_diagnostics
from visualize_trajectory import save_trajectory_comparison

logger = logging.getLogger(__name__)


def _padded(traj: Trajectory, n: int) -> Trajectory:
    """traj extended with NaN rows to n steps, so a run that stopped early still plots."""
    missing = n - traj.horizon
    if missing <= 0:
        return traj
    return Trajectory({k: np.vstack([v, np.full((missing, v.shape[1]), np.nan)]) for k, v in traj.items()})


def summarize(result: ClosedLoopResult, day_ahead_obj: float) -> str:
    config = result.config
    horizon = "shrinking to midnight" if result.horizon is None else \
        f"{result.horizon} steps ({result.horizon * config.dt:g} h)"
    lines = [f"{result.algorithm} closed loop, window {horizon}: {result.n_steps}/{config.N} steps"]
    if result.failed_step is not None:
        lines.append(f"  STOPPED at step {result.failed_step} "
                     f"(hour {result.failed_step * config.dt:g}): {result.failure}")
    lines.append(f"  cost              {result.cost:.4f}   day-ahead {day_ahead_obj:.4f}   "
                 f"gap {100 * (result.cost / day_ahead_obj - 1):+.3f}%"
                 + ("" if result.n_steps == config.N else "   (partial day)"))
    if result.n_steps >= config.N and config.n_battery:
        dev = result.trajectory["soc"][config.N - 1] - config.battery_qT
        lines.append(f"  SOC at midnight   minus qT in [{dev.min():+.4f}, {dev.max():+.4f}]")
    worst = result.max_violations()
    lines.append("  worst violation   " + "  ".join(f"{k} {v:.1e}" for k, v in worst.items()))
    if result.n_steps:
        lines.append("  prediction error  " + "  ".join(
            f"{k} {v.max():.1e}" for k, v in result.prediction_error.items() if v.size))
        lines.append(f"  |imbalance|       max {np.abs(result.imbalance).max():.1e}")
        rt = result.solve_runtime
        lines.append(f"  solve time        total {rt.sum():.1f}s   mean {rt.mean():.3f}s   max {rt.max():.3f}s")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--algorithms", nargs="+", default=["centralized"], choices=sorted(ALGORITHMS),
                        help="Controllers to run in closed loop (default: centralized).")
    parser.add_argument("--n-buses", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--avg-degree", type=float, default=2.2)
    parser.add_argument("--horizon-hours", type=float, default=24.0,
                        help="Length of the rolling MPC window in hours (default 24).")
    parser.add_argument("--shrinking", action="store_true",
                        help="Shrink the window to end at midnight instead (reproduces the day-ahead "
                             "optimum on a nominal plant; a check of the loop).")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler("app.log"), logging.StreamHandler(sys.stdout)],
    )

    config = Config(n_buses=args.n_buses, avg_degree=args.avg_degree, seed=args.seed,
                    T_lookahead=0.0 if args.shrinking else args.horizon_hours)
    horizon = None if args.shrinking else int(round(args.horizon_hours / config.dt))
    if horizon is not None and horizon < 1:
        sys.exit(f"--horizon-hours must cover at least one step of {config.dt} h")
    logger.info(f"{config.n_buses} buses (seed {config.seed}): {config.n_gens} gen, {config.n_loads} load, "
                f"{config.n_thermal} thermal, {config.n_battery} battery; window "
                f"{'shrinking' if horizon is None else f'{horizon} steps'}")

    if not check_feasible(config):
        sys.exit(f"The day-ahead problem is infeasible for n_buses={config.n_buses}, seed={config.seed}.")
    day_ahead = centralized.solve(config)

    closed_loop = {}
    for name in args.algorithms:
        logger.info(f"=== {name} in closed loop ===")
        closed_loop[name] = run_closed_loop(config, ALGORITHMS[name], horizon, algorithm=name)

    summary = "\n\n".join(summarize(res, day_ahead.obj) for res in closed_loop.values())
    logger.info("closed-loop summary\n\n" + summary + "\n")

    os.makedirs(args.results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{config.n_buses}bus_seed{config.seed}_" + ("shrinking" if horizon is None else f"H{horizon}")

    pickle_path = os.path.join(args.results_dir, f"mpc_raw_{timestamp}.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump({"config": config, "day_ahead": day_ahead, "closed_loop": closed_loop}, f)
    print(f"wrote {pickle_path}")

    comparison = {DAY_AHEAD_LABEL: [ExperimentResult(
        config=config, algorithm=DAY_AHEAD_LABEL, trajectory=day_ahead.trajectory, obj=day_ahead.obj,
        runtime=day_ahead.runtime, p_residual=None, s_residual=None, convergence=None)]}
    for name, res in closed_loop.items():
        label = closed_loop_label(name)
        comparison[label] = [ExperimentResult(
            config=config, algorithm=label, trajectory=_padded(res.trajectory, config.N), obj=res.cost,
            runtime=float(res.solve_runtime.sum()), p_residual=None, s_residual=None, convergence=None)]
    for path in save_trajectory_comparison(comparison, results_dir=args.results_dir, run_name=f"mpc_{tag}"):
        print(f"wrote {path}")

    fig = plot_closed_loop_diagnostics(closed_loop, day_ahead)
    path = os.path.join(args.results_dir, f"mpc_{tag}_diagnostics_{timestamp}.png")
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
