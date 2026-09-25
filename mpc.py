"""
Receding-horizon (MPC) closed loop of the dynamic OPF.

At every step k the controller solves the dynamic OPF over the window
[k, k + H) from the plant's measured state (Config.window), applies the first
step of the plan to the plant (plant.Plant) and moves on. Any solver with the
signature solve(config) -> SolveResult (algorithms.ALGORITHMS) can be the
controller.

Index convention, as in the solvers: step k's control is applied with the
demand and ambient temperature at k, and the state it produces is stored at
index k and checked against the comfort band at k.
"""
import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from Config import Config
from Trajectory import Trajectory
from algorithms.base import SolveResult
from algorithms.common import to_centralized_layout
from plant import Plant

logger = logging.getLogger(__name__)


@dataclass
class ClosedLoopResult:
    """
    One closed-loop run. Every per-step array has one entry per completed
    step; a run that hit an infeasible window stops there (failed_step,
    failure) and keeps what it had.

    trajectory        realized p, theta (n_buses wide), soc (per battery) and
                      temp (per thermal unit, degrees C): the layout of
                      centralized.solve
    p_cmd             commanded injections, the first step of each plan
    stage_cost        generator cost of the realized dispatch at each step
    plan_obj          objective of each window's plan (the whole window)
    solve_runtime     solver runtime per step, seconds
    imbalance         injection mismatch the plant's generators absorbed
    prediction_error  per key ("p", "soc", "temp"): largest |plan's first
                      step - realized| at each step, i.e. model mismatch
    violations        per constraint: bound excess at each step (0 = satisfied),
                      see violations_by_step
    plans             every window's plan, in the centralized layout; plans[k]
                      starts at step k
    """
    config: Config
    algorithm: str
    horizon: Optional[int]
    trajectory: Trajectory
    p_cmd: np.ndarray
    stage_cost: np.ndarray
    plan_obj: np.ndarray
    solve_runtime: np.ndarray
    imbalance: np.ndarray
    prediction_error: Dict[str, np.ndarray]
    violations: Dict[str, np.ndarray]
    plans: List[Trajectory] = field(repr=False)
    failed_step: Optional[int] = None
    failure: Optional[str] = None

    @property
    def n_steps(self) -> int:
        return len(self.stage_cost)

    @property
    def cost(self) -> float:
        return float(self.stage_cost.sum())

    def max_violations(self) -> Dict[str, float]:
        return {k: float(v.max()) if v.size else 0.0 for k, v in self.violations.items()}


def _excess_by_step(values: np.ndarray, lower, upper) -> np.ndarray:
    """Per row, the largest amount by which values leave [lower, upper]; 0 when inside."""
    if values.shape[1] == 0:
        return np.zeros(values.shape[0])
    return np.maximum(0.0, np.maximum(lower - values, values - upper).max(axis=1))


def violations_by_step(traj: Trajectory, config: Config) -> Dict[str, np.ndarray]:
    """
    Bound excess of a realized trajectory (centralized layout, starting at step
    0 of config) at every step. Only bounds are checked: in a realized
    trajectory the dynamics hold by construction, since the plant produced it.
    The first step has no ramp, as the day starts unanchored.
    """
    n = traj.horizon
    gen = slice(0, config.n_gens)
    thermal = slice(config.n_gens + config.n_loads, config.n_gens + config.n_loads + config.n_thermal)
    battery = slice(config.n_gens + config.n_loads + config.n_thermal, config.n_buses)
    p, theta = traj["p"], traj["theta"]
    p_gen = p[:, gen]

    ramp = np.zeros(n)
    if n > 1:
        ramp[1:] = _excess_by_step(np.diff(p_gen, axis=0), config.gen_R_min, config.gen_R_max)

    lines = np.array(config.lines).reshape(-1, 2)
    flows = -config.B[lines[:, 0], lines[:, 1]] * (theta[:, lines[:, 0]] - theta[:, lines[:, 1]])
    limits = config.line_flow_limits[lines[:, 0], lines[:, 1]]

    return {
        "gen_capacity": _excess_by_step(p_gen, config.gen_P_min, config.gen_P_max),
        "gen_ramp": ramp,
        "thermal_band": _excess_by_step(traj["temp"], config.thermal_T_min_ext[:, :n].T,
                                        config.thermal_T_max_ext[:, :n].T),
        "thermal_box": _excess_by_step(-p[:, thermal], config.thermal_p_min, config.thermal_p_max),
        "soc_bounds": _excess_by_step(traj["soc"], config.battery_q_min, config.battery_q_max),
        "battery_box": _excess_by_step(p[:, battery], config.battery_p_min, config.battery_p_max),
        "line_limit": _excess_by_step(flows, -limits, limits),
    }


def run_closed_loop(config: Config, solve: Callable[[Config], SolveResult], horizon: Optional[int],
                    plant: Optional[Plant] = None, n_steps: Optional[int] = None,
                    algorithm: str = "") -> ClosedLoopResult:
    """
    Runs solve as a receding-horizon controller on plant for n_steps steps
    (default config.N, one day).

    horizon is the window length in steps; the data must cover the last window,
    so config needs T_lookahead >= (horizon - 1) * dt. horizon=None shrinks the
    window to end at config.N (midnight) instead. With a plant equal to the
    model, the closed loop then reproduces the day-ahead optimum, which makes
    it a check of the loop itself.

    plant defaults to Plant(config), the nominal plant. An infeasible window
    (solve raises RuntimeError) ends the run; the result keeps the steps before
    it.
    """
    n_steps = config.N if n_steps is None else n_steps
    plant = plant if plant is not None else Plant(config)
    if plant.k != 0:
        raise ValueError(f"plant is at step {plant.k}; the closed loop starts from step 0")

    realized: List[dict] = []
    p_cmd, plan_obj, runtimes, plans = [], [], [], []
    prediction_error: Dict[str, list] = {"p": [], "soc": [], "temp": []}
    failed_step, failure = None, None

    for k in range(n_steps):
        length = config.N - k if horizon is None else horizon
        window = config.window(k, length, plant.soc, plant.temp, plant.p_gen_prev)
        try:
            sol = solve(window)
        except RuntimeError as exc:
            failed_step, failure = k, str(exc)
            logger.warning(f"{algorithm or 'MPC'}: window at step {k} failed, stopping the run: {exc}")
            break

        plan = to_centralized_layout(window, sol.trajectory)
        step = plant.step(plan["p"][0])

        realized.append(step)
        p_cmd.append(plan["p"][0].copy())
        plan_obj.append(sol.obj)
        runtimes.append(sol.runtime)
        plans.append(plan)
        for key in prediction_error:
            diff = plan[key][0] - step[key]
            prediction_error[key].append(float(np.abs(diff).max()) if diff.size else 0.0)

        logger.info(f"{algorithm or 'MPC'} step {k + 1}/{n_steps}: window of {length}, "
                    f"plan objective {sol.obj:.3f}, solve {sol.runtime:.3f}s, imbalance {step['imbalance']:.2e}")

    def stacked(key: str, width: int) -> np.ndarray:
        return np.array([r[key] for r in realized]).reshape(len(realized), width)

    trajectory = Trajectory({
        "p": stacked("p", config.n_buses),
        "theta": stacked("theta", config.n_buses),
        "soc": stacked("soc", config.n_battery),
        "temp": stacked("temp", config.n_thermal),
    })
    p_gen = trajectory["p"][:, :config.n_gens]
    stage_cost = (config.gen_cost_alpha * p_gen ** 2 + config.gen_cost_beta * p_gen).sum(axis=1)

    return ClosedLoopResult(
        config=config,
        algorithm=algorithm,
        horizon=horizon,
        trajectory=trajectory,
        p_cmd=np.array(p_cmd).reshape(len(realized), config.n_buses),
        stage_cost=stage_cost,
        plan_obj=np.array(plan_obj),
        solve_runtime=np.array(runtimes),
        imbalance=np.array([r["imbalance"] for r in realized]),
        prediction_error={k: np.array(v) for k, v in prediction_error.items()},
        violations=violations_by_step(trajectory, config),
        plans=plans,
        failed_step=failed_step,
        failure=failure,
    )
