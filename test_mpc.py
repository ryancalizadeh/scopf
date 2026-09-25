"""
Checks the receding-horizon closed loop (mpc.py, plant.py) and the pieces it
adds to Config and the solvers.

    python test_mpc.py            # n = 8, plus the n = 15 Bellman check
    python test_mpc.py --fast     # n = 8 only

1. Lookahead data: T_lookahead leaves a seed's day data unchanged, the *_ext
   arrays extend it, and the comfort band repeats on day 2.
2. Windows: a window at step 0 spanning the day is the day problem, and the
   ramp anchor binds in both the centralized solver and the ADMM Generator prox.
3. Plant alone: the day-ahead controls, applied open loop, reproduce the
   day-ahead states and angles.
4. Bellman check: with a nominal plant and the window shrinking to midnight,
   the closed loop must reproduce the day-ahead optimum (the tail of an optimal
   plan is optimal for the tail problem, and the dispatch is unique because
   the cost is strictly convex in p_gen). Any off-by-one in the state feedback
   or the ramp anchor breaks this.
5. Rolling 24 h window, nominal plant: every window solves, no bound is
   violated and the plant does exactly what each plan predicted.
"""
import sys

import numpy as np

from Config import Config
from algorithms import centralized
from algorithms.common import Generator
from mpc import run_closed_loop
from plant import Plant

TOL = 1e-6


def test_lookahead(n_buses: int) -> None:
    day = Config(n_buses)
    ext = Config(n_buses, T_lookahead=24)
    N = day.N
    for key in ("load_P", "thermal_T_amb", "thermal_T_min", "thermal_T_max",
                "gen_P_max", "gen_cost_alpha", "battery_q0", "thermal_eta", "B"):
        assert np.array_equal(getattr(day, key), getattr(ext, key)), f"T_lookahead changed {key}"
    for key in ("load_P", "thermal_T_amb", "thermal_T_min", "thermal_T_max"):
        full = getattr(ext, key + "_ext")
        assert full.shape[1] == N + 96, f"{key}_ext has {full.shape[1]} steps"
        assert np.array_equal(full[:, :N], getattr(day, key)), f"{key}_ext does not start with {key}"
        assert np.array_equal(getattr(day, key + "_ext"), getattr(day, key)), f"{key}_ext without lookahead"
    assert np.array_equal(ext.thermal_T_min_ext[:, N:], ext.thermal_T_min), "comfort band does not repeat"
    assert np.array_equal(ext.thermal_T_max_ext[:, N:], ext.thermal_T_max), "comfort band does not repeat"
    print(f"  n={n_buses}: day data unchanged by T_lookahead; lookahead extends it")


def test_windows(config: Config) -> None:
    N = config.N
    ref = centralized.solve(config)
    win = centralized.solve(config.window(0, N, config.battery_q0, config.thermal_T0, None))
    assert abs(win.obj - ref.obj) <= 1e-9 * abs(ref.obj), f"window objective {win.obj} vs day {ref.obj}"

    # Anchor every generator at full output: the first step may fall by at most R.
    p_prev = config.gen_P_max.copy()
    window = config.window(0, N, config.battery_q0, config.thermal_T0, p_prev)
    anchored = centralized.solve(window)
    first = anchored.trajectory["p"][0, :config.n_gens]
    assert (first - p_prev >= config.gen_R_min - TOL).all(), "centralized ignores the ramp anchor"
    assert (ref.trajectory["p"][0, :config.n_gens] - p_prev < config.gen_R_min).any(), \
        "anchor test is vacuous: the unanchored plan already satisfies it"

    # Same for the ADMM generator prox, from a proximal point far below the anchor.
    z = config.make_base_trajectory().at_bus[0]
    z["p"][:] = 0.0
    out = Generator(window, bus_index=0, gen_index=0).prox(z, rho=2.0)
    assert out["p"][0, 0] - p_prev[0] >= config.gen_R_min[0] - TOL, "Generator prox ignores the ramp anchor"
    print(f"  day window objective matches ({ref.obj:.4f}); ramp anchor binds in centralized and ADMM")


def test_plant_open_loop(config: Config) -> None:
    ref = centralized.solve(config).trajectory
    plant = Plant(config)
    worst = {"p": 0.0, "theta": 0.0, "soc": 0.0, "temp": 0.0}
    worst_imbalance = 0.0
    for k in range(config.N):
        step = plant.step(ref["p"][k])
        for key in worst:
            worst[key] = max(worst[key], float(np.abs(step[key] - ref[key][k]).max()))
        worst_imbalance = max(worst_imbalance, abs(step["imbalance"]))
    assert all(v < TOL for v in worst.values()), f"plant diverges from the solver's model: {worst}"
    assert worst_imbalance < 1e-8, f"imbalance {worst_imbalance:.2e}"
    print(f"  open-loop plant matches the day-ahead plan (worst {max(worst.values()):.1e}, "
          f"imbalance {worst_imbalance:.1e})")


def assert_clean(result, what: str) -> None:
    assert result.failed_step is None, f"{what}: window at step {result.failed_step} failed: {result.failure}"
    assert result.n_steps == result.config.N, f"{what}: ran {result.n_steps} steps"
    bad = {k: v for k, v in result.max_violations().items() if v > TOL}
    assert not bad, f"{what}: violations {bad}"
    mismatch = {k: float(v.max()) for k, v in result.prediction_error.items() if v.size}
    assert all(v < TOL for v in mismatch.values()), f"{what}: nominal plant departs from the plan: {mismatch}"


def test_bellman(config: Config) -> None:
    ref = centralized.solve(config)
    result = run_closed_loop(config, centralized.solve, horizon=None, algorithm="centralized")
    assert_clean(result, "shrinking horizon")
    gap = abs(result.cost - ref.obj) / abs(ref.obj)
    gen_err = float(np.abs(result.trajectory["p"][:, :config.n_gens] - ref.trajectory["p"][:, :config.n_gens]).max())
    assert gap < 1e-8, f"closed-loop cost {result.cost} vs day-ahead {ref.obj} (rel {gap:.1e})"
    # CLARABEL's default tolerances pin the dispatch only to ~1e-4 (the day-ahead
    # solve itself moves 1.3e-4 at n=8 when they are tightened to 1e-10, and the
    # closed-loop error then falls from 5.9e-4 to 9e-6), so this bound is set by
    # solver accuracy. An indexing bug moves the dispatch by far more.
    assert gen_err < 2e-3, f"closed-loop dispatch departs from the day-ahead optimum by {gen_err:.1e}"
    print(f"  n={config.n_buses} shrinking horizon: closed loop = day-ahead (cost rel {gap:.1e}, "
          f"dispatch {gen_err:.1e}), {result.solve_runtime.sum():.1f}s of solves")


def test_rolling(config: Config) -> None:
    ref = centralized.solve(config)
    horizon = int(round(24 / config.dt))
    result = run_closed_loop(config, centralized.solve, horizon=horizon, algorithm="centralized")
    assert_clean(result, "rolling horizon")
    midnight = result.trajectory["soc"][config.N - 1] - config.battery_qT
    print(f"  n={config.n_buses} rolling {horizon}-step window: feasible throughout, no violations; "
          f"cost {result.cost:.4f} vs day-ahead {ref.obj:.4f} ({100 * (result.cost / ref.obj - 1):+.3f}%), "
          f"midnight SOC - qT in [{midnight.min():+.3f}, {midnight.max():+.3f}]")


def main() -> None:
    fast = "--fast" in sys.argv

    print("1. lookahead data")
    for n in (8, 15, 45):
        test_lookahead(n)

    config = Config(8, T_lookahead=24)
    print("2. windows and the ramp anchor")
    test_windows(config)
    print("3. plant alone")
    test_plant_open_loop(config)
    print("4. Bellman check")
    test_bellman(config)
    if not fast:
        test_bellman(Config(15))
    print("5. rolling horizon")
    test_rolling(config)
    print("all MPC checks passed")


if __name__ == "__main__":
    main()
