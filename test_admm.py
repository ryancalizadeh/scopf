"""
Checks the ADMM proxes and the ADMM solver against the centralized solver.

    python test_admm.py            # unit tests + n=5 and n=8 integration
    python test_admm.py --slow     # also n=45, the only size with active line limits

Unit tests compare each prox against a direct cvxpy solve of the same problem
and check the properties a projection must have (idempotence, rho-invariance,
foreign columns untouched). Integration runs admm_vanilla.solve and compares
the dispatch, objective and states to algorithms.centralized.solve, then
applies test_centralized.check_solution to the ADMM result.
"""
import sys
import time

import numpy as np
import cvxpy as cp

from Config import Config
from Trajectory import Trajectory
from algorithms import centralized, admm_vanilla
from algorithms.common import (Network, Generator, Load, Battery, Thermal,
                               make_bus_behaviour, check_solution, thermal_shift)
from test_centralized import check_solution as check_centralized


def random_trajectory(config: Config, rng: np.random.Generator, scale: float = 1.0) -> Trajectory:
    N, n = config.N, config.n_buses
    return Trajectory({
        "p": scale * rng.normal(size=(N, n)),
        "theta": scale * 0.3 * rng.normal(size=(N, n)),
        "soc": rng.uniform(0, 1, size=(N, n)),
        "temp": rng.normal(size=(N, n)),
    })


def network_reference(config: Config, z: Trajectory) -> Trajectory:
    """
    Joint-space projection onto DC flow + line limits, one cvxpy problem per
    timestep (the timesteps are independent, and small problems solve far more
    accurately than one whole-horizon problem).
    """
    n = config.n_buses
    B_dc = -config.B
    P = cp.Variable(n)
    theta = cp.Variable(n)
    p_z = cp.Parameter(n)
    theta_z = cp.Parameter(n)
    constraints = [P == B_dc @ theta]
    for i, j in config.lines:
        flow = -config.B[i, j] * (theta[i] - theta[j])
        constraints.append(cp.abs(flow) <= config.line_flow_limits[i, j])
    prob = cp.Problem(cp.Minimize(cp.sum_squares(P - p_z) + cp.sum_squares(theta - theta_z)), constraints)
    ret = z.copy()
    for t in range(config.N):
        p_z.value, theta_z.value = z["p"][t], z["theta"][t]
        prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)
        ret["p"][t] = np.asarray(P.value)
        ret["theta"][t] = np.asarray(theta.value)
    return ret


def generator_reference(config: Config, k: int, z_p: np.ndarray, rho: float) -> np.ndarray:
    """Direct (non-DPP) solve of the generator prox."""
    p = cp.Variable(config.N)
    cost = config.gen_cost_alpha[k] * cp.sum_squares(p) + config.gen_cost_beta[k] * cp.sum(p)
    prob = cp.Problem(cp.Minimize(cost + rho / 2 * cp.sum_squares(p - z_p)), [
        p >= config.gen_P_min[k], p <= config.gen_P_max[k],
        cp.diff(p) >= config.gen_R_min[k], cp.diff(p) <= config.gen_R_max[k],
    ])
    prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)
    return np.asarray(p.value)


def assert_foreign_columns_untouched(z: Trajectory, out: Trajectory, owned: set) -> None:
    for key in z.keys():
        if key not in owned:
            assert np.array_equal(z[key], out[key]), f"prox modified foreign key {key!r}"


def test_network(config: Config, n_draws: int = 3) -> None:
    net = Network(config)
    rng = np.random.default_rng(config.n_buses)
    for draw in range(n_draws):
        # Large draws force line-limit violations so the QP branch is exercised.
        scale = 1.0 if draw == 0 else 10.0 * draw
        z = random_trajectory(config, rng, scale)
        before = net.n_qp_solves
        x = net.prox(z, 1.0)
        n_qp = net.n_qp_solves - before
        assert np.abs(x["p"] - x["theta"] @ net.B_dc).max() < 1e-9, "p != B_dc theta"
        assert_foreign_columns_untouched(z, x, {"p", "theta"})
        assert (x - net.prox(x, 1.0)).norm() < 1e-8, "network projection not idempotent"
        assert (x - net.prox(z, 100.0)).norm() < 1e-12, "network projection depends on rho"
        ref = network_reference(config, z)
        err = max(np.abs(x["p"] - ref["p"]).max(), np.abs(x["theta"] - ref["theta"]).max())
        assert err < 1e-6, f"network projection disagrees with cvxpy reference by {err:.2e}"
        checks = check_solution(x, config)
        assert checks["line_limit"] < 1e-7, f"line limit violated by {checks['line_limit']:.2e}"
        print(f"  n={config.n_buses} draw {draw} (scale {scale:g}): ref err {err:.1e}, "
              f"qp timesteps {n_qp}/{config.N}")


def test_devices(config: Config) -> None:
    rng = np.random.default_rng(100 + config.n_buses)
    z = random_trajectory(config, rng)
    n_gens, n_loads, n_thermal = config.n_gens, config.n_loads, config.n_thermal
    for bus in range(config.n_buses):
        dev = make_bus_behaviour(config, bus)
        zb = z.at_bus[bus]
        out = dev.prox(zb, 2.0)
        if isinstance(dev, Generator):
            assert_foreign_columns_untouched(zb, out, {"p"})
            for rho in (0.5, 2.0, 50.0):
                got = dev.prox(zb, rho)["p"][:, 0]
                ref = generator_reference(config, bus, zb["p"][:, 0], rho)
                assert np.abs(got - ref).max() < 1e-5, f"generator prox at rho={rho} off by {np.abs(got - ref).max():.2e}"
        else:
            owned = {"p"} if isinstance(dev, Load) else {"p", "soc"} if isinstance(dev, Battery) else {"p", "temp"}
            assert_foreign_columns_untouched(zb, out, owned)
            # QP-based projections are accurate to ~sqrt(solver tol), see common._QP_TOL.
            assert (out - dev.prox(out, 2.0)).norm() < 1e-5, f"{type(dev).__name__} not idempotent"
            assert (out - dev.prox(zb, 50.0)).norm() < 1e-9, f"{type(dev).__name__} depends on rho"
        if isinstance(dev, Thermal):
            # Sign trap: a bus draw stays a draw. p = -0.5 in must come back non-positive.
            zt = zb.copy()
            zt["p"][:] = -0.5
            assert dev.prox(zt, 2.0)["p"].max() <= 1e-9, "thermal prox flipped the sign of p"
    # The device proxes are the only constraints g imposes; one pass must satisfy all of them.
    full = Trajectory({k: v.copy() for k, v in z.items()})
    for bus in range(config.n_buses):
        full.at_bus[bus] = make_bus_behaviour(config, bus).prox(full.at_bus[bus], 2.0)
    checks = check_solution(full, config)
    device_keys = ["gen_capacity", "gen_ramp", "load_mismatch", "thermal_dynamics", "thermal_band",
                   "thermal_box", "soc_dynamics", "soc_terminal", "soc_bounds", "battery_box"]
    worst = max(checks[k] for k in device_keys)
    assert worst < 1e-6, f"device constraints violated after one g pass: {worst:.2e}"
    print(f"  n={config.n_buses}: {n_gens} gen, {n_loads} load, {n_thermal} thermal, "
          f"{config.n_battery} battery proxes agree with references; worst device violation {worst:.1e}")


def to_centralized_layout(config: Config, z: Trajectory) -> Trajectory:
    """Repacks the ADMM trajectory into the width layout of centralized.solve."""
    n_gens, n_loads, n_thermal = config.n_gens, config.n_loads, config.n_thermal
    thermal_cols = slice(n_gens + n_loads, n_gens + n_loads + n_thermal)
    battery_cols = slice(n_gens + n_loads + n_thermal, config.n_buses)
    shift = thermal_shift(config)
    return Trajectory({
        "p": z["p"].copy(),
        "theta": z["theta"].copy(),
        "soc": z["soc"][:, battery_cols].copy(),
        "temp": (z["temp"] + shift)[:, thermal_cols],
    })


def test_integration(n_buses: int, obj_tol: float = 1e-3, gen_tol: float = 1e-2, **kwargs) -> Trajectory:
    """
    ADMM against the centralized optimum. Only the objective and the generator
    dispatch are unique (the cost is strictly convex in p_gen alone): the
    battery split, the thermal schedule and with them theta and the states can
    sit anywhere on the optimal face, so those distances are reported, not
    asserted. Returns the ADMM trajectory.
    """
    config = Config(n_buses=n_buses)
    ref = centralized.solve(config)
    start = time.perf_counter()
    res = admm_vanilla.solve(config, **kwargs)
    iters = len(res.convergence.r)
    threshold = 1e-3 * np.sqrt(n_buses / 8)
    print(f"  n={n_buses}: {iters} iterations in {time.perf_counter() - start:.1f}s "
          f"(r={res.p_residual:.2e}, s={res.s_residual:.2e}, rho={res.convergence.rho[-1]:.3g})")
    assert res.p_residual < threshold and res.s_residual < threshold, "ADMM hit max_iterations"

    n_gens = config.n_gens
    gap = abs(res.obj - ref.obj) / ref.obj
    gen_dist = np.abs(res.trajectory["p"][:, :n_gens] - ref.trajectory["p"][:, :n_gens]).max()
    p_dist = np.abs(res.trajectory["p"] - ref.trajectory["p"]).max()
    demean = lambda th: th - th.mean(axis=1, keepdims=True)
    theta_dist = np.abs(demean(res.trajectory["theta"]) - demean(ref.trajectory["theta"])).max()
    repacked = to_centralized_layout(config, res.trajectory)
    soc_dist = np.abs(repacked["soc"] - ref.trajectory["soc"]).max()
    temp_dist = np.abs(repacked["temp"] - ref.trajectory["temp"]).max()
    print(f"  objective {res.obj:.3f} vs {ref.obj:.3f} (gap {gap:.2e}); |dp_gen| {gen_dist:.2e}; "
          f"non-unique: |dp| {p_dist:.2e}, |dtheta| {theta_dist:.2e}, |dsoc| {soc_dist:.2e}, |dtemp| {temp_dist:.2e}")
    violations = {k: f"{v:.1e}" for k, v in res.checks.items() if k not in ("objective", "line_utilization") and v > 1e-6}
    print(f"  violations > 1e-6: {violations or 'none'}; line utilization {res.checks['line_utilization']:.3f}")

    assert gap < obj_tol, f"objective gap {gap:.2e}"
    assert gen_dist < gen_tol, f"generator dispatch differs by {gen_dist:.2e}"
    # Regression gate: the centralized checker on the ADMM result. z satisfies
    # the device constraints exactly; the network coupling (and so the power
    # balance) holds to the primal residual.
    check_centralized(config, repacked, imbalance_tol=10 * threshold)
    return res.trajectory


def test_executors(n_buses: int, reference: Trajectory) -> None:
    """The threaded and process executors must reproduce the sequential run."""
    config = Config(n_buses=n_buses)
    for parallel in ("threads", "processes"):
        start = time.perf_counter()
        res = admm_vanilla.solve(config, parallel=parallel)
        dist = (res.trajectory - reference).norm()
        print(f"  n={n_buses} {parallel}: {len(res.convergence.r)} iterations in "
              f"{time.perf_counter() - start:.1f}s, distance to sequential {dist:.1e}")
        assert dist < 1e-6, f"{parallel} executor diverged from the sequential run by {dist:.2e}"


def main(slow: bool) -> None:
    print("Network projection")
    for n in (5, 8, 45):
        test_network(Config(n_buses=n))
    print("Device proxes")
    for n in (5, 8):
        test_devices(Config(n_buses=n))
    print("Integration (fixed rho = 2)")
    sequential = {n: test_integration(n) for n in (5, 8)}
    print("Executors")
    test_executors(8, sequential[8])
    if slow:
        print("Integration, n = 45 (active line limits)")
        test_integration(45)
    print("all tests passed")


if __name__ == "__main__":
    main(slow="--slow" in sys.argv)
