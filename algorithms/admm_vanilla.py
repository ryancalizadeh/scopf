import copy
import logging
import time
import numpy as np
from Config import Config
from Trajectory import Trajectory
from admm import admm
from algorithms.base import SolveResult, ConvergenceHistory
from algorithms.common import F, make_bus_behaviours, rho_for_size, check_solution
from simulate import simulate_dispatch, stabilise_dispatch, peak_coi_angle_deg

logger = logging.getLogger(__name__)


def _flat_start(config: Config, N: int) -> Trajectory:
    """Unit voltages, no current, rotors at rest ("omega" is the speed deviation)."""
    return Trajectory({
        "v": np.ones((N, config.n_buses), dtype=complex),
        "i": np.zeros((N, config.n_buses), dtype=complex),
        "s": np.zeros((N, config.n_buses), dtype=complex),
        "delta": np.zeros((N, config.n_gens)),
        "omega": np.zeros((N, config.n_gens)),
    })


def _threshold(n_buses: int) -> float:
    # The residuals are norms over N * (6 n_buses + 2 n_gens) entries, so a
    # threshold proportional to sqrt(n_buses) keeps the per-entry (RMS)
    # mismatch constant across network sizes: 1e-3 at 8 buses.
    return 1e-3 * np.sqrt(n_buses / 8)


def warm_start(config: Config, parallel: "bool | str" = False, margin_deg: float = 90.0):
    """
    Starting trajectory for the transient ADMM, built without the centralized
    solver:
      1. ADMM on the static problem (the same split with N = 1: pre-fault
         network, loads, limits, cost) gives the economic dispatch;
      2. `stabilise_dispatch` moves the smallest fraction of the faulted
         generator's output onto the others that keeps the simulated peak COI
         angle below margin_deg;
      3. the forward simulation of that dispatch is the start.

    Why this is needed: from a flat start the splitting locks the faulted
    generator's P(0) in during the first few hundred iterations (the generator
    projection keeps only ~3 % of an economic push on P(0), because moving
    dispatch drags every machine's transient along by ~T^2/2M) and the rho ramp
    then freezes it, ~50 % from the optimum at the same cost gap; from an
    unstable start the first projections overshoot. rho schedules, Anderson
    acceleration and proximal damping were all tried (2026-09-11) and none
    moves the dispatch; a start that is economic and stable does.
    Returns (z0, info).
    """
    static = copy.copy(config)
    static.N, static.T, static.n_clear = 1, config.dt, 0
    f = F(static)
    g = make_bus_behaviours(static, parallel=parallel)
    try:
        xs, zs, us, rs, ss, rhos = admm(
            f, g, _flat_start(static, 1), rho=rho_for_size(config.n_buses),
            threshold=_threshold(config.n_buses), max_iterations=5000,
        )
    finally:
        close = getattr(g, "close", None)
        if close is not None:
            close()
    S0 = zs[-1]["s"][0, :config.n_gens]
    P_static, Q0 = np.real(S0), np.imag(S0)
    P, alpha, traj = stabilise_dispatch(config, P_static, Q0, margin_deg=margin_deg)
    z0 = Trajectory({k: np.array(traj[k]) for k in ["v", "i", "s", "delta", "omega"]})
    info = {"static_iterations": len(rs), "P_static": P_static, "P_start": P, "alpha": alpha,
            "start_peak_deg": peak_coi_angle_deg(config, traj)}
    return z0, info


def solve(config: Config, parallel: "bool | str" = False, warm: bool = True) -> SolveResult:
    """
    parallel selects the bus-projection executor (see common.make_bus_behaviours):
    False / "sequential", True / "threads", or "processes".
    warm=True starts from warm_start (economic, transiently stable); False is the
    flat start, kept for experiments (it converges to a far more expensive dispatch
    on fault cases, see warm_start).
    """
    n_buses = config.n_buses
    n_gens = config.n_gens
    N = config.N

    start = time.perf_counter()
    if warm:
        z0, ws = warm_start(config, parallel=parallel)
        logger.info(f"warm start: static ADMM {ws['static_iterations']} it, faulted-gen P "
                    f"{ws['P_static'][config.fault_bus]:.4f} -> {ws['P_start'][config.fault_bus]:.4f} "
                    f"(alpha={ws['alpha']:.3f}), start peak {ws['start_peak_deg']:.1f} deg")
    else:
        z0 = _flat_start(config, N)

    f = F(config)
    g = make_bus_behaviours(config, parallel=parallel)
    # Uncapped geometric rho ramp with a size-dependent growth rate. The value
    # of rho itself hardly matters here (all proxes but the P(0) cost are
    # projections); it is the dual rescaling on every rho change that breaks
    # the limit cycles, so the ramp must run until the stop rule fires, and
    # larger networks need slower growth to avoid bias (see common.rho_for_size).
    try:
        xs, zs, us, rs, ss, rhos = admm(
            f, g, z0, rho=rho_for_size(n_buses), threshold=_threshold(n_buses), max_iterations=10000
        )
    finally:
        # Worker processes/threads of the parallel executors are released here
        # rather than left to garbage collection.
        close = getattr(g, "close", None)
        if close is not None:
            close()
    runtime = time.perf_counter() - start

    z = xs[-1]
    checks = check_solution(z, config)
    dispatch = np.real(z["s"])[0, :n_gens]

    return SolveResult(
        dispatch=dispatch,
        obj=checks["objective"],
        runtime=runtime,
        p_residual=checks["network_residual"],
        s_residual=checks["power_balance_residual"],
        convergence=ConvergenceHistory(r=rs, s=ss, rho=rhos),
        trajectory=z,
    )
