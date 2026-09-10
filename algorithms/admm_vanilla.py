import logging
import time
import numpy as np
from Config import Config
from Trajectory import Trajectory
from admm import admm
from algorithms.base import SolveResult, ConvergenceHistory
from algorithms.common import F, make_bus_behaviours, rho_for_size, check_solution

logger = logging.getLogger(__name__)


def solve(config: Config, parallel: "bool | str" = False) -> SolveResult:
    """
    parallel selects the bus-projection executor (see common.make_bus_behaviours):
    False / "sequential", True / "threads", or "processes".
    """
    n_buses = config.n_buses
    n_gens = config.n_gens
    N = config.N

    f = F(config)
    g = make_bus_behaviours(config, parallel=parallel)

    # Flat start over the whole horizon: unit voltages, no current, rotor
    # angles at zero and every machine at synchronous speed ("omega" is the
    # speed deviation, so zero).
    z0 = Trajectory({
        "v": np.ones((N, n_buses), dtype=complex),
        "i": np.zeros((N, n_buses), dtype=complex),
        "s": np.zeros((N, n_buses), dtype=complex),
        "delta": np.zeros((N, n_gens)),
        "omega": np.zeros((N, n_gens)),
    })

    start = time.perf_counter()
    # Uncapped geometric rho ramp with a size-dependent growth rate. The value
    # of rho itself hardly matters here (all proxes but the P(0) cost are
    # projections); it is the dual rescaling on every rho change that breaks
    # the limit cycles, so the ramp must run until the stop rule fires, and
    # larger networks need slower growth to avoid bias (see common.rho_for_size).
    # The residuals are norms over N * (6 n_buses + 2 n_gens) entries, so a
    # threshold proportional to sqrt(n_buses) keeps the per-entry (RMS)
    # mismatch constant across network sizes: 1e-3 at 8 buses.
    threshold = 1e-3 * np.sqrt(n_buses / 8)
    try:
        xs, zs, us, rs, ss, rhos = admm(
            f, g, z0, rho=rho_for_size(n_buses), threshold=threshold, max_iterations=10000
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
