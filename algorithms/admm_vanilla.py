import logging
import time
import numpy as np
from Config import Config
from Trajectory import Trajectory
from admm import admm
from algorithms.base import SolveResult, ConvergenceHistory
from algorithms.common import F, make_bus_behaviours, rho_geometric, check_solution

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
    # Geometric rho ramp: the residual-balancing heuristic is known not to work
    # for this nonconvex splitting (see common.rho_heuristic).
    try:
        xs, zs, us, rs, ss, rhos = admm(
            f, g, z0, rho=rho_geometric(), threshold=1e-2, max_iterations=10000
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
