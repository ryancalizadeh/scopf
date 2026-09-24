import logging
import time
from typing import Callable, Optional
import numpy as np
from Config import Config
from admm import admm
from algorithms.base import SolveResult, ConvergenceHistory
from algorithms.common import make_bus_behaviours, rho_fixed, check_solution, Network

logger = logging.getLogger(__name__)


def solve(config: Config, parallel: "bool | str" = False, *,
          rho: Optional[Callable] = None, max_iterations: int = 1000) -> SolveResult:
    """
    Dynamic DC-OPF of centralized.py by ADMM, split as
        f = Network        (DC power flow + line limits, one projection per iteration)
        g = BusBehaviours  (per-bus device constraints, carrying the generator cost)
    over the base-trajectory layout of Config.make_base_trajectory.

    parallel selects the bus-projection executor (see common.make_bus_behaviours):
    False / "sequential", True / "threads", or "processes".
    rho is a schedule (iteration, rho_prev, r, s) -> rho; default a fixed 2.0
    (see common.rho_fixed for the comparison against the adaptive schedules).

    Returns the last z, which satisfies every device constraint exactly and the
    network constraints to within the primal residual, so the reported
    objective is that of a device-feasible dispatch.
    """
    n_buses = config.n_buses

    f = Network(config)
    g = make_bus_behaviours(config, parallel=parallel)

    z0 = config.make_base_trajectory()

    start = time.perf_counter()
    threshold = 1e-3 * np.sqrt(n_buses / 8) # TODO Consider replacing this
    try:
        xs, zs, us, rs, ss, rhos = admm(
            f, g, z0, rho=rho or rho_fixed(2.0), threshold=threshold,
            max_iterations=max_iterations,
        )
    finally:
        # Worker processes/threads of the parallel executors are released here
        # rather than left to garbage collection.
        close = getattr(g, "close", None)
        if close is not None:
            close()
    runtime = time.perf_counter() - start

    z = zs[-1]
    checks = check_solution(z, config)
    logger.info(f"ADMM finished after {len(rs)} iterations: r={rs[-1]:.3g}, s={ss[-1]:.3g}, "
                f"objective={checks['objective']:.3f}, network residual={checks['network_residual']:.3g}")

    return SolveResult(
        trajectory=z,
        obj=checks["objective"],
        runtime=runtime,
        p_residual=rs[-1],
        s_residual=ss[-1],
        convergence=ConvergenceHistory(r=rs, s=ss, rho=rhos),
        checks=checks,
    )
