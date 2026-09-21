import logging
import time
import numpy as np
from Config import Config
from Trajectory import Trajectory
from admm import admm
from algorithms.base import SolveResult, ConvergenceHistory
from algorithms.common import make_bus_behaviours, rho_for_size, check_solution, Network

logger = logging.getLogger(__name__)


def solve(config: Config, parallel: "bool | str" = False) -> SolveResult:
    """
    parallel selects the bus-projection executor (see common.make_bus_behaviours):
    False / "sequential", True / "threads", or "processes".
    """
    n_buses = config.n_buses

    # @claude put the network behaviour in here
    f = F(config)
    g = make_bus_behaviours(config, parallel=parallel)

    z0 = config.make_base_trajectory()

    start = time.perf_counter()
    threshold = 1e-3 * np.sqrt(n_buses / 8) # TODO Consider replacing this
    try:
        xs, zs, us, rs, ss, rhos = admm(
            f, g, z0, rho=rho_for_size(n_buses), threshold=threshold, max_iterations=10000,
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

    return SolveResult(
        trajectory=z,
        obj=checks["objective"],
        runtime=runtime,
        p_residual=checks["network_residual"],
        s_residual=checks["power_balance_residual"], # TODO This line and the above might need to change. Might just pass the whole checks object with all constraint residuals
        convergence=ConvergenceHistory(r=rs, s=ss, rho=rhos),
    )
