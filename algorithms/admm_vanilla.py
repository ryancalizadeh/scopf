import logging
import time
import numpy as np
from Config import Config
from Trajectory import Trajectory
from admm import admm
from algorithms.base import SolveResult, ConvergenceHistory
from algorithms.common import F, make_bus_behaviours, rho_for_size, check_solution
from algorithms.metric import LeverageMetric

logger = logging.getLogger(__name__)


def leverage_metric_updater(g, alpha_max: float = 0.99, refresh: "int | None" = None, first: int = 0):
    """
    metric_update callback for admm.admm: at iteration `first` and then every `refresh`
    iterations (never, if None) measures each generator's response to a push on P(0) at the
    current z (g.leverage_directions) and returns the rank-one leverage metric built from it
    (algorithms.metric.LeverageMetric.from_responses).
    """
    def update(iteration, z, u):
        due = iteration == first or (refresh is not None and iteration > first and (iteration - first) % refresh == 0)
        if not due:
            return None
        responses = g.leverage_directions(z)
        metric = LeverageMetric.from_responses(responses, alpha_max=alpha_max)
        logger.info("leverage metric at iteration %d: pass-through r_P = %s",
                    iteration, {i: round(rP, 4) for i, (_, rP) in sorted(responses.items())})
        return metric
    return update


def solve(config: Config, parallel: "bool | str" = False, metric: bool = False,
          metric_refresh: "int | None" = 200, alpha_max: float = 0.99, metric_first: int = 0) -> SolveResult:
    """
    parallel selects the bus-projection executor (see common.make_bus_behaviours):
    False / "sequential", True / "threads", or "processes".
    metric=True runs the variable-metric ADMM (algorithms.metric): the proxes are taken in the
    rank-one leverage metric measured at iteration metric_first and refreshed every
    metric_refresh iterations (None = never); alpha_max caps the shrink along each generator's
    leverage direction. metric=False (default) is the plain Euclidean iteration.

    The metric is off by default because it does not change the outcome (measured 2026-09-14,
    15-bus fault case, flat start): it restores the pass-through of an economic push on P(0)
    (one F∘G round trip moves the faulted generator's P(0) 5x further), but the iteration
    still lands at the same ~44 % dispatch error as the Euclidean one. The cause is not the
    conditioning but a limit cycle of the fixed-rho iteration, which the rho ramp damps into a
    biased average: at fixed rho the residual oscillates (r ~ 0.5-3) with the fault on and
    converges without it (line trip only: 0.4 % error in 400 iterations), independently of rho
    (10 / 50 / 200 give identical traces: all proxes but the P(0) clip are projections) and of
    the metric (rank-one leverage, or down-weighting the t >= 1 blocks: w = 0.3 / 0.1 still
    cycle, w <= 0.03 diverges).
    """
    n_buses = config.n_buses
    n_gens = config.n_gens
    N = config.N

    f = F(config)
    g = make_bus_behaviours(config, parallel=parallel)
    metric_update = leverage_metric_updater(g, alpha_max, metric_refresh, metric_first) if metric else None

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
            f, g, z0, rho=rho_for_size(n_buses), threshold=threshold, max_iterations=10000,
            metric_update=metric_update,
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
