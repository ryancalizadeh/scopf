import time
import numpy as np
from Config import Config
from Trajectory import Trajectory
from admm import admm
from algorithms.base import SolveResult, ConvergenceHistory
from algorithms.common import F, make_bus_behaviours, rho_geometric, check_solution


def solve(config: Config, parallel: bool = False) -> SolveResult:
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
    xs, zs, us, rs, ss, rhos = admm(
        f, g, z0, rho=rho_geometric(), threshold=2e-3, max_iterations=10000
    )
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
