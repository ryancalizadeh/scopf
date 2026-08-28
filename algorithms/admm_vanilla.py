import time
import numpy as np
from Config import Config
from ZDict import ZDict
from admm import admm
from algorithms.base import SolveResult, ConvergenceHistory
from algorithms.common import F, make_bus_behaviours, rho_heuristic, check_solution


def solve(config: Config) -> SolveResult:
    config_dict = config.as_dict()
    n_buses = config.n_buses
    n_gens = config.n_gens

    f = F(config_dict)
    g = make_bus_behaviours(config_dict)

    z0 = ZDict({
        "v": np.ones(n_buses, dtype=complex),
        "i": np.zeros(n_buses, dtype=complex),
        "s": np.zeros(n_buses, dtype=complex),
    })

    start = time.perf_counter()
    xs, zs, us, rs, ss, rhos = admm(
        f, g, z0, rho=rho_heuristic, threshold=5e-5, max_iterations=10000
    )
    runtime = time.perf_counter() - start

    z = xs[-1]
    checks = check_solution(z, config_dict)
    dispatch = np.real(z["s"])[:n_gens]

    return SolveResult(
        dispatch=dispatch,
        obj=checks["objective"],
        runtime=runtime,
        p_residual=checks["network_residual"],
        s_residual=checks["power_balance_residual"],
        convergence=ConvergenceHistory(r=rs, s=ss, rho=rhos),
    )
