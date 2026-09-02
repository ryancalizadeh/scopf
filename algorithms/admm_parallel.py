from Config import Config
from algorithms.base import SolveResult
from algorithms.admm_vanilla import solve as _solve_vanilla


def solve(config: Config) -> SolveResult:
    return _solve_vanilla(config, parallel=True)
