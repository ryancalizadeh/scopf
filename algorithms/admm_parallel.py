from Config import Config
from algorithms.base import SolveResult
from algorithms.admm_vanilla import solve as _solve_vanilla


def solve(config: Config) -> SolveResult:
    # Bus projections in worker processes (see common.BusBehavioursProcesses);
    # pass parallel="threads" for the pinned-thread executor instead.
    return _solve_vanilla(config, parallel="processes")
