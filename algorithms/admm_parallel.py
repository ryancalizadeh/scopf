from Config import Config
from algorithms.base import SolveResult
from algorithms.admm_vanilla import solve as _solve_vanilla


# Below this many buses the pinned-thread executor is faster (the per-iteration
# process messaging outweighs the parallel gain); above it, worker processes win.
PROCESS_THRESHOLD_BUSES = 40


def executor_for(config: Config) -> str:
    return "processes" if config.n_buses >= PROCESS_THRESHOLD_BUSES else "threads"


def solve(config: Config) -> SolveResult:
    return _solve_vanilla(config, parallel=executor_for(config))
