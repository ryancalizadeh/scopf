from typing import Dict, List
import numpy as np
from Config import Config
from ExperimentResult import ExperimentResult
from algorithms import ALGORITHMS
from algorithms.centralized import check_feasible
from visualize import plot_convergence
import logging
from algorithms.base import SolveResult

logger = logging.getLogger(__name__)


def experiment_runner(configs: List[Config], plot_convergence_flag: bool = False
                       ) -> Dict[str, List[ExperimentResult]]:
    """
    Compares centralized, ADMM vanilla, and ADMM Parallel.
    Takes as input a list of configs (one per run, typically identical
    parameters with distinct seeds so each run is a genuinely different
    problem instance). Configs that fail check_feasible are dropped before
    any algorithm sees them: solve() would raise on them anyway, and ADMM has
    no such check, so it would just burn through max_iterations. Runs each
    implemented algorithm once per remaining config. Returns raw per-run
    results keyed by algorithm name; algorithms that raise NotImplementedError
    are skipped.
    """
    feasible_configs = []
    for config in configs:
        if check_feasible(config):
            feasible_configs.append(config)
        else:
            logger.warning(f"Skipping infeasible config (seed={config.seed}, n_buses={config.n_buses})")
    configs = feasible_configs

    results: Dict[str, List[ExperimentResult]] = {}
    n_runs = len(configs)

    for name, solve in ALGORITHMS.items():
        logger.info(f"Running {name} for {n_runs} runs...")
        try:
            runs: List[ExperimentResult] = []
            first_convergence = None
            for i, config in enumerate(configs):
                sol: SolveResult = solve(config)
                if i == 0:
                    first_convergence = sol.convergence
                runs.append(ExperimentResult(
                    config=config,
                    algorithm=name,
                    trajectory=sol.trajectory,
                    obj=sol.obj,
                    runtime=sol.runtime,
                    p_residual=sol.p_residual,
                    s_residual=sol.s_residual,
                    convergence=sol.convergence
                ))
                logger.debug(f"Run {i + 1}/{n_runs} for {name} completed in {sol.runtime:.2f} seconds with objective {sol.obj} and primal residual {sol.p_residual}.")


            if plot_convergence_flag and first_convergence is not None:
                plot_convergence(first_convergence, title=f"{name} convergence")

            results[name] = runs
        except NotImplementedError:
            logger.warning(f"skipping {name}: not implemented")

    return results
