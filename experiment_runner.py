from typing import Dict, List
from Config import Config
from ExperimentResult import ExperimentResult
from algorithms import ALGORITHMS
from visualize import plot_convergence


def experiment_runner(config: Config, n_runs: int = 6, plot_convergence_flag: bool = False
                       ) -> Dict[str, List[ExperimentResult]]:
    """
    Compares centralized, ADMM vanilla, ADMM Parallel, and ADMM Parallel + relaxation
    Takes as input a configuration, runs each implemented algorithm n_runs times.
    Returns raw per-run results keyed by algorithm name; algorithms that raise
    NotImplementedError are skipped.
    """
    results: Dict[str, List[ExperimentResult]] = {}

    for name, solve in ALGORITHMS.items():
        try:
            runs: List[ExperimentResult] = []
            first_convergence = None
            for i in range(n_runs):
                sol = solve(config)
                if i == 0:
                    first_convergence = sol.convergence
                runs.append(ExperimentResult(
                    config=config,
                    algorithm=name,
                    dispatch=sol.dispatch,
                    obj=sol.obj,
                    runtime=sol.runtime,
                    p_residual=sol.p_residual,
                    s_residual=sol.s_residual,
                ))

            if plot_convergence_flag and first_convergence is not None:
                plot_convergence(first_convergence, title=f"{name} convergence")

            results[name] = runs
        except NotImplementedError:
            print(f"skipping {name}: not implemented")

    return results
