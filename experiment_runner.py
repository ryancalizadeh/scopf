from typing import Any, Dict
from Config import Config

def experiment_runner(config: Config, n_runs: int=6) -> Dict[str, Any]:
    """
    Compares centralized, ADMM vanilla, ADMM Parallel, and ADMM Parallel + relaxation
    Takes as input a configuration, runs the algorithm for each configuration n_runs times
    Returns dispatch values, as well as mean and stddev of the objective value, residuals, and runtime
    """
    raise NotImplementedError("ExperimentRunner is not implemented yet. Please implement the function to run the experiments based on the input configuration and return the results.")
