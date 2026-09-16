import os
import pickle
from datetime import datetime
from experiment_runner import experiment_runner
from ExperimentResult import aggregate
from visualize import plot_runtime_scaling
import logging
import sys

RESULTS_DIR = "results"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


def _run_sweep(configs: dict, results_dir: str, run_timestamp: str, sweep_name: str):
    raw_results = {}
    aggregated_results = {}

    for key, config in configs.items():
        logger.info(f"Running experiment for {key}")
        results = experiment_runner(config, n_runs=1, plot_convergence_flag=True)
        raw_results[key] = results
        aggregated_results[key] = {name: aggregate(runs) for name, runs in results.items()}

        with open(os.path.join(results_dir, f"{sweep_name}_raw_{run_timestamp}.pkl"), 'wb') as f:
            pickle.dump(raw_results, f)
        with open(os.path.join(results_dir, f"{sweep_name}_aggregated_{run_timestamp}.pkl"), 'wb') as f:
            pickle.dump(aggregated_results, f)

    return raw_results, aggregated_results


def run_experiments():
    with open('configs.pkl', 'rb') as f:
        configs = pickle.load(f)
    configs_exp1 = configs['exp1']

    os.makedirs(RESULTS_DIR, exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("=== exp1: n_buses vs runtime ===")
    _, exp1_aggregated = _run_sweep(configs_exp1, RESULTS_DIR, run_timestamp, "exp1")

    exp1_by_n_buses = {config.n_buses: agg for config, agg in
                        zip(configs_exp1.values(), exp1_aggregated.values())}

    fig1 = plot_runtime_scaling(exp1_by_n_buses, xlabel="n_buses", title="exp1: n_buses vs runtime")
    fig1.savefig(os.path.join(RESULTS_DIR, f"exp1_runtime_{run_timestamp}.png"))


if __name__ == "__main__":
    run_experiments()
