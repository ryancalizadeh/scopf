import numpy as np
import os
import pickle
from datetime import datetime
from experiment_runner import experiment_runner

RESULTS_DIR = "results"

def run_experiments():
    with open('configs.pkl', 'rb') as f:
        configs = pickle.load(f)
    configs_exp1 = configs['exp1']
    configs_exp2 = configs['exp2']

    os.makedirs(RESULTS_DIR, exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp1_results_path = os.path.join(RESULTS_DIR, f"exp1_results_{run_timestamp}.pkl")

    exp1_results = {}
    for key, config in configs_exp1.items():
        print(f"Running experiment for {key}")
        results = experiment_runner(config)
        exp1_results[key] = results

        with open(exp1_results_path, 'wb') as f:
            pickle.dump(exp1_results, f)

def visualize_results(file_path: str):
    raise NotImplementedError("Visualization function is not implemented yet. Please implement the function to visualize the results.")

if __name__ == "__main__":
    run_experiments()