import numpy as np
from Config import Config
import pickle

def generate_configs():
    """
    Generates a list of Config objects with different parameters for experiments
    Returns a list of Config objects
    """
    gen_ratio = 0.2
    load_ratios = 0.6

    # Experiment 1:
    exp1_configs = {}
    n_buses_list = [5, 15, 45, 135, 405]
    avg_degrees = 2.0

    for n_buses in n_buses_list:
        config = Config(n_buses=n_buses, gen_ratio=gen_ratio, load_ratio=load_ratios, avg_degree=avg_degrees)
        exp1_configs[f"exp1_n_buses_{n_buses}"] = config

    # Experiment 2:
    exp2_configs = {}
    n_buses = 15
    avg_degrees = [2.0, 2.3, 2.6, 2.9]

    for avg_degree in avg_degrees:
        config = Config(n_buses=n_buses, gen_ratio=gen_ratio, load_ratio=load_ratios, avg_degree=avg_degree)
        exp2_configs[f"exp2_avg_degree_{avg_degree}"] = config

    # Save the configurations to a pickle file
    with open('configs.pkl', 'wb') as f:
        pickle.dump({'exp1': exp1_configs, 'exp2': exp2_configs}, f)

if __name__ == "__main__":
    generate_configs()