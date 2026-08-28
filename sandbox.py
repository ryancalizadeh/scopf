"""
Manual smoke test for the algorithm implementations, using a small hardcoded
3-bus 2-generator config (as a plain dict, matching the low-level Proxable
interface rather than a randomly generated Config).
"""
import numpy as np
from Config import Config
from algorithms import centralized, admm_vanilla


def make_config() -> Config:
    config = Config(n_buses=3, gen_ratio=2 / 3, load_ratio=1 / 3, avg_degree=2.0)

    config.Y_bus = np.array([[ 4.902-19.608j, -2.941+11.765j, -1.961 +7.843j],
                              [-2.941+11.765j,  5.294-21.176j, -2.353 +9.412j],
                              [-1.961 +7.843j, -2.353 +9.412j,  4.314-17.255j]])
    config.G = np.real(config.Y_bus)
    config.B = np.imag(config.Y_bus)
    config.costs = np.array([1.0, 1.2])
    config.load_P = np.array([1.0])
    config.load_Q = np.array([0.5])
    config.V_max = np.array([1.1, 1.1, 1.1])
    config.V_min = np.array([0.9, 0.9, 0.9])
    config.P_max = np.array([1.2, 1.2])
    config.P_min = np.array([0.2, 0.2])
    config.Q_max = np.array([1.3, 1.3])
    config.Q_min = np.array([-0.2, -0.2])

    return config


if __name__ == "__main__":
    config = make_config()

    centralized_sol = centralized.solve(config)
    print("centralized:", centralized_sol)

    admm_sol = admm_vanilla.solve(config)
    print("admm_vanilla:", admm_sol)
