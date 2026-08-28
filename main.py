import numpy as np
import pickle

def main():
    with open('configs.pkl', 'rb') as f:
        configs = pickle.load(f)
    configs_exp1 = configs['exp1']
    configs_exp2 = configs['exp2']

    for key, config in configs_exp1.items():
        # G max:
        print(f"{key}: G max: {config.G.max()}, G min: {config.G.min()}")

        # B max:
        print(f"{key}: B max: {config.B.max()}, B min: {config.B.min()}")

        # Max connectivity bus:
        max_connectivity_bus = np.argmax(np.sum(config.Y_bus != 0, axis=1))
        print(f"{key}: Max connectivity bus: {max_connectivity_bus}, Connectivity: {np.sum(config.Y_bus[max_connectivity_bus] != 0)}")


if __name__ == "__main__":
    main()