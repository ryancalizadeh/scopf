import numpy as np

class Config:
    """
    This data class contains all the configuration parameters for an experiment
    """
    n_buses: int
    n_gens: int
    n_loads: int
    Y_bus: np.ndarray
    G: np.ndarray
    B: np.ndarray
    costs: np.ndarray
    load_P: np.ndarray
    load_Q: np.ndarray
    V_max: np.ndarray
    V_min: np.ndarray
    P_max: np.ndarray
    P_min: np.ndarray
    Q_max: np.ndarray
    Q_min: np.ndarray

    def __init__(self, n_buses: int, gen_ratio: float, load_ratio: float, avg_degree: float):
        self.n_buses = n_buses
        self.n_gens = int(n_buses * gen_ratio)
        self.n_loads = int(n_buses * load_ratio)
        self.Y_bus, self.G, self.B = self.generate_Y_bus(avg_degree)

        self.load_P = np.random.uniform(0.1, 1.0, self.n_loads)
        self.load_Q = np.random.uniform(0.0, 0.4, self.n_loads)

        self.V_max = np.full(n_buses, 1.05)
        self.V_min = np.full(n_buses, 0.95)

        self.P_min = np.zeros(self.n_gens)
        self.P_max = np.random.uniform(1.0, 2.0, self.n_gens)
        self.Q_max = np.random.uniform(0.5, 1.0, self.n_gens)
        self.Q_min = -self.Q_max

    def generate_Y_bus(self, avg_degree: float):
        # TODO For now this is fine. In the future, I should follow the methodology of Birchfield 2017
        n = self.n_buses
        edges = set()

        # random spanning tree to guarantee connectivity
        order = np.random.permutation(n)
        for k in range(1, n):
            i = order[k]
            j = order[np.random.randint(0, k)]
            edges.add((min(i, j), max(i, j)))

        # add extra random edges to approximate the target average degree
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        target_edges = int(round(avg_degree * n / 2))
        remaining = [pair for pair in all_pairs if pair not in edges]
        np.random.shuffle(remaining)
        for pair in remaining:
            if len(edges) >= target_edges:
                break
            edges.add(pair)

        Y_bus = np.zeros((n, n), dtype=complex)
        for i, j in edges:
            R = np.random.uniform(0.01, 0.03)
            X = np.random.uniform(0.1, 0.2)
            y = 1 / (R + 1j * X)
            Y_bus[i, j] -= y
            Y_bus[j, i] -= y
            Y_bus[i, i] += y
            Y_bus[j, j] += y

        return Y_bus, Y_bus.real, Y_bus.imag # type: ignore