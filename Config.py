import numpy as np
from Trajectory import Trajectory

class Config:
    """
    This data class contains all the configuration parameters for an experiment
    TODO Update this
    """
    T: float
    dt: float
    N: int
    n_buses: int
    n_gens: int
    n_loads: int
    B: np.ndarray

    def __init__(self, n_buses: int, gen_ratio: float, load_ratio: float, avg_degree: float,
                 T: float = 12, dt: float = 0.1, load_fraction: float = 0.3):
        rng = np.random.default_rng(seed=42)  # For reproducibility
        self.T = T
        self.dt = dt
        self.N = int(round(T / dt))
        self.n_buses = n_buses
        self.n_gens = int(n_buses * gen_ratio)
        self.n_loads = int(n_buses * load_ratio)
        self.Y_bus, self.G, self.B = self.generate_Y_bus(avg_degree)

        self.costs = rng.uniform(1.0, 2.0, self.n_gens)

        self.load_P = rng.uniform(0.1, 1.0, self.n_loads)
        self.load_Q = rng.uniform(0.0, 0.4, self.n_loads)

        self.V_max = np.full(n_buses, 1.15)
        self.V_min = np.full(n_buses, 0.85)

        self.P_min = np.zeros(self.n_gens)
        self.P_max = rng.uniform(1.0, 2.0, self.n_gens)
        self.Q_max = rng.uniform(0.5, 1.0, self.n_gens)
        self.Q_min = -self.Q_max


        self.line_flow_limits = 1.0  # This can be adjusted as needed

    def make_base_trajectory(self) -> Trajectory:
        raise NotImplementedError("make_base_trajectory is not implemented yet.")

    def generate_Y_bus(self, avg_degree: float):
        # TODO For now this is fine. In the future, I should follow the methodology of Birchfield 2017
        rng = np.random.default_rng(seed=42)  # For reproducibility
        n = self.n_buses
        edges = set()

        # random spanning tree to guarantee connectivity
        order = rng.permutation(n)
        for k in range(1, n):
            i = order[k]
            j = order[rng.integers(0, k)]
            edges.add((min(i, j), max(i, j)))

        # add extra random edges to approximate the target average degree
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        target_edges = int(round(avg_degree * n / 2))
        remaining = [pair for pair in all_pairs if pair not in edges]
        rng.shuffle(remaining)
        for pair in remaining:
            if len(edges) >= target_edges:
                break
            edges.add(pair)

        Y_bus = np.zeros((n, n), dtype=complex)
        for i, j in edges:
            R = rng.uniform(0.003, 0.01)
            X = rng.uniform(0.03, 0.07)
            y = 1 / (R + 1j * X)
            Y_bus[i, j] -= y
            Y_bus[j, i] -= y
            Y_bus[i, i] += y
            Y_bus[j, j] += y

        self.lines = sorted(edges)

        return Y_bus, Y_bus.real, Y_bus.imag # type: ignore

    def __getitem__(self, key: str):
        return getattr(self, key)