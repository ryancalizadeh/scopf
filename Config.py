import numpy as np

class Config:
    """
    This data class contains all the configuration parameters for an experiment
    """
    T: float
    dt: float
    N: int
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
    line_flow_limits: float
    omega_s = 2 * np.pi * 60  # Synchronous speed in rad/s
    D: np.ndarray
    M: np.ndarray
    Xd: np.ndarray
    load_step_factor: float
    load_fraction: float

    def __init__(self, n_buses: int, gen_ratio: float, load_ratio: float, avg_degree: float,
                 T: float = 2.0, dt: float = 0.1, load_step_factor: float = 1.2,
                 load_fraction: float = 0.3):
        rng = np.random.default_rng(seed=42)  # For reproducibility
        self.T = T
        self.dt = dt
        self.N = int(T / dt)
        # Disturbance for the transient stability problem: every load is scaled
        # by this factor for all time steps n >= 1 (n = 0 is the pre-disturbance
        # steady state).
        self.load_step_factor = load_step_factor
        self.n_buses = n_buses
        self.n_gens = int(n_buses * gen_ratio)
        self.n_loads = int(n_buses * load_ratio)
        self.Y_bus, self.G, self.B = self.generate_Y_bus(avg_degree)

        self.D = rng.uniform(0.01, 0.05, self.n_gens)  # Damping coefficients
        self.M = rng.uniform(0.1, 0.5, self.n_gens)  # Inertia constants
        self.Xd = rng.uniform(0.1, 0.3, self.n_gens)  # Direct-axis reactances

        self.costs = rng.uniform(1.0, 2.0, self.n_gens)

        self.load_P = rng.uniform(0.1, 1.0, self.n_loads)
        self.load_Q = rng.uniform(0.0, 0.4, self.n_loads)

        self.V_max = np.full(n_buses, 1.15)
        self.V_min = np.full(n_buses, 0.85)

        self.P_min = np.zeros(self.n_gens)
        self.P_max = rng.uniform(1.0, 2.0, self.n_gens)
        self.Q_max = rng.uniform(0.5, 1.0, self.n_gens)
        self.Q_min = -self.Q_max

        # The raw random loads generally exceed the generation capacity (and the
        # reactive losses on the random tree-like network exhaust the Q limits),
        # which makes the OPF infeasible. Rescale the loads so that the total
        # real load is load_fraction of the total P_max (reactive loads scaled
        # by the same factor); 0.3 keeps the static OPF feasible up to 135 buses.
        self.load_fraction = load_fraction
        load_scale = load_fraction * self.P_max.sum() / self.load_P.sum()
        self.load_P = self.load_P * load_scale
        self.load_Q = self.load_Q * load_scale

        self.line_flow_limits = 1.0  # This can be adjusted as needed

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
            # Line impedances kept modest: with X ~ 0.1-0.2 p.u. on a tree-like
            # topology the reactive losses exceed the generators' Q limits.
            R = rng.uniform(0.003, 0.01)
            X = rng.uniform(0.03, 0.07)
            y = 1 / (R + 1j * X)
            Y_bus[i, j] -= y
            Y_bus[j, i] -= y
            Y_bus[i, i] += y
            Y_bus[j, j] += y

        return Y_bus, Y_bus.real, Y_bus.imag # type: ignore

    def __getitem__(self, key: str):
        return getattr(self, key)