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
    H: np.ndarray
    Xd: np.ndarray
    load_step_factor: float
    load_fraction: float
    t_clear: float
    n_clear: int
    fault_shunt: complex
    fault_line: tuple
    fault_bus: int
    lines: list
    Y_transient: np.ndarray
    Y_fault: np.ndarray
    Y_post: np.ndarray

    def __init__(self, n_buses: int, gen_ratio: float, load_ratio: float, avg_degree: float,
                 T: float = 2.0, dt: float = 0.05, load_step_factor: float = 1.0,
                 load_fraction: float = 0.3, t_clear: float = 0.6,
                 fault_shunt: complex = -1e3j):
        rng = np.random.default_rng(seed=42)  # For reproducibility
        self.T = T
        self.dt = dt
        self.N = int(round(T / dt))
        # Legacy load-step disturbance, superseded by the line fault below.
        # Kept (defaulting to 1.0, i.e. no step) so older pickles still load.
        self.load_step_factor = load_step_factor
        self.n_buses = n_buses
        self.n_gens = int(n_buses * gen_ratio)
        self.n_loads = int(n_buses * load_ratio)
        self.Y_bus, self.G, self.B = self.generate_Y_bus(avg_degree)

        self.D = rng.uniform(0.01, 0.05, self.n_gens)  # Damping coefficients
        # Inertia constants. M = 2H / omega_s with H the stored-energy constant
        # in seconds; H of 2-6 s covers typical synchronous machines. (Earlier
        # versions drew M directly from 0.1-0.5, i.e. H of 23-92 s, which made
        # the rotors far too heavy to be disturbed by any realistic fault.)
        self.H = rng.uniform(2.0, 6.0, self.n_gens)
        self.M = 2.0 * self.H / self.omega_s
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

        # Contingency: a bolted fault on a randomly chosen line, applied from
        # the first time step and cleared at t_clear by tripping that line.
        self.t_clear = t_clear
        self.n_clear = int(round(t_clear / dt))
        self.fault_shunt = fault_shunt
        self._setup_fault(rng)

    # ------------------------------------------------------------------
    # Fault contingency
    # ------------------------------------------------------------------

    def _setup_fault(self, rng) -> None:
        """
        Picks the faulted line and builds the three admittance matrices used by
        the transient: Y_bus (pre-disturbance), Y_fault (fault on) and Y_post
        (fault cleared by tripping the line).

        For t >= 1 the loads are represented as constant impedances folded into
        the admittance matrix, so that the network stays solvable at the
        depressed voltages of a fault (constant-power loads cannot be served at
        |V| ~ 0.01). The load current is unchanged in magnitude: the term
        y_L * V inside Y_transient @ V carries exactly what the load device drew
        before, so what becomes zero for t >= 1 is only the *residual* device
        injection I - Y_transient @ V at that bus.
        """
        candidates = self._eligible_fault_lines()
        if not candidates:
            raise ValueError(
                f"No eligible fault line for n_buses={self.n_buses}: every line incident to a "
                "generator bus is a bridge (its removal would island the network). Increase "
                "avg_degree so the topology has more cycles."
            )
        k, l, fault_bus = candidates[rng.integers(len(candidates))]
        self.fault_line = (k, l)
        self.fault_bus = fault_bus

        # Constant-impedance loads at the nominal voltage of 1 p.u.
        Y_transient = self.Y_bus.copy()
        for bus in range(self.n_gens, self.n_gens + self.n_loads):
            idx = bus - self.n_gens
            Y_transient[bus, bus] += self.load_P[idx] - 1j * self.load_Q[idx]
        self.Y_transient = Y_transient

        # Fault on: a large shunt at the faulted end of the line.
        self.Y_fault = Y_transient.copy()
        self.Y_fault[fault_bus, fault_bus] += self.fault_shunt

        # Fault cleared: the faulted line is removed.
        self.Y_post = Y_transient.copy()
        y_kl = -self.Y_bus[k, l]
        self.Y_post[k, l] += y_kl
        self.Y_post[l, k] += y_kl
        self.Y_post[k, k] -= y_kl
        self.Y_post[l, l] -= y_kl

    def _eligible_fault_lines(self) -> list:
        """
        Lines that may carry the fault: incident to a generator bus (faults far
        from any machine barely move the rotor angles) and removable without
        islanding the network (the line is tripped to clear the fault).
        Returns (k, l, fault_bus) with fault_bus the generator end.
        """
        eligible = []
        for (k, l) in self.lines:
            if k >= self.n_gens and l >= self.n_gens:
                continue
            if not self._connected_without((k, l)):
                continue
            eligible.append((k, l, k if k < self.n_gens else l))
        return eligible

    def _connected_without(self, line: tuple) -> bool:
        """True if the network stays connected when `line` is removed."""
        adjacency = {bus: set() for bus in range(self.n_buses)}
        for (a, b) in self.lines:
            if (a, b) == line:
                continue
            adjacency[a].add(b)
            adjacency[b].add(a)
        seen = {0}
        stack = [0]
        while stack:
            bus = stack.pop()
            for neighbour in adjacency[bus]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        return len(seen) == self.n_buses

    def Y_at(self, n: int) -> np.ndarray:
        """
        Admittance matrix at time step n: pre-disturbance at n = 0, faulted for
        1 <= n <= n_clear, post-fault (line tripped) afterwards.
        """
        if n == 0:
            return self.Y_bus
        return self.Y_fault if n <= self.n_clear else self.Y_post

    def load_admittance(self) -> np.ndarray:
        """
        Constant-impedance load admittances y_L (one per load, in load order),
        as folded into Y_transient. The transient consumption of a load bus is
        |V(t)|^2 * conj(y_L); the trajectory's S is zero there for t >= 1
        because the current is carried by the admittance matrix instead.
        """
        return self.load_P - 1j * self.load_Q

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

        self.lines = sorted(edges)

        return Y_bus, Y_bus.real, Y_bus.imag # type: ignore

    def __getitem__(self, key: str):
        return getattr(self, key)