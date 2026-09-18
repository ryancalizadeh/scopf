import numpy as np
from Trajectory import Trajectory

class Config:
    """
    This data class contains all the configuration parameters for an experiment
    TODO Update this
    """
    T: float # by default 24 hours, from midnight to midnight
    dt: float # by default 15 minutes (0.25 hours)
    N: int
    n_buses: int
    n_gens: int
    n_loads: int
    n_thermal: int
    n_battery: int
    B: np.ndarray

    def __init__(self,
                 n_buses: int,
                 gen_ratio: float,
                 load_ratio: float,
                 thermal_ratio: float,
                 battery_ratio: float,
                 avg_degree: float,
                 T: float = 24,
                 dt: float = 0.25
                ):
        rng = np.random.default_rng(seed=42)  # For reproducibility
        self.T = T
        self.dt = dt
        self.N = int(T / dt)

        assert np.isclose(
            gen_ratio + load_ratio + thermal_ratio + battery_ratio, 1.0
        ), "gen_ratio, load_ratio, thermal_ratio, and battery_ratio must sum to 1"

        self.n_buses = n_buses
        self.n_gens = int(n_buses * gen_ratio)
        self.n_loads = int(n_buses * load_ratio)
        self.n_thermal = int(n_buses * thermal_ratio)
        self.n_battery = (
            n_buses - self.n_gens - self.n_loads - self.n_thermal
        )
        assert (
            self.n_gens + self.n_loads + self.n_thermal + self.n_battery
            == self.n_buses
        )


        # Network:
        self.Y_bus, self.G, self.B = self.generate_Y_bus(avg_degree)
        self.line_flow_limits = 1.0  # This can be adjusted as needed

        # Gens:
        # P_min <= p(t) <= P_max
        # R_min <= p(t) - p(t+1) <= R_max
        # Cost: c(p) = alpha*P**2 + beta*P

        self.gen_P_min = np.zeros(self.n_gens)
        self.gen_P_max = rng.uniform(1.5, 3.0, self.n_gens)

        self.gen_R_min = -0.3 * self.gen_P_max
        self.gen_R_max =  0.3 * self.gen_P_max

        self.gen_cost_alpha = rng.uniform(0.05, 0.2, self.n_gens)
        self.gen_cost_beta = rng.uniform(10.0, 15.0, self.n_gens)

        # Loads:
        # Typical daily demand shape: overnight minimum, morning ramp, and
        # evening peak.  Each load receives a distinct scale and a small,
        # smooth random perturbation while remaining positive in p.u.
        hours = np.arange(self.N) * self.dt
        daily_shape = (
            0.62
            + 0.16 * np.exp(-0.5 * ((hours - 8.0) / 2.0) ** 2)
            + 0.30 * np.exp(-0.5 * ((hours - 19.0) / 3.0) ** 2)
        )
        daily_shape /= daily_shape.mean()

        if self.n_loads:
            load_scales = rng.uniform(0.7, 1.1, self.n_loads)[:, None]
            noise = rng.normal(0.0, 0.025, (self.n_loads, self.N))
            # Smooth the independent noise to avoid unrealistic rapid changes.
            noise = (noise + np.roll(noise, 1, axis=1) +
                     np.roll(noise, -1, axis=1)) / 3.0
            self.load_P = np.maximum(
                0.05, load_scales * daily_shape[None, :] * (1.0 + noise)
            )
        else:
            self.load_P = np.empty((0, self.N))

        # Batteries:
        # q denotes state of charge (SOC), while p denotes active power exchange.
        # Typical battery SOC ranges are between ~0.2 and 1.0, and the initial
        # SOC is chosen near the middle of the feasible range.
        self.battery_p_max = rng.uniform(0.5, 1.5, self.n_battery)
        self.battery_q_max = rng.uniform(0.8, 1.0, self.n_battery)
        self.battery_q_min = rng.uniform(0.1, 0.3, self.n_battery)
        self.battery_p0 = np.zeros(self.n_battery)
        self.battery_p_min = -self.battery_p_max
        self.battery_PT = self.battery_p0.copy()


        # Thermal loads:
         

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