import numpy as np
from Trajectory import Trajectory

class Config:
    """
    Container for the experiment configuration and generated operating data.

    Public attributes populated by __init__ include:

    - T: Horizon length in hours (default 24, midnight-to-midnight).
    - dt: Time step size in hours (default 0.25, i.e., 15 minutes).
    - N: Number of time steps, equal to int(T / dt).
    - seed: Random seed used to generate the network and device data.
    - n_buses: Total number of buses in the network.
    - n_gens: Number of generator buses.
    - n_loads: Number of load buses.
    - n_thermal: Number of thermal-load buses.
    - n_battery: Number of battery buses.
    - Y_bus: Complex admittance matrix for the network.
    - G: Real part of the admittance matrix (conductance matrix).
    - B: Imaginary part of the admittance matrix (susceptance matrix).
    - line_flow_limits: Per-line power-flow limits; default is ones matching B.
    - gen_P_min: Minimum active power output of each generator.
    - gen_P_max: Maximum active power output of each generator.
    - gen_R_min: Downward ramp-rate limit of each generator.
    - gen_R_max: Upward ramp-rate limit of each generator.
    - gen_cost_alpha: Quadratic cost coefficient for each generator.
    - gen_cost_beta: Linear cost coefficient for each generator.
    - load_P: Time-varying active power demand for each load bus over the horizon.
    - battery_p_max: Maximum charging/discharging power of each battery.
    - battery_p_min: Minimum battery power limit vector.
    - battery_q_max: Maximum state-of-charge upper bound of each battery.
    - battery_q_min: Minimum state-of-charge lower bound of each battery.
    - battery_q0: Battery SOC initial condition.
    - battery_qT: Battery SOC terminal condition.
    - thermal_mu: Conductance-like envelope parameters for thermal models.
    - thermal_c: Thermal mass parameters for thermal models.
    - thermal_eta: Heating/cooling effectiveness parameters.
    - thermal_T_amb: Ambient temperature profile across thermal buses and time.
    - thermal_T_min: Lower comfort bounds for each thermal bus over time.
    - thermal_T_max: Upper comfort bounds for each thermal bus over time.
    - thermal_T0: Initial temperature of each thermal bus.
    - thermal_p_min: Minimum control power for thermal units.
    - thermal_p_max: Maximum control power for thermal units.
    """
    T: float # by default 24 hours, from midnight to midnight
    dt: float # by default 15 minutes (0.25 hours)
    N: int
    n_buses: int
    n_gens: int
    n_loads: int
    n_thermal: int
    n_battery: int
    Y_bus: np.ndarray
    G: np.ndarray
    B: np.ndarray
    line_flow_limits: np.ndarray
    gen_P_min: np.ndarray
    gen_P_max: np.ndarray
    gen_R_min: np.ndarray
    gen_R_max: np.ndarray
    gen_cost_alpha: np.ndarray
    gen_cost_beta: np.ndarray
    load_P: np.ndarray
    battery_p_max: np.ndarray
    battery_p_min: np.ndarray
    battery_q_max: np.ndarray
    battery_q_min: np.ndarray
    battery_q0: np.ndarray
    battery_qT: np.ndarray
    thermal_mu: np.ndarray
    thermal_c: np.ndarray
    thermal_eta: np.ndarray
    thermal_T_amb: np.ndarray
    thermal_T_min: np.ndarray
    thermal_T_max: np.ndarray
    thermal_T0: np.ndarray
    thermal_p_min: np.ndarray
    thermal_p_max: np.ndarray

    def __init__(self,
                 n_buses: int,
                 avg_degree: float = 2.2,
                 gen_ratio: float = 0.25,
                 load_ratio: float = 0.25,
                 thermal_ratio: float = 0.25,
                 battery_ratio: float = 0.25,
                 T: float = 24,
                 dt: float = 0.25,
                 seed: int = 42
                ):
        self.seed = seed
        rng = np.random.default_rng(seed=self.seed)
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
        # line_flow_limits is set after the devices are sized, once peak demand
        # is known (see below).

        # Gens:
        # P_min <= p(t) <= P_max
        # R_min <= p(t) - p(t+1) <= R_max
        # Cost: c(p) = alpha*P**2 + beta*P

        self.gen_P_min = np.zeros(self.n_gens)
        self.gen_P_max = rng.uniform(1.5, 3.0, self.n_gens)

        self.gen_R_min = -0.3 * self.gen_P_max
        self.gen_R_max =  0.3 * self.gen_P_max

        self.gen_cost_alpha = rng.uniform(0.01, 0.08, self.n_gens)
        self.gen_cost_beta = rng.uniform(5.0, 7.0, self.n_gens)

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
        # Start mid-range: q0 must lie inside [q_min, q_max] or the SOC bounds
        # are violated at t = 0 before any dispatch is chosen.
        self.battery_q0 = 0.5 * (self.battery_q_min + self.battery_q_max)
        self.battery_p_min = -self.battery_p_max
        self.battery_qT = self.battery_q0.copy()


        # Thermal loads:
        # governed by: T(t+1) = (1-mu/c)*T(t) + eta/c*p(t) + mu/c*T_amb(t)
        # p(t) > 0 heats (and draws that power from the bus), p(t) < 0 cools.
        # T(t) \in [T_min(t), T_max(t)]
        # T(0) = T_0
        # P(t) \in [P_min, P_max]
        self.thermal_mu = rng.uniform(0.15, 0.25, self.n_thermal)  # envelope conductance
        self.thermal_c = rng.uniform(1.5, 2.5, self.n_thermal)     # thermal mass
        # Steady-state lift above ambient at full heat is eta*p_max/mu. The x4
        # sizes heaters to reach the 20-23 C occupied band from an 8-15 C
        # ambient (~15 C of lift); at the unscaled eta the band is unreachable.
        # x3 clears the median draw but leaves some seeds short, so x4 is used
        # to keep every network size feasible.
        self.thermal_eta = rng.uniform(0.8, 1.2, self.n_thermal) * 4  # heating/cooling effectiveness

        # Ambient temperature: a fall day in Vancouver, BC. Roughly a smooth
        # diurnal cycle bottoming out overnight (~8C) and peaking mid-afternoon
        # (~15C), plus small smoothed noise per thermal load (weather is
        # spatially correlated but not identical across buses).
        T_amb_low = 8.0
        T_amb_high = 15.0
        T_amb_mean = 0.5 * (T_amb_low + T_amb_high)
        T_amb_amp = 0.5 * (T_amb_high - T_amb_low)
        # Diurnal cycle troughs near 5am and peaks near 15:00 (3pm).
        daily_T_amb = T_amb_mean - T_amb_amp * np.cos(
            2 * np.pi * (hours - 5.0) / 24.0
        )

        if self.n_thermal:
            amb_noise = rng.normal(0.0, 0.3, (self.n_thermal, self.N))
            amb_noise = (amb_noise + np.roll(amb_noise, 1, axis=1) +
                         np.roll(amb_noise, -1, axis=1)) / 3.0
            self.thermal_T_amb = daily_T_amb[None, :] + amb_noise
        else:
            self.thermal_T_amb = np.empty((0, self.N))

        # Comfort deadbands: buildings occupied 8am-5pm, held tightly to
        # [20, 23] C during those hours; outside occupied hours the deadband
        # relaxes to a wide setback [12, 28] C so temperature can drift.
        occupied = (hours >= 8.0) & (hours < 17.0)
        T_min_profile = np.where(occupied, 20.0, 12.0)
        T_max_profile = np.where(occupied, 23.0, 28.0)

        self.thermal_T_min = np.tile(T_min_profile, (self.n_thermal, 1))
        self.thermal_T_max = np.tile(T_max_profile, (self.n_thermal, 1))

        self.thermal_T0 = np.full(self.n_thermal, 21.0)
        self.thermal_p_min = np.zeros(self.n_thermal)
        self.thermal_p_max = np.ones(self.n_thermal)

        # Line limits, sized now that the devices are sized. A flat 1.0 p.u. is
        # far too tight: the network must route the fixed load plus thermal
        # heating out of a handful of generator buses, some of which sit on a
        # single line and must push their whole output through it.
        #
        # The smallest feasible limit depends on the random topology (measured
        # between 1.0 and 3.5 p.u. over n_buses = 5..105), not on size alone, so
        # it is sized off the largest generator with margin for the routing.
        # This stays binding in practice - the optimum rides the limit on the
        # busiest line - while keeping every generated network feasible.
        self.line_flow_limits = 1.5 * float(self.gen_P_max.max()) * np.ones_like(self.B)


    def make_base_trajectory(self) -> Trajectory:
        """
        Starting point for the distributed solvers. Every key has width n_buses
        so that Trajectory.at_bus[i] hands each device its own column:

            p      bus net injection (sign as in algorithms/centralized.py)
            theta  bus voltage angle
            soc    battery SOC, zero on non-battery columns
            temp   thermal state minus thermal_T0, zero on non-thermal columns

        Loads sit at their exact demand, generators split the total fixed load
        evenly (clipped to their capacity), flexible devices draw nothing, and
        the states start inside their feasible sets (soc at q0, temp at T0).
        """
        N, n = self.N, self.n_buses
        n_gens, n_loads, n_thermal = self.n_gens, self.n_loads, self.n_thermal

        p = np.zeros((N, n))
        p[:, n_gens:n_gens + n_loads] = -self.load_P.T
        if n_gens:
            share = self.load_P.sum(axis=0) / n_gens                     # (N,)
            p[:, :n_gens] = np.clip(share[:, None], self.gen_P_min, self.gen_P_max)

        soc = np.zeros((N, n))
        soc[:, n_gens + n_loads + n_thermal:] = self.battery_q0

        return Trajectory({
            "p": p,
            "theta": np.zeros((N, n)),
            "soc": soc,
            "temp": np.zeros((N, n)),
        })

    def generate_Y_bus(self, avg_degree: float):
        # TODO For now this is fine. In the future, I should follow the methodology of Birchfield 2017
        rng = np.random.default_rng(seed=self.seed)
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