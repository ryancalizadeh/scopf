import numpy as np
import casadi as ca
import cvxpy as cp
from scipy.linalg import expm
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, cast
from Config import Config

# The generator DAE helpers accept plain numbers as well as casadi symbols
# (MX / SX / DM); the arithmetic inside them is +, *, ca.cos and ca.sin only.
Scalar = Any
from Trajectory import Trajectory
from Proxable import Proxable


class ConstPowerLoad(Proxable):
    """
    A class implementing projections onto the behaviour of a constant power load
    """

    def __init__(self, config: Config, bus_index: int, load_index: int, max_iter=20, tol=1e-5):
        self.config = config
        self.bus_index = bus_index
        self.load_index = load_index
        self.max_iter = max_iter
        self.tol = tol

        V_max = config["V_max"][bus_index]
        V_min = config["V_min"][bus_index]
        # A load draws power from the network, i.e. it injects the negative
        # of its consumption.
        self.S = -(config["load_P"][load_index] + 1j * config["load_Q"][load_index])

        self.opti = ca.Opti()
        self.V_re = self.opti.variable()
        self.V_im = self.opti.variable()
        self.I_re = self.opti.variable()
        self.I_im = self.opti.variable()

        self.P = self.opti.parameter()
        self.Q = self.opti.parameter()
        self.opti.set_value(self.P, np.real(self.S))
        self.opti.set_value(self.Q, np.imag(self.S))

        self.v0r = self.opti.parameter()
        self.v0i = self.opti.parameter()
        self.i0r = self.opti.parameter()
        self.i0i = self.opti.parameter()

        self.opti.minimize((self.V_re - self.v0r)**2 + (self.V_im - self.v0i)**2 + (self.I_re - self.i0r)**2 + (self.I_im - self.i0i)**2)

        self.opti.subject_to(self.V_re * self.I_re + self.V_im * self.I_im == self.P)
        self.opti.subject_to(self.V_im * self.I_re - self.V_re * self.I_im == self.Q)
        self.opti.subject_to(self.V_re**2 + self.V_im**2 <= V_max)
        self.opti.subject_to(self.V_re**2 + self.V_im**2 >= V_min)

        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        self.opti.solver('ipopt', opts)

    def prox(self, z: Trajectory, rho: float = 1.0) -> Trajectory:
        ret = z.copy()
        V0 = ret["v"][0, 0]
        I0 = ret["i"][0, 0]

        self.opti.set_value(self.v0r, np.real(V0))
        self.opti.set_value(self.v0i, np.imag(V0))
        self.opti.set_value(self.i0r, np.real(I0))
        self.opti.set_value(self.i0i, np.imag(I0))

        sol = self.opti.solve()

        ret["v"][0, 0] = sol.value(self.V_re) + 1j * sol.value(self.V_im)
        ret["i"][0, 0] = sol.value(self.I_re) + 1j * sol.value(self.I_im)
        ret["s"][0, 0] = self.S

        return ret


class Generator(Proxable):
    """
    A class implementing projections onto the behaviour of a flexible generator
    """

    def __init__(self, config: Config, bus_index: int, gen_index: int):
        self.config = config
        self.bus_index = bus_index
        self.gen_index = gen_index

        V_min = config["V_min"][bus_index]
        V_max = config["V_max"][bus_index]

        self.opti = ca.Opti()
        self.V_re = self.opti.variable()
        self.V_im = self.opti.variable()
        self.I_re = self.opti.variable()
        self.I_im = self.opti.variable()
        self.P = self.opti.variable()
        self.Q = self.opti.variable()

        self.v0r = self.opti.parameter()
        self.v0i = self.opti.parameter()
        self.i0r = self.opti.parameter()
        self.i0i = self.opti.parameter()
        self.P0 = self.opti.parameter()
        self.Q0 = self.opti.parameter()

        self.opti.minimize((self.V_re - self.v0r)**2 + (self.V_im - self.v0i)**2 + (self.I_re - self.i0r)**2 + (self.I_im - self.i0i)**2 + (self.P - self.P0)**2 + (self.Q - self.Q0)**2)

        self.opti.subject_to(self.V_re * self.I_re + self.V_im * self.I_im == self.P)
        self.opti.subject_to(self.V_im * self.I_re - self.V_re * self.I_im == self.Q)
        self.opti.subject_to(self.V_re**2 + self.V_im**2 <= V_max)
        self.opti.subject_to(self.V_re**2 + self.V_im**2 >= V_min)

        opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        self.opti.solver('ipopt', opts)

    def prox(self, z: Trajectory, rho: float = 1.0) -> Trajectory:
        ret = z.copy()
        V0 = ret["v"][0, 0]
        I0 = ret["i"][0, 0]
        S = ret["s"][0, 0]

        self.opti.set_value(self.v0r, np.real(V0))
        self.opti.set_value(self.v0i, np.imag(V0))
        self.opti.set_value(self.i0r, np.real(I0))
        self.opti.set_value(self.i0i, np.imag(I0))
        self.opti.set_value(self.P0, np.real(S))
        self.opti.set_value(self.Q0, np.imag(S))

        sol = self.opti.solve()

        ret["v"][0, 0] = sol.value(self.V_re) + 1j * sol.value(self.V_im)
        ret["i"][0, 0] = sol.value(self.I_re) + 1j * sol.value(self.I_im)
        ret["s"][0, 0] = sol.value(self.P) + 1j * sol.value(self.Q)

        return ret


class BusBehaviours(Proxable):
    """
    A class implementing projections onto the behaviours of a set of buses, each with its own behaviour (e.g., constant power load, generator, etc.)
    """
    def __init__(self, behaviours: list[Proxable]):
        self.behaviours = behaviours

    def prox(self, z: Trajectory, rho: float = 1.0) -> Trajectory:
        ret = z.copy()
        for i, behaviour in enumerate(self.behaviours):
            ret_i = behaviour.prox(ret.at_bus[i], rho)
            ret.at_bus[i] = ret_i
        return ret


class BusBehavioursParallel(Proxable):
    """
    Same as BusBehaviours, but evaluates each bus's prox in parallel using a
    thread pool. Each behaviour owns its own solver instance and only reads
    a single-bus slice of z, so the per-bus prox calls are independent and
    safe to run concurrently.
    """
    def __init__(self, behaviours: list[Proxable], max_workers: int | None = None):
        self.behaviours = behaviours
        self.max_workers = max_workers

    def prox(self, z: Trajectory, rho: float = 1.0) -> Trajectory:
        ret = z.copy()
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(
                lambda item: item[1].prox(z.at_bus[item[0]], rho),
                enumerate(self.behaviours),
            ))
        for i, ret_i in enumerate(results):
            ret.at_bus[i] = ret_i
        return ret


class F(Proxable):
    """
    A class implementing Prox_{rho, f}(z) = min_x f(x) + rho/2||x - z||^2 where:

    f(x) = f(V, I, S) = sum_{j=1}^{n_gens} c_j P_j^2 + i_{B_net}(V, I, S) + i_C(V, I, S)

    Where i_S(w) is the indicator function of the set S at w, and

    B_net = {V, I, S | I = Y_bus @ V}

    C = {V, I, S | P_{min, i} < P_i < P_{max, i}, Q_{min, i} < Q_i < Q_{max, i} for all i in n_gens}

    This reduces to the convex optimization problem:

    Prox_{rho, f}(V0, I0, S0) = min_{V, I, S} sum_{j=1}^{n_gens} c_j P_j^2 + rho/2(||V-V0||^2 + ||I-I0||^2 + ||S-S0||^2)

    subject to: I = Y_bus @ V

                P_{min, i} < P_i < P_{max, i}

                Q_{min, i} < Q_i < Q_{max, i}
    """

    def __init__(self, config: Config):
        self.config = config

        self.n_buses = config["n_buses"]
        self.n_gens = config["n_gens"]
        self.Y_bus = config["Y_bus"]
        self.costs = config["costs"]

        self.V = cp.Variable(self.n_buses, complex=True)
        self.I = cp.Variable(self.n_buses, complex=True)
        self.S = cp.Variable(self.n_buses, complex=True)

        self.V0 = cp.Parameter(self.n_buses, complex=True)
        self.I0 = cp.Parameter(self.n_buses, complex=True)
        self.S0 = cp.Parameter(self.n_buses, complex=True)

        self._rho = None
        self.problem = None

    def _build_problem(self, rho: float) -> None:
        """
        rho enters the objective as a plain constant (not a cp.Parameter)
        """
        P = cp.real(self.S[:self.n_gens])
        Q = cp.imag(self.S[:self.n_gens])

        objective = cp.sum(cp.multiply(self.costs, cp.square(P)))
        objective += (rho / 2) * (
            cp.sum_squares(cp.abs(self.V - self.V0))
            + cp.sum_squares(cp.abs(self.I - self.I0))
            + cp.sum_squares(cp.abs(self.S - self.S0))
        )

        constraints = [
            self.I == self.Y_bus @ self.V,
            P >= self.config["P_min"],
            P <= self.config["P_max"],
            Q >= self.config["Q_min"],
            Q <= self.config["Q_max"],
        ]

        self.problem = cp.Problem(cp.Minimize(objective), constraints)
        self._rho = rho

    def prox(self, z: Trajectory, rho: float = 1.0) -> Trajectory:
        if self.problem is None or rho != self._rho:
            self._build_problem(rho)
        problem = cast(cp.Problem, self.problem)

        self.V0.value = z["v"][0]
        self.I0.value = z["i"][0]
        self.S0.value = z["s"][0]

        problem.solve()

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"Optimization did not converge: {problem.status}")

        if self.V.value is None or self.I.value is None or self.S.value is None:
            raise ValueError("Optimization did not converge: one of the variables is None.")

        return Trajectory({
            "v": self.V.value[np.newaxis, :],
            "i": self.I.value[np.newaxis, :],
            "s": self.S.value[np.newaxis, :],
        })


def rho_heuristic(iteration, rho_prev, r, s, tau=2, mu=10):
    if rho_prev == 0:
        return 2.0
    elif np.linalg.norm(r) > mu * np.linalg.norm(s):
        return tau * rho_prev
    elif np.linalg.norm(s) > mu * np.linalg.norm(r):
        return rho_prev / tau
    else:
        return rho_prev


def make_bus_behaviours(config: Config, parallel: bool = False) -> Proxable:
    n_buses = config["n_buses"]
    n_gens = config["n_gens"]
    behaviours = [
        *(Generator(config, bus_index=i, gen_index=i) for i in range(n_gens)),
        *(ConstPowerLoad(config, bus_index=i, load_index=i - n_gens) for i in range(n_gens, n_buses)),
    ]
    if parallel:
        return BusBehavioursParallel(behaviours)
    return BusBehaviours(behaviours)


def check_solution(z: Trajectory, config: Config) -> Dict[str, float]:
    """
    Evaluates a candidate solution z = (V, I, S) against the OPF problem
    described by config, returning the objective value along with the
    residual of each constraint (0 means satisfied).
    """
    n_gens = config["n_gens"]
    Y_bus = config["Y_bus"]
    costs = config["costs"]
    V_max = config["V_max"]
    V_min = config["V_min"]
    P_max = config["P_max"]
    P_min = config["P_min"]
    Q_max = config["Q_max"]
    Q_min = config["Q_min"]

    V = z["v"][0]
    I = z["i"][0]
    S = z["s"][0]

    P = np.real(S)
    Q = np.imag(S)

    objective = float(np.sum(costs * P[:n_gens] ** 2))

    # I = Y_bus @ V
    network_residual = float(np.linalg.norm(I - Y_bus @ V))

    # S = V * conj(I)
    power_balance_residual = float(np.linalg.norm(S - V * np.conj(I)))

    # V_min <= |V|^2 <= V_max
    V_sq = np.abs(V) ** 2
    voltage_violation = np.maximum(0, V_min - V_sq) + np.maximum(0, V_sq - V_max)
    voltage_residual = float(np.linalg.norm(voltage_violation))

    # P_min <= P <= P_max, Q_min <= Q <= Q_max (generators only)
    P_gen, Q_gen = P[:n_gens], Q[:n_gens]
    P_violation = np.maximum(0, P_min - P_gen) + np.maximum(0, P_gen - P_max)
    Q_violation = np.maximum(0, Q_min - Q_gen) + np.maximum(0, Q_gen - Q_max)
    generation_residual = float(np.linalg.norm(np.concatenate([P_violation, Q_violation])))

    return {
        "objective": objective,
        "network_residual": network_residual,
        "power_balance_residual": power_balance_residual,
        "voltage_residual": voltage_residual,
        "generation_residual": generation_residual,
    }


# ---------------------------------------------------------------------------
# Generator DAE model (classical model: constant |E| behind transient reactance)
#
# Dynamic state   x = [delta, omega]      (rotor angle [rad], absolute speed [rad/s])
# Input           u = [P_0, P_e]          (mechanical power, electrical power)
#
# Continuous time:  x_dot = A x + B_u u + c
#   delta_dot = omega - omega_s
#   omega_dot = (P_0 - P_e - D (omega - omega_s)) / M
#
# Algebraic (network interface, I = injection into the network):
#   E e^{j delta} = V + j Xd I
#
# The differential equations are linear in (x, u), so they are discretized
# exactly (zero-order hold on u over each step of length dt):
#   x_{n+1} = A_d x_n + B_d u_n + c_d
# Every helper below builds scalar expressions with +/* only (and ca.cos/ca.sin
# for the algebraic part), so they work with floats and casadi symbols alike.
# ---------------------------------------------------------------------------


def gen_continuous_matrices(config: Config, gen_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Continuous-time swing dynamics of generator gen_idx in the form
        x_dot = A x + B_u u + c,   x = [delta, omega],  u = [P_0, P_e]
    Returns numeric (A (2x2), B_u (2x2), c (2,)).
    """
    D = float(config.D[gen_idx])
    M = float(config.M[gen_idx])
    omega_s = float(config.omega_s)
    A = np.array([[0.0, 1.0], [0.0, -D / M]])
    B_u = np.array([[0.0, 0.0], [1.0 / M, -1.0 / M]])
    c = np.array([-omega_s, D * omega_s / M])
    return A, B_u, c


def gen_discrete_dynamics(config: Config, gen_idx: int, dt: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Exact zero-order-hold discretization of gen_continuous_matrices with step dt
    (defaults to config.dt):
        x_{n+1} = A_d x_n + B_d u_n + c_d
    with A_d = expm(A dt), Phi = int_0^dt expm(A s) ds, B_d = Phi B_u, c_d = Phi c.
    A_d and Phi are read off a single matrix exponential of the augmented
    matrix [[A dt, I dt], [0, 0]].
    """
    if dt is None:
        dt = config.dt
    A, B_u, c = gen_continuous_matrices(config, gen_idx)
    aug = np.zeros((4, 4))
    aug[:2, :2] = A * dt
    aug[:2, 2:] = np.eye(2) * dt
    aug_d = expm(aug)
    A_d = aug_d[:2, :2]
    Phi = aug_d[:2, 2:]
    return A_d, Phi @ B_u, Phi @ c


def gen_diff_eqs(
        config: Config,
        gen_idx: int,
        V_re: Scalar,
        V_im: Scalar,
        I_re: Scalar,
        I_im: Scalar,
        S_re: Scalar,
        S_im: Scalar,
        delta: Scalar,
        omega: Scalar,
        P_0: Scalar
    ) -> Dict[str, Scalar]:
    """
    Computes the differential equations for a generator at a given index.
    Returns the derivatives of the dynamic states (delta, omega) i.e. computes
    x_dot = f(x, y, u)
    """
    A, B_u, c = gen_continuous_matrices(config, gen_idx)
    A, B_u, c = A.tolist(), B_u.tolist(), c.tolist()

    return {
        "delta_dot": A[0][0] * delta + A[0][1] * omega + B_u[0][0] * P_0 + B_u[0][1] * S_re + c[0],
        "omega_dot": A[1][0] * delta + A[1][1] * omega + B_u[1][0] * P_0 + B_u[1][1] * S_re + c[1],
    }


def gen_discrete_step(
        config: Config,
        gen_idx: int,
        delta: Scalar,
        omega: Scalar,
        S_re: Scalar,
        P_0: Scalar,
        dt: float | None = None
    ) -> Dict[str, Scalar]:
    """
    One exact (zero-order hold) step of the generator swing dynamics:
        [delta_next, omega_next] = A_d [delta, omega] + B_d [P_0, S_re] + c_d
    where S_re is the electrical power held over the step and P_0 the
    mechanical power. Works with floats and casadi symbols.
    """
    A_d, B_d, c_d = gen_discrete_dynamics(config, gen_idx, dt)
    A_d, B_d, c_d = A_d.tolist(), B_d.tolist(), c_d.tolist()

    return {
        "delta_next": A_d[0][0] * delta + A_d[0][1] * omega + B_d[0][0] * P_0 + B_d[0][1] * S_re + c_d[0],
        "omega_next": A_d[1][0] * delta + A_d[1][1] * omega + B_d[1][0] * P_0 + B_d[1][1] * S_re + c_d[1],
    }


def gen_alg_eqs(
        config: Config,
        gen_idx: int,
        V_re: Scalar,
        V_im: Scalar,
        I_re: Scalar,
        I_im: Scalar,
        S_re: Scalar,
        S_im: Scalar,
        delta: Scalar,
        omega: Scalar,
        E: Scalar,
        P_0: Scalar
    ) -> Dict[str, Scalar]:
    """
    Computes the algebraic equations for a generator at a given index.
    Returns the residuals of the algebraic equations (power balance, etc.) i.e. computes
    r = g(x, y, u)

    E is the (constant) magnitude of the voltage behind the transient reactance.
    """
    Xd = float(config.Xd[gen_idx])
    # Rotor equation: E e^{j delta} = V + j Xd I
    r_re = E * ca.cos(delta) - V_re + Xd * I_im
    r_im = E * ca.sin(delta) - V_im - Xd * I_re

    return {
        "r_re": r_re,
        "r_im": r_im,
    }


def gen_coi_angle(config: Config, delta):
    """
    Centre-of-inertia rotor angle delta_coi = sum_i M_i delta_i / sum_i M_i.
    delta may be a numpy vector or a casadi column of length n_gens.
    """
    M = np.asarray(config.M, dtype=float)
    M_total = float(M.sum())
    return sum(float(M[i]) * delta[i] for i in range(len(M))) / M_total