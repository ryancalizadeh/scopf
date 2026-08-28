import numpy as np
import casadi as ca
import cvxpy as cp
from typing import Dict, cast
from ZDict import ZDict
from Proxable import Proxable


class ConstPowerLoad(Proxable):
    """
    A class implementing projections onto the behaviour of a constant power load
    """

    def __init__(self, config: Dict, bus_index: int, load_index: int, max_iter=20, tol=1e-5):
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

    def prox(self, z: ZDict, rho: float = 1.0) -> ZDict:
        ret = z.copy()
        V0 = ret["v"][0]
        I0 = ret["i"][0]

        self.opti.set_value(self.v0r, np.real(V0))
        self.opti.set_value(self.v0i, np.imag(V0))
        self.opti.set_value(self.i0r, np.real(I0))
        self.opti.set_value(self.i0i, np.imag(I0))

        sol = self.opti.solve()

        ret["v"][0] = sol.value(self.V_re) + 1j * sol.value(self.V_im)
        ret["i"][0] = sol.value(self.I_re) + 1j * sol.value(self.I_im)
        ret["s"][0] = self.S

        return ret


class Generator(Proxable):
    """
    A class implementing projections onto the behaviour of a flexible generator
    """

    def __init__(self, config: Dict, bus_index: int, gen_index: int):
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

    def prox(self, z: ZDict, rho: float = 1.0) -> ZDict:
        ret = z.copy()
        V0 = ret["v"][0]
        I0 = ret["i"][0]
        S = ret["s"][0]

        self.opti.set_value(self.v0r, np.real(V0))
        self.opti.set_value(self.v0i, np.imag(V0))
        self.opti.set_value(self.i0r, np.real(I0))
        self.opti.set_value(self.i0i, np.imag(I0))
        self.opti.set_value(self.P0, np.real(S))
        self.opti.set_value(self.Q0, np.imag(S))

        sol = self.opti.solve()

        ret["v"][0] = sol.value(self.V_re) + 1j * sol.value(self.V_im)
        ret["i"][0] = sol.value(self.I_re) + 1j * sol.value(self.I_im)
        ret["s"][0] = sol.value(self.P) + 1j * sol.value(self.Q)

        return ret


class BusBehaviours(Proxable):
    """
    A class implementing projections onto the behaviours of a set of buses, each with its own behaviour (e.g., constant power load, generator, etc.)
    """
    def __init__(self, behaviours: list[Proxable]):
        self.behaviours = behaviours

    def prox(self, z: ZDict, rho: float = 1.0) -> ZDict:
        ret = z.copy()
        for i, behaviour in enumerate(self.behaviours):
            ret_i = behaviour.prox(ret[i:i+1], rho)
            ret[i] = ret_i[0]
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

    def __init__(self, config):
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

    def prox(self, z: ZDict, rho: float = 1.0) -> ZDict:
        if self.problem is None or rho != self._rho:
            self._build_problem(rho)
        problem = cast(cp.Problem, self.problem)

        self.V0.value = z["v"]
        self.I0.value = z["i"]
        self.S0.value = z["s"]

        problem.solve()

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise ValueError(f"Optimization did not converge: {problem.status}")

        if self.V.value is None or self.I.value is None or self.S.value is None:
            raise ValueError("Optimization did not converge: one of the variables is None.")

        return ZDict({"v": self.V.value, "i": self.I.value, "s": self.S.value})


def rho_heuristic(iteration, rho_prev, r, s, tau=2, mu=10):
    if rho_prev == 0:
        return 2.0
    elif np.linalg.norm(r) > mu * np.linalg.norm(s):
        return tau * rho_prev
    elif np.linalg.norm(s) > mu * np.linalg.norm(r):
        return rho_prev / tau
    else:
        return rho_prev


def make_bus_behaviours(config: Dict) -> BusBehaviours:
    n_buses = config["n_buses"]
    n_gens = config["n_gens"]
    return BusBehaviours([
        *(Generator(config, bus_index=i, gen_index=i) for i in range(n_gens)),
        *(ConstPowerLoad(config, bus_index=i, load_index=i - n_gens) for i in range(n_gens, n_buses)),
    ])


def check_solution(z: ZDict, config: Dict) -> Dict[str, float]:
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

    V = z["v"]
    I = z["i"]
    S = z["s"]

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
