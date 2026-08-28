import numpy as np
import casadi as ca
import cvxpy as cp
import matplotlib.pyplot as plt
from typing import Callable, Dict, Union, overload, cast
from abc import ABC, abstractmethod\

class ZDict:
    """
    A dict-like container mapping str -> np.ndarray that supports elementwise
    arithmetic (+, -, *, unary -) and a norm()
    """

    def __init__(self, data: Dict[str, np.ndarray]):
        self.data = dict(data)

    @overload
    def __getitem__(self, key: str) -> np.ndarray: ...
    @overload
    def __getitem__(self, key: Union[int, slice]) -> "ZDict": ...

    def __getitem__(self, key):
        """
        Indexing by str returns the underlying array for that key. Indexing
        by anything else (int, slice, ...) applies the same index to every
        value and returns a ZDict of the results.
        """
        if isinstance(key, str):
            return self.data[key]
        return ZDict({k: v[key] for k, v in self.data.items()})

    @overload
    def __setitem__(self, key: str, value: np.ndarray) -> None: ...
    @overload
    def __setitem__(self, key: Union[int, slice], value: "ZDict") -> None: ...

    def __setitem__(self, key, value) -> None:
        """
        Setting by str replaces the array for that key. Setting by anything
        else assigns element-wise from another ZDict with matching keys.
        """
        if isinstance(key, str):
            self.data[key] = value
        else:
            for k in self.data.keys():
                self.data[k][key] = value[k]

    def keys(self):
        return self.data.keys()

    def items(self):
        return self.data.items()

    def values(self):
        return self.data.values()

    def copy(self) -> "ZDict":
        return ZDict({k: v.copy() for k, v in self.data.items()})

    def zeroslike(self) -> "ZDict":
        return ZDict({k: np.zeros_like(v) for k, v in self.data.items()})

    def __add__(self, other: "ZDict") -> "ZDict":
        return ZDict({k: self.data[k] + other.data[k] for k in self.data.keys()})

    def __sub__(self, other: "ZDict") -> "ZDict":
        return ZDict({k: self.data[k] - other.data[k] for k in self.data.keys()})

    def __mul__(self, a: float) -> "ZDict":
        return ZDict({k: a * v for k, v in self.data.items()})

    def __rmul__(self, a: float) -> "ZDict":
        return self.__mul__(a)

    def __neg__(self) -> "ZDict":
        return self.__mul__(-1)

    def norm(self) -> float:
        return float(np.linalg.norm(np.concatenate([v.ravel() for v in self.data.values()])))

class Proxable(ABC):
    """
    Abstract class implementing the solution to
    Prox_{rho, f}(z) = min_x f(x) + rho/2||x - z||^2
    """
    @abstractmethod
    def prox(self, z: ZDict, rho: float) -> ZDict:
        pass

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


    def prox(self, z: ZDict, rho: float=1.0) -> ZDict:
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


    def prox(self, z: ZDict, rho: float=1.0) -> ZDict:
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

    def prox(self, z: ZDict, rho: float=1.0) -> ZDict:
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

def admm(f: Proxable,
         g: Proxable,
         z0: ZDict,
         rho=lambda i, prev, r, s: 2.0,
         threshold=1e-3,
         max_iterations=1000,
         callback=None):
    """
    Minimizes a constrained optimization problem using the Alternating Direction Method of Multipliers (ADMM).

    Parameters
    ----------
    f : Proxable
        The (possibly constrained) objective function to be minimized.
    g : Proxable
        The projection operator representing the constraints.
    z0 : ZDict
        The initial guess for the solution.
    callback : callable, optional
        Called as callback(iteration, x, z, u) at the end of each iteration.
    """

    # Initialize x0, z0, mu0
    zs: list[ZDict] = [z0.copy()]
    xs: list[ZDict] = [z0.zeroslike()]
    us: list[ZDict] = [z0.zeroslike()]

    rs = [(xs[-1] - zs[-1]).norm()]
    ss = [(zs[-1] - zs[-1]).norm()]  # Initialize ss with zero since there's no previous z
    rhos = [2.0]

    for iteration in range(max_iterations-1):
        new_rho = rho(iteration, rhos[-1], rs[-1], ss[-1])
        rhos.append(new_rho)
        # Rescale u when rho changes to keep λ = rho*u continuous
        if new_rho != rhos[-2]:
            us[-1] = us[-1] * (rhos[-2] / new_rho)

        xs.append(f.prox(zs[-1] - us[-1], rhos[-1]))
        zs.append(g.prox(xs[-1] + us[-1], rhos[-1]))
        us.append(us[-1] + (xs[-1] - zs[-1]))

        rs.append((xs[-1] - zs[-1]).norm())
        ss.append(rhos[-1] * (zs[-1] - zs[-2]).norm())

        if callback is not None:
            callback(iteration, xs[-1], zs[-1], us[-1], rs[-1], ss[-1])

        if rs[-1] < threshold and ss[-1] < threshold:
            break

    return xs, zs, us, rs, ss, rhos

def make_config():
    config = {}
    n_buses = 3
    n_gens = 2
    
    Y_bus = np.array([[ 4.902-19.608j, -2.941+11.765j, -1.961 +7.843j],
                      [-2.941+11.765j,  5.294-21.176j, -2.353 +9.412j],
                      [-1.961 +7.843j, -2.353 +9.412j,  4.314-17.255j]])

    config["n_buses"] = n_buses
    config["n_gens"] = n_gens
    config["Y_bus"] = Y_bus
    config["G"] = np.real(Y_bus)
    config["B"] = np.imag(Y_bus)
    config["costs"] = np.array([1.0, 1.2])
    config["load_P"] = np.array([1.0])
    config["load_Q"] = np.array([0.5])
    config["V_max"] = np.array([1.1, 1.1, 1.1])
    config["V_min"] = np.array([0.9, 0.9, 0.9])
    config["P_max"] = np.array([1.2, 1.2])
    config["P_min"] = np.array([0.2, 0.2])
    config["Q_max"] = np.array([1.3, 1.3])
    config["Q_min"] = np.array([-0.2, -0.2])

    return config

def solve_opf_centralized(config: Dict):
    G = config["G"]
    B = config["B"]
    n_buses = config["n_buses"]
    n_gens = config["n_gens"]
    costs = config["costs"]
    load_P = config["load_P"]
    load_Q = config["load_Q"]
    V_max = config["V_max"]
    V_min = config["V_min"]
    P_max = config["P_max"]
    P_min = config["P_min"]
    Q_max = config["Q_max"]
    Q_min = config["Q_min"]

    opti = ca.Opti()

    V_re = opti.variable(n_buses)
    V_im = opti.variable(n_buses)
    I_re = opti.variable(n_buses)
    I_im = opti.variable(n_buses)
    P = opti.variable(n_buses)
    Q = opti.variable(n_buses)

    opti.minimize(sum(costs[i] * P[i]**2 for i in range(n_gens)))

    # Power flow constraints
    opti.subject_to(I_re == G @ V_re - B @ V_im)
    opti.subject_to(I_im == B @ V_re + G @ V_im)

    for i in range(n_buses):
        opti.subject_to(P[i] == V_re[i] * I_re[i] + V_im[i] * I_im[i])
        opti.subject_to(Q[i] == V_im[i] * I_re[i] - V_re[i] * I_im[i])
    
    for i in range(n_gens, n_buses):
        opti.subject_to(P[i] == -load_P[i - n_gens])
        opti.subject_to(Q[i] == -load_Q[i - n_gens])
    
    # Reference (slack) bus angle
    opti.subject_to(V_im[0] == 0)

    # Operational constraints
    for i in range(n_buses):
        V_sq = V_re[i]**2 + V_im[i]**2
        opti.subject_to(V_sq >= V_min[i])
        opti.subject_to(V_sq <= V_max[i])
    
    for i in range(n_gens):
        opti.subject_to(P[i] >= P_min[i])
        opti.subject_to(P[i] <= P_max[i])
        opti.subject_to(Q[i] >= Q_min[i])
        opti.subject_to(Q[i] <= Q_max[i])

    opti.set_initial(V_re, np.ones(n_buses))
    opti.set_initial(V_im, np.zeros(n_buses))

    opti.solver('ipopt')

    sol = opti.solve()

    print("V_re:", sol.value(V_re))
    print("V_im:", sol.value(V_im))
    print("I_re:", sol.value(I_re))
    print("I_im:", sol.value(I_im))
    print("P:", sol.value(P))
    print("Q:", sol.value(Q))

    return sol

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

def rho_heuristic(iteration, rho_prev, r, s, tau=2, mu=10):
	if rho_prev == 0:
		return 2.0
	elif np.linalg.norm(r) > mu*np.linalg.norm(s):
		print(f"Warning: Norm of r at iteration {iteration} is {np.linalg.norm(r)}, multiplying rho by {tau} \n\n\n")
		return tau * rho_prev
	elif np.linalg.norm(s) > mu*np.linalg.norm(r):
		print(f"Warning: Norm of s at iteration {iteration} is {np.linalg.norm(s)}, dividing rho by {tau} \n\n\n")
		return rho_prev / tau
	else:
		return rho_prev

def solve_opf_admm(config: Dict) -> ZDict:
    n_buses = config["n_buses"]
    n_gens = config["n_gens"]

    f = F(config)
    g = BusBehaviours([
        *(Generator(config, bus_index=i, gen_index=i) for i in range(n_gens)),
        *(ConstPowerLoad(config, bus_index=i, load_index=i - n_gens) for i in range(n_gens, n_buses)),
    ])

    z0 = ZDict({
        "v": np.ones(n_buses, dtype=complex),
        "i": np.zeros(n_buses, dtype=complex),
        "s": np.zeros(n_buses, dtype=complex),
    })

    def callback(iteration, x, z, u, r, s):
        if iteration % 10 == 0:
            print(f"iter {iteration:4d}: r={r:.6f}, s={s:.6f}")

    xs, zs, us, rs, ss, rhos = admm(f, g, z0, rho=rho_heuristic, threshold=5e-5, max_iterations=10000, callback=callback)

    print("V:", xs[-1]["v"])
    print("I:", xs[-1]["i"])
    print("S:", xs[-1]["s"])

    plt.plot(rs, label="r (primal residual)")
    plt.plot(ss, label="s (dual residual)")
    plt.xlabel("iteration")
    plt.ylabel("residual")
    plt.yscale("log")
    plt.legend()
    plt.title("ADMM residuals")
    plt.show()

    return xs[-1]

if __name__ == "__main__":
    config = make_config()
    # sol = solve_opf_centralized(config)

    sol = solve_opf_admm(config)
    checks = check_solution(sol, config)
    print(checks)