import os
import queue
import threading
import numpy as np
import casadi as ca
import cvxpy as cp
from scipy.linalg import expm
from typing import Any, Dict, cast
from Config import Config

# The generator DAE helpers accept plain numbers as well as casadi symbols
# (MX / SX / DM); the arithmetic inside them is +, *, ca.cos and ca.sin only.
Scalar = Any
from Trajectory import Trajectory
from Proxable import Proxable


# Transient stability limit on the rotor angle relative to the centre of inertia
DELTA_COI_MAX = np.deg2rad(100.0)

_IPOPT_QUIET = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}


def _project_voltage_magnitude(v: complex, V_min: float, V_max: float) -> complex:
    """
    Projects the phasor v onto the annulus V_min <= |v| <= V_max, keeping its
    angle. A zero phasor is mapped to V_min at angle 0.
    """
    mag = abs(v)
    if mag == 0.0:
        return complex(V_min, 0.0)
    return v * (min(max(mag, V_min), V_max) / mag)


def _load_scale(config: Config) -> np.ndarray:
    """
    Per-time-step load multiplier: 1 at n = 0 (pre-disturbance steady state),
    config.load_step_factor for n >= 1 (the load-step disturbance), matching
    algorithms.centralized.
    """
    scale = np.ones(config.N)
    scale[1:] = config.load_step_factor
    return scale


def _solve_or_last_iterate(opti: ca.Opti):
    """
    Solves an Opti problem; if IPOPT stops without converging (e.g. max_iter),
    falls back to the last iterate instead of aborting the whole ADMM run.
    Returns (value_getter, converged).
    """
    try:
        sol = opti.solve()
        return sol.value, True
    except RuntimeError:
        return opti.debug.value, False


class ConstPowerLoad(Proxable):
    """
    A class implementing the prox operator on the indicator function of the behaviour set of a constant power load (i.e. projection onto the behaviour set)
    Specifically, the load is modelled as a constant complex power taken from config.load_P and config.load_Q, scaled by config.load_step_factor for t >= 1 (the load-step disturbance used by algorithms.centralized).
    The behaviour of the load is defined as:
    B = {V(t), I(t), S(t) | S(t) = V(t) * conj(I(t)), S(t) = -scale(t) * S_load for all t}
    and V_bb = {V(t) | V_min^2 <= |V(0)|^2 <= V_max^2}
    and f = ind_B + ind_V_bb, so prox_{rho, f} = proj_(B cap V_bb)
    The projection is computed using casadi and IPOPT, and is set up once. Previous solutions are used to warmstart the solver for the next iteration.
    Since this prox reduces to a pure projection, rho is ignored.
    S(t) is fixed, so the projection only has to find (V(t), I(t)) closest to (V0(t), I0(t)) on the hyperbola V(t) conj(I(t)) = S(t); it is separable over t and solved as one block-diagonal NLP.
    """

    def __init__(self, config: Config, bus_index: int, load_index: int, max_iter=100, tol=1e-8):
        self.config = config
        self.bus_index = bus_index
        self.load_index = load_index
        N = config.N
        self.N = N

        V_min = float(config.V_min[bus_index])
        V_max = float(config.V_max[bus_index])

        # A load draws power from the network, i.e. it injects the negative of
        # its consumption.
        scale = _load_scale(config)
        self.P_load = -scale * float(config.load_P[load_index])
        self.Q_load = -scale * float(config.load_Q[load_index])
        self.S_load = self.P_load + 1j * self.Q_load

        opti = ca.Opti()
        self.V_re = opti.variable(N)
        self.V_im = opti.variable(N)
        self.I_re = opti.variable(N)
        self.I_im = opti.variable(N)

        self.v0r = opti.parameter(N)
        self.v0i = opti.parameter(N)
        self.i0r = opti.parameter(N)
        self.i0i = opti.parameter(N)

        opti.minimize(
            ca.sumsqr(self.V_re - self.v0r) + ca.sumsqr(self.V_im - self.v0i)
            + ca.sumsqr(self.I_re - self.i0r) + ca.sumsqr(self.I_im - self.i0i)
        )

        for n in range(N):
            opti.subject_to(self.V_re[n] * self.I_re[n] + self.V_im[n] * self.I_im[n] == self.P_load[n])
            opti.subject_to(self.V_im[n] * self.I_re[n] - self.V_re[n] * self.I_im[n] == self.Q_load[n])

        V_sq0 = self.V_re[0]**2 + self.V_im[0]**2
        opti.subject_to(V_sq0 >= V_min**2)
        opti.subject_to(V_sq0 <= V_max**2)

        opti.solver('ipopt', {**_IPOPT_QUIET, 'ipopt.max_iter': int(max_iter), 'ipopt.tol': float(tol)})
        self.opti = opti
        self._warm = None
        self.n_failures = 0

    def prox(self, z: Trajectory, rho: float = 1.0) -> Trajectory:
        V0 = z["v"][:, 0]
        I0 = z["i"][:, 0]

        self.opti.set_value(self.v0r, np.real(V0))
        self.opti.set_value(self.v0i, np.imag(V0))
        self.opti.set_value(self.i0r, np.real(I0))
        self.opti.set_value(self.i0i, np.imag(I0))

        warm = self._warm if self._warm is not None else (np.real(V0), np.imag(V0), np.real(I0), np.imag(I0))
        for var, val in zip((self.V_re, self.V_im, self.I_re, self.I_im), warm):
            self.opti.set_initial(var, val)

        value, converged = _solve_or_last_iterate(self.opti)
        if not converged:
            self.n_failures += 1

        V_re, V_im, I_re, I_im = (np.reshape(value(var), self.N) for var in (self.V_re, self.V_im, self.I_re, self.I_im))
        self._warm = (V_re, V_im, I_re, I_im)

        ret = z.copy()
        ret["v"][:, 0] = V_re + 1j * V_im
        ret["i"][:, 0] = I_re + 1j * I_im
        ret["s"][:, 0] = self.S_load
        return ret


class Generator(Proxable):
    """
    The behaviour of the generator is defined as:
    B = {V(t), I(t), S(t)=P(t)+jQ(t), delta(t), omega(t) |
        S(t) = V(t) * conj(I(t)),
        delta_dot(t) = omega(t) - omega_s,
        omega_dot(t) = (P(0) - P(t) - D (omega(t) - omega_s)) / M,
        E e^{j delta(t)} = V(t) + j Xd I(t),
        omega(0) = omega_s,
        }
    The DAE model is discretized using exact zero-order hold (ZOH) discretization (gen_discrete_step),
    the mechanical power is the pre-disturbance dispatch P(0), and E >= 0 is a constant internal to the
    projection (it is not part of the consensus trajectory; the last value is kept in self.E).
    V_bb = {V(t) | V_min^2 <= |V(0)|^2 <= V_max^2} (config.V_min / V_max are magnitudes)
    and f = ind_B + ind_V_bb, so prox_{rho, f} = proj_(B cap V_bb)
    The projection operator is computed using casadi and IPOPT, and is set up once. Previous solutions are used to warmstart the solver for the next iteration.
    Since this prox reduces to a pure projection, rho is ignored.
    """

    def __init__(self, config: Config, bus_index: int, gen_index: int, max_iter=500, tol=1e-8):
        self.config = config
        self.bus_index = bus_index
        self.gen_index = gen_index
        N = config.N
        self.N = N
        self.omega_s = float(config.omega_s)

        V_min = float(config.V_min[bus_index])
        V_max = float(config.V_max[bus_index])

        opti = ca.Opti()
        self.V_re = opti.variable(N)
        self.V_im = opti.variable(N)
        self.I_re = opti.variable(N)
        self.I_im = opti.variable(N)
        self.P = opti.variable(N)
        self.Q = opti.variable(N)
        self.delta = opti.variable(N)
        self.omega = opti.variable(N)
        self.E_var = opti.variable()

        self.v0r = opti.parameter(N)
        self.v0i = opti.parameter(N)
        self.i0r = opti.parameter(N)
        self.i0i = opti.parameter(N)
        self.p0 = opti.parameter(N)
        self.q0 = opti.parameter(N)
        self.d0 = opti.parameter(N)
        self.w0 = opti.parameter(N)

        opti.minimize(
            ca.sumsqr(self.V_re - self.v0r) + ca.sumsqr(self.V_im - self.v0i)
            + ca.sumsqr(self.I_re - self.i0r) + ca.sumsqr(self.I_im - self.i0i)
            + ca.sumsqr(self.P - self.p0) + ca.sumsqr(self.Q - self.q0)
            + ca.sumsqr(self.delta - self.d0) + ca.sumsqr(self.omega - self.w0)
        )

        for n in range(N):
            # Hyperbola constraints: S = V conj(I)
            opti.subject_to(self.P[n] == self.V_re[n] * self.I_re[n] + self.V_im[n] * self.I_im[n])
            opti.subject_to(self.Q[n] == self.V_im[n] * self.I_re[n] - self.V_re[n] * self.I_im[n])
            # Algebraic equations: E e^{j delta} = V + j Xd I
            r = gen_alg_eqs(
                config, gen_index,
                self.V_re[n], self.V_im[n], self.I_re[n], self.I_im[n],
                self.P[n], self.Q[n], self.delta[n], self.omega[n], self.E_var, self.P[0],
            )
            opti.subject_to(r["r_re"] == 0)
            opti.subject_to(r["r_im"] == 0)

        # Exact ZOH discretization of the swing equation, mechanical power = P(0)
        for n in range(N - 1):
            step = gen_discrete_step(config, gen_index, self.delta[n], self.omega[n], self.P[n], self.P[0])
            opti.subject_to(self.delta[n + 1] == step["delta_next"])
            opti.subject_to(self.omega[n + 1] == step["omega_next"])

        # Pre-disturbance steady state
        opti.subject_to(self.omega[0] == self.omega_s)
        opti.subject_to(self.E_var >= 0)

        V_sq0 = self.V_re[0]**2 + self.V_im[0]**2
        opti.subject_to(V_sq0 >= V_min**2)
        opti.subject_to(V_sq0 <= V_max**2)

        opti.solver('ipopt', {**_IPOPT_QUIET, 'ipopt.max_iter': int(max_iter), 'ipopt.tol': float(tol)})
        self.opti = opti
        self._vars = (self.V_re, self.V_im, self.I_re, self.I_im, self.P, self.Q, self.delta, self.omega)
        self._warm = None
        self.E = None
        self.n_failures = 0

    def prox(self, z: Trajectory, rho: float = 1.0) -> Trajectory:
        V0 = z["v"][:, 0]
        I0 = z["i"][:, 0]
        S0 = z["s"][:, 0]
        d0 = z["delta"][:, 0]
        w0 = z["omega"][:, 0]

        params = (self.v0r, self.v0i, self.i0r, self.i0i, self.p0, self.q0, self.d0, self.w0)
        values = (np.real(V0), np.imag(V0), np.real(I0), np.imag(I0), np.real(S0), np.imag(S0), d0, w0)
        for par, val in zip(params, values):
            self.opti.set_value(par, val)

        if self._warm is None:
            warm_vars, warm_E = values, 1.0
        else:
            warm_vars, warm_E = self._warm
        for var, val in zip(self._vars, warm_vars):
            self.opti.set_initial(var, val)
        self.opti.set_initial(self.E_var, warm_E)

        value, converged = _solve_or_last_iterate(self.opti)
        if not converged:
            self.n_failures += 1

        sol_vars = tuple(np.reshape(value(var), self.N) for var in self._vars)
        self.E = float(value(self.E_var))
        self._warm = (sol_vars, self.E)

        V_re, V_im, I_re, I_im, P, Q, delta, omega = sol_vars
        ret = z.copy()
        ret["v"][:, 0] = V_re + 1j * V_im
        ret["i"][:, 0] = I_re + 1j * I_im
        ret["s"][:, 0] = P + 1j * Q
        ret["delta"][:, 0] = delta
        ret["omega"][:, 0] = omega
        return ret


class EmptyBus(Proxable):
    """
    A class implementing projections onto the behaviour of an empty bus (no load, no generator)
    The behaviour of the empty bus is defined as:
    B = {V(t), I(t), S(t) | S(t) = V(t) * conj(I(t)), I(t) = 0 for all t}
    and V_bb = {V(t) | V_min^2 <= |V(0)|^2 <= V_max^2}
    f = ind_B + ind_V_bb, so prox_{rho, f} = proj_(B cap V_bb)
    This projection is straightforward. I(t) = 0 implies S(t) = 0 and V is unconstrained
    except for the magnitude bound at t = 0, thus prox(V0, I0, S0) = (proj_annulus(V0(0)), V0(t >= 1), 0, 0),
    and rho is ignored.
    """

    def __init__(self, config: Config, bus_index: int):
        self.config = config
        self.bus_index = bus_index
        self.V_min = float(config.V_min[bus_index])
        self.V_max = float(config.V_max[bus_index])

    def prox(self, z: Trajectory, rho: float = 1.0) -> Trajectory:
        ret = z.copy()
        ret["i"][:] = 0
        ret["s"][:] = 0
        ret["v"][0, 0] = _project_voltage_magnitude(complex(ret["v"][0, 0]), self.V_min, self.V_max)
        return ret


class F(Proxable):
    """
    f = c(w) + ind_Bnet(w) + ind_C(w)
    where w = (V, I, S=P+jQ, delta, omega),
    c(w) = sum_{j=1}^{n_gens} c_j P_j(0)^2
    B_net = {V, I, S | I(t) = Y_bus @ V(t) for all t, Im V_0(0) = 0 (slack angle reference)}
    C = {V, I, S, delta, omega |
        P_{min, i} <= P_i(0) <= P_{max, i},
        Q_{min, i} <= Q_i(0) <= Q_{max, i},
        |I_{kl}(0)| <= I_{kl}^{max} for all lines (k, l),
        |delta_i(t) - delta_coi(t)| <= 100 deg for all t,
    }
    omega is unconstrained by f, so its prox is the identity and it bypasses the solver.

    f is convex (quadratic cost, linear network / limit constraints, second-order-cone line
    limits) and its proximal operator is computed with cvxpy. The problem is built once in
    __init__ (_build_problem) in DPP form: rho is a nonnegative cp.Parameter and, to keep the
    objective parameter-affine, the prox term is expanded as
        rho/2 ||x - z||^2 = rho/2 ||x||^2 - <rho z, x> + const
    with rho*z passed in as parameters. cvxpy therefore canonicalises the problem once and every
    prox call is a parameter update plus a solve (CLARABEL by default).
    """

    _KEYS = ("v_re", "v_im", "i_re", "i_im", "p", "q")

    def __init__(self, config: Config, solver: str = "CLARABEL"):
        self.config = config
        self.solver = solver
        self.n_buses = config.n_buses
        self.n_gens = config.n_gens
        self.N = config.N

        shape = (self.n_buses, self.N)
        self.x = {k: cp.Variable(shape) for k in self._KEYS}
        self.delta = cp.Variable((self.n_gens, self.N))

        self.rho = cp.Parameter(nonneg=True)
        self.rz = {k: cp.Parameter(shape) for k in self._KEYS}
        self.rz_delta = cp.Parameter((self.n_gens, self.N))

        self._build_problem()

        # Canonicalise once now (outside any ADMM timing) with a dummy solve.
        self.rho.value = 1.0
        for k in self._KEYS:
            self.rz[k].value = np.zeros(shape)
        self.rz_delta.value = np.zeros((self.n_gens, self.N))
        self.problem.solve(solver=self.solver)

    def _build_problem(self) -> None:
        """
        Builds the DPP-compliant cvxpy problem once; rho and rho*z are parameters.
        """
        config = self.config
        n_buses, n_gens = self.n_buses, self.n_gens
        G, B = config.G, config.B
        V_re, V_im, I_re, I_im, P, Q = (self.x[k] for k in self._KEYS)
        delta = self.delta

        cost = sum(float(config.costs[i]) * cp.square(P[i, 0]) for i in range(n_gens))
        prox_term = (self.rho / 2) * (sum(cp.sum_squares(self.x[k]) for k in self._KEYS) + cp.sum_squares(delta))
        prox_term -= sum(cp.sum(cp.multiply(self.rz[k], self.x[k])) for k in self._KEYS)
        prox_term -= cp.sum(cp.multiply(self.rz_delta, delta))

        constraints = [
            # Network equations for all t
            I_re == G @ V_re - B @ V_im,
            I_im == B @ V_re + G @ V_im,
            # Slack angle reference for the pre-disturbance steady state
            V_im[0, 0] == 0,
            # Generation limits at t = 0
            P[:n_gens, 0] >= config.P_min,
            P[:n_gens, 0] <= config.P_max,
            Q[:n_gens, 0] >= config.Q_min,
            Q[:n_gens, 0] <= config.Q_max,
        ]

        # Line flow limits at t = 0 (second-order cones)
        for k in range(n_buses):
            for l in range(k + 1, n_buses):
                if G[k, l] == 0 and B[k, l] == 0:
                    continue  # no line between k and l
                I_re_kl = -(V_re[k, 0] - V_re[l, 0]) * G[k, l] + (V_im[k, 0] - V_im[l, 0]) * B[k, l]
                I_im_kl = -(V_re[k, 0] - V_re[l, 0]) * B[k, l] - (V_im[k, 0] - V_im[l, 0]) * G[k, l]
                constraints.append(cp.norm(cp.hstack([I_re_kl, I_im_kl])) <= config.line_flow_limits)

        # Transient stability constraint (identically zero with a single generator)
        if n_gens > 1:
            M = np.asarray(config.M, dtype=float)
            delta_coi = (M @ delta) / float(M.sum())  # shape (N,)
            for i in range(n_gens):
                constraints.append(delta[i, :] - delta_coi <= DELTA_COI_MAX)
                constraints.append(delta[i, :] - delta_coi >= -DELTA_COI_MAX)

        self.problem = cp.Problem(cp.Minimize(cost + prox_term), constraints)

    def prox(self, z: Trajectory, rho: float = 1.0) -> Trajectory:
        # Trajectory arrays are (N, width); the cvxpy variables are (width, N).
        V0 = z["v"].T
        I0 = z["i"].T
        S0 = z["s"].T
        z_values = {
            "v_re": np.real(V0), "v_im": np.imag(V0),
            "i_re": np.real(I0), "i_im": np.imag(I0),
            "p": np.real(S0), "q": np.imag(S0),
        }
        self.rho.value = rho
        for k in self._KEYS:
            self.rz[k].value = rho * z_values[k]
        self.rz_delta.value = rho * z["delta"].T

        self.problem.solve(solver=self.solver)

        if self.problem.status not in ("optimal", "optimal_inaccurate"):
            raise ValueError(f"F.prox did not converge: {self.problem.status}")

        x = {k: cast(np.ndarray, self.x[k].value) for k in self._KEYS}
        delta = cast(np.ndarray, self.delta.value)

        return Trajectory({
            "v": (x["v_re"] + 1j * x["v_im"]).T,
            "i": (x["i_re"] + 1j * x["i_im"]).T,
            "s": (x["p"] + 1j * x["q"]).T,
            "delta": delta.T,
            "omega": z["omega"].copy(),
        })


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


class _AffinityWorker(threading.Thread):
    """
    Persistent worker thread that owns a fixed subset of bus behaviours and
    evaluates their proxes for every trajectory submitted to it.
    """

    def __init__(self, behaviours: list[Proxable], indices: list[int]):
        super().__init__(daemon=True)
        self.behaviours = behaviours
        self.indices = indices
        self._tasks: "queue.Queue" = queue.Queue()
        self._results: "queue.Queue" = queue.Queue()

    def submit(self, z: Trajectory, rho: float) -> None:
        self._tasks.put((z, rho))

    def collect(self) -> list[tuple[int, Trajectory]]:
        result = self._results.get()
        if isinstance(result, BaseException):
            raise result
        return result

    def stop(self) -> None:
        self._tasks.put(None)

    def run(self) -> None:
        while True:
            task = self._tasks.get()
            if task is None:
                return
            z, rho = task
            try:
                self._results.put([(i, self.behaviours[i].prox(z.at_bus[i], rho)) for i in self.indices])
            except BaseException as exc:  # propagate to the caller instead of killing the thread
                self._results.put(exc)


class BusBehavioursParallel(Proxable):
    """
    Same as BusBehaviours, but evaluates the per-bus proxes concurrently on a
    pool of persistent worker threads. Each behaviour owns its own solver
    instance and only reads a single-bus slice of z, so the per-bus prox calls
    are independent.

    Every bus is pinned to one worker thread for the lifetime of this object
    (round-robin assignment). This matters: a casadi Opti/IPOPT problem that is
    solved from different OS threads over successive calls segfaults (observed
    with the Generator projections, even when the calls are serialised with a
    lock), while solving it concurrently from a fixed thread is fine. Pinning
    also keeps each behaviour's warm start on the thread that produced it.
    """
    def __init__(self, behaviours: list[Proxable], max_workers: int | None = None):
        self.behaviours = behaviours
        n_workers = max(1, min(len(behaviours), max_workers or (os.cpu_count() or 1)))
        self.max_workers = n_workers
        self._workers = [
            _AffinityWorker(behaviours, list(range(w, len(behaviours), n_workers)))
            for w in range(n_workers)
        ]
        for worker in self._workers:
            worker.start()

    def prox(self, z: Trajectory, rho: float = 1.0) -> Trajectory:
        ret = z.copy()
        for worker in self._workers:
            worker.submit(z, rho)
        for worker in self._workers:
            for i, ret_i in worker.collect():
                ret.at_bus[i] = ret_i
        return ret

    def close(self) -> None:
        for worker in self._workers:
            worker.stop()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


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
    n_loads = config["n_loads"]
    behaviours = [
        *(Generator(config, bus_index=i, gen_index=i) for i in range(n_gens)),
        *(ConstPowerLoad(config, bus_index=i, load_index=i - n_gens) for i in range(n_gens, n_gens + n_loads)),
        *(EmptyBus(config, bus_index=i) for i in range(n_gens + n_loads, n_buses))
    ]
    if parallel:
        return BusBehavioursParallel(behaviours)
    return BusBehaviours(behaviours)


def check_solution(z: Trajectory, config: Config) -> Dict[str, float]:
    """
    TODO Update this to support the transient stability problem as well. For now, it only checks the OPF problem.
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