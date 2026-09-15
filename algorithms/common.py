import os
import logging
import queue
import threading
import multiprocessing
import numpy as np
import casadi as ca
import cvxpy as cp
from scipy.linalg import expm, qr
from typing import Any, Dict, cast
from Config import Config

# The generator DAE helpers accept plain numbers as well as casadi symbols
# (MX / SX / DM); the arithmetic inside them is +, *, ca.cos and ca.sin only.
Scalar = Any
from Trajectory import Trajectory
from Proxable import Proxable
from algorithms.metric import LeverageMetric, inner

logger = logging.getLogger(__name__)


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
    A class implementing the prox operator on the indicator function of the behaviour set of a
    constant power load (i.e. projection onto the behaviour set).

    The load is constant-power only at t = 0 (the pre-disturbance steady state), where it draws
    the complex power in config.load_P / load_Q. For t >= 1 it is modelled as a constant
    impedance that Config folds into the transient admittance matrices (Y_fault / Y_post), so
    that the network stays solvable at the depressed voltages of a fault; a constant-power load
    cannot be served at |V| ~ 0.01.

    IMPORTANT: this does NOT mean the load stops drawing current for t >= 1. The term y_L * V
    inside Y @ V carries exactly the current the load device drew before, so the bus current is
    unchanged. What is zero for t >= 1 is only the *residual device injection*
    I(t) - Y(t) @ V(t) at that bus, which is what the device's own variables represent here.
    Consequently this prox reports I(t) = 0 and S(t) = 0 for t >= 1: those zeros mean "no
    injection on top of the network", not "no load". The true transient consumption of the bus
    is |V(t)|^2 * conj(y_L) with y_L from config.load_admittance().

    The behaviour set is therefore
    B = {V(t), I(t), S(t) | S(0) = V(0) conj(I(0)) = -S_load, I(t) = S(t) = 0 for t >= 1}
    and V_bb = {V(t) | V_min^2 <= |V(0)|^2 <= V_max^2}
    and f = ind_B + ind_V_bb, so prox_{rho, f} = proj_(B cap V_bb).

    Only the t = 0 block needs a solver (a projection onto the hyperbola V conj(I) = -S_load
    intersected with the voltage annulus); it is set up once with casadi/IPOPT and warm-started
    from the previous solution. Since this prox reduces to a pure projection, rho is ignored.
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
        # its consumption. Only the pre-disturbance step is constant power.
        self.P_load = -float(config.load_P[load_index])
        self.Q_load = -float(config.load_Q[load_index])
        self.S_load = self.P_load + 1j * self.Q_load
        # Constant-impedance representation used for t >= 1 (folded into Y by
        # Config); kept here so the true transient consumption is recoverable.
        self.y_load = float(config.load_P[load_index]) - 1j * float(config.load_Q[load_index])

        opti = ca.Opti()
        self.V_re = opti.variable()
        self.V_im = opti.variable()
        self.I_re = opti.variable()
        self.I_im = opti.variable()

        self.v0r = opti.parameter()
        self.v0i = opti.parameter()
        self.i0r = opti.parameter()
        self.i0i = opti.parameter()

        opti.minimize(
            (self.V_re - self.v0r)**2 + (self.V_im - self.v0i)**2
            + (self.I_re - self.i0r)**2 + (self.I_im - self.i0i)**2
        )

        opti.subject_to(self.V_re * self.I_re + self.V_im * self.I_im == self.P_load)
        opti.subject_to(self.V_im * self.I_re - self.V_re * self.I_im == self.Q_load)

        V_sq0 = self.V_re**2 + self.V_im**2
        opti.subject_to(V_sq0 >= V_min**2)
        opti.subject_to(V_sq0 <= V_max**2)

        opti.solver('ipopt', {**_IPOPT_QUIET, 'ipopt.max_iter': int(max_iter), 'ipopt.tol': float(tol)})
        self.opti = opti
        self._warm = None
        self.n_failures = 0

    def transient_power(self, V: np.ndarray) -> np.ndarray:
        """
        True power consumed by this load during the transient, |V|^2 conj(y_L),
        for the voltages V (one per time step). Needed because the trajectory's
        S is zero at load buses for t >= 1 (see the class docstring).
        """
        return np.abs(np.asarray(V)) ** 2 * np.conj(self.y_load)

    def prox(self, z: Trajectory, rho: float = 1.0) -> Trajectory:
        V0 = complex(z["v"][0, 0])
        I0 = complex(z["i"][0, 0])

        self.opti.set_value(self.v0r, V0.real)
        self.opti.set_value(self.v0i, V0.imag)
        self.opti.set_value(self.i0r, I0.real)
        self.opti.set_value(self.i0i, I0.imag)

        warm = self._warm if self._warm is not None else (V0.real, V0.imag, I0.real, I0.imag)
        for var, val in zip((self.V_re, self.V_im, self.I_re, self.I_im), warm):
            self.opti.set_initial(var, val)

        value, converged = _solve_or_last_iterate(self.opti)
        if not converged:
            self.n_failures += 1

        V_re, V_im, I_re, I_im = (float(value(var)) for var in (self.V_re, self.V_im, self.I_re, self.I_im))
        self._warm = (V_re, V_im, I_re, I_im)

        ret = z.copy()
        ret["v"][0, 0] = V_re + 1j * V_im
        ret["i"][0, 0] = I_re + 1j * I_im
        ret["s"][0, 0] = self.S_load
        # t >= 1: the load lives in the admittance matrix, so the device adds
        # no injection of its own. V is unconstrained here and passes through.
        ret["i"][1:, 0] = 0
        ret["s"][1:, 0] = 0
        return ret


class Generator(Proxable):
    """
    The behaviour of the generator is defined as (omega(t) is the rotor speed
    deviation from synchronous speed, omega_abs(t) - omega_s, in rad/s):
    B = {V(t), I(t), S(t)=P(t)+jQ(t), delta(t), omega(t) |
        S(t) = V(t) * conj(I(t)),
        delta_dot(t) = omega(t),
        omega_dot(t) = (P(0) - P(t) - D omega(t)) / M,
        E e^{j delta(t)} = V(t) + j Xd I(t),
        omega(0) = 0,
        }
    The DAE model is discretized with the exact first-order hold (gen_discrete_step_foh): the
    electrical power is interpolated linearly across each step, which a fault transient requires
    at dt = 0.05 s. The mechanical power is the pre-disturbance dispatch P(0), and E >= 0 is a
    constant internal to the projection (it is not part of the consensus trajectory; the last
    value is kept in self.E).
    V_bb = {V(t) | V_min^2 <= |V(0)|^2 <= V_max^2} (config.V_min / V_max are magnitudes)
    and f = ind_B + ind_V_bb, so prox_{rho, f} = proj_(B cap V_bb)
    The projection operator is computed using casadi and IPOPT, and is set up once. Previous solutions are used to warmstart the solver for the next iteration.
    Since this prox reduces to a pure projection, rho is ignored.

    The projection is taken in the rank-one leverage metric of algorithms.metric when one is set
    (set_metric): ||w - z||_M^2 = ||w - z||^2 - alpha (d^T (w - z))^2 with d the unit response of
    the Euclidean projection to a push on P(0) (leverage_direction). With alpha = 0 (the default)
    this is the plain Euclidean projection.
    """

    def __init__(self, config: Config, bus_index: int, gen_index: int, max_iter=500, tol=1e-8):
        self.config = config
        self.bus_index = bus_index
        self.gen_index = gen_index
        N = config.N
        self.N = N

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

        # Leverage metric: unit direction (same 8 blocks as the variables) and alpha.
        self.md = tuple(opti.parameter(N) for _ in range(8))
        self.malpha = opti.parameter()
        # Per-time-step weights of the prox term (set_time_weights); 1 = Euclidean.
        self.mw = opti.parameter(N)

        deviations = (
            self.V_re - self.v0r, self.V_im - self.v0i,
            self.I_re - self.i0r, self.I_im - self.i0i,
            self.P - self.p0, self.Q - self.q0,
            self.delta - self.d0, self.omega - self.w0,
        )
        along_d = sum(ca.dot(dk, dev) for dk, dev in zip(self.md, deviations))
        opti.minimize(sum(ca.dot(self.mw, dev**2) for dev in deviations) - self.malpha * along_d**2)

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

        # Exact FOH discretization of the swing equation, mechanical power = P(0)
        for n in range(N - 1):
            step = gen_discrete_step_foh(
                config, gen_index, self.delta[n], self.omega[n], self.P[n], self.P[n + 1], self.P[0]
            )
            opti.subject_to(self.delta[n + 1] == step["delta_next"])
            opti.subject_to(self.omega[n + 1] == step["omega_next"])

        # Pre-disturbance steady state: zero speed deviation
        opti.subject_to(self.omega[0] == 0)
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
        self.set_metric(None, 0.0)
        self.set_time_weights(np.ones(N))

    # ------------------------------------------------------------------
    # Metric
    # ------------------------------------------------------------------

    def set_time_weights(self, w: np.ndarray) -> None:
        """
        Diagonal time metric: the prox term becomes sum_t w_t ||w(t) - z(t)||^2 over all blocks.
        A weight that is uniform within each time step leaves every per-time-step-separable prox
        (F, loads, empty buses) unchanged, so setting it here alone is still exact metric ADMM.
        """
        self.time_weights = np.asarray(w, dtype=float)
        self.opti.set_value(self.mw, self.time_weights)

    @staticmethod
    def _blocks(z: Trajectory) -> tuple:
        """Single-bus Trajectory -> the 8 real blocks in the prox's variable order."""
        V, I, S = z["v"][:, 0], z["i"][:, 0], z["s"][:, 0]
        return (np.real(V), np.imag(V), np.real(I), np.imag(I), np.real(S), np.imag(S),
                np.asarray(z["delta"][:, 0], dtype=float), np.asarray(z["omega"][:, 0], dtype=float))

    @staticmethod
    def _from_blocks(template: Trajectory, blocks: tuple) -> Trajectory:
        V_re, V_im, I_re, I_im, P, Q, delta, omega = blocks
        ret = template.copy()
        ret["v"][:, 0] = V_re + 1j * V_im
        ret["i"][:, 0] = I_re + 1j * I_im
        ret["s"][:, 0] = P + 1j * Q
        ret["delta"][:, 0] = delta
        ret["omega"][:, 0] = omega
        return ret

    def set_metric(self, direction: "Trajectory | None", alpha: float) -> None:
        """
        direction: unit single-bus Trajectory d (None = Euclidean), alpha in [0, 1).
        See algorithms.metric.LeverageMetric.
        """
        self.metric_direction = direction
        self.metric_alpha = float(alpha)
        blocks = self._blocks(direction) if direction is not None else tuple(np.zeros(self.N) for _ in range(8))
        for par, val in zip(self.md, blocks):
            self.opti.set_value(par, val)
        self.opti.set_value(self.malpha, self.metric_alpha if direction is not None else 0.0)

    def leverage_direction(self, z: Trajectory, eps: float = 1e-2) -> "tuple[Trajectory, float]":
        """
        Response r = (proj(z + eps e_P(0)) - proj(z)) / eps of the *Euclidean* projection to a
        push on P(0), as a single-bus Trajectory, and its P(0) entry r_P (the pass-through).
        The current metric is restored afterwards; the warm start is left at proj(z).
        """
        saved = (self.metric_direction, self.metric_alpha)
        self.set_metric(None, 0.0)
        try:
            pushed = z.copy()
            pushed["s"][0, 0] += eps
            p1 = self.prox(pushed)
            p0 = self.prox(z)
        finally:
            self.set_metric(*saved)
        r = (p1 - p0) * (1.0 / eps)
        return r, float(np.real(r["s"][0, 0]))

    def prox(self, z: Trajectory, rho: float = 1.0) -> Trajectory:
        values = self._blocks(z)
        params = (self.v0r, self.v0i, self.i0r, self.i0i, self.p0, self.q0, self.d0, self.w0)
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
        return self._from_blocks(z, sol_vars)


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


class FCvxpy(Proxable):
    """
    Reference implementation of the network prox F (see class F for the
    definition of f). Kept for verification and as a fallback: it solves the
    whole prox as one cvxpy cone program over all buses and time steps.

    f is convex (quadratic cost, linear network / limit constraints, second-order-cone line
    limits) and its proximal operator is computed with cvxpy. The problem is built once in
    __init__ (_build_problem) in DPP form: rho is a nonnegative cp.Parameter and, to keep the
    objective parameter-affine, the prox term is expanded as
        rho/2 ||x - z||^2 = rho/2 ||x||^2 - <rho z, x> + const
    with rho*z passed in as parameters. cvxpy therefore canonicalises the problem once and every
    prox call is a parameter update plus a solve (CLARABEL by default). The canonicalisation
    time grows superlinearly with n_buses * N (0.6 s at 15 buses, 19 s at 45 buses for N = 20),
    which is why F below exploits the separability of f instead.
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
            # Slack angle reference for the pre-disturbance steady state
            V_im[0, 0] == 0,
            # Generation limits at t = 0
            P[:n_gens, 0] >= config.P_min,
            P[:n_gens, 0] <= config.P_max,
            Q[:n_gens, 0] >= config.Q_min,
            Q[:n_gens, 0] <= config.Q_max,
        ]

        # Network equations, with the admittance following the fault schedule;
        # while the fault is on the faulted bus is pinned to V = 0 instead of
        # its network row (its device current is the free fault current).
        for n in range(self.N):
            Y_n = np.asarray(config.Y_at(n))
            G_n, B_n = np.real(Y_n), np.imag(Y_n)
            rows = config.network_buses_at(n)
            constraints.append(I_re[rows, n] == (G_n @ V_re[:, n] - B_n @ V_im[:, n])[rows])
            constraints.append(I_im[rows, n] == (B_n @ V_re[:, n] + G_n @ V_im[:, n])[rows])
            if config.is_fault_step(n):
                constraints.append(V_re[config.fault_bus, n] == 0)
                constraints.append(V_im[config.fault_bus, n] == 0)

        # Voltage upper bound at t = 0 (second-order cones); the lower bound is
        # nonconvex and lives in the per-bus projections only.
        constraints.append(cp.norm(cp.vstack([V_re[:, 0], V_im[:, 0]]), 2, axis=0) <= np.asarray(config.V_max, dtype=float))

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


class F(Proxable):
    """
    f = c(w) + ind_Bnet(w) + ind_C(w)
    where w = (V, I, S=P+jQ, delta, omega),
    c(w) = sum_{j=1}^{n_gens} c_j P_j(0)^2
    B_net = {V, I, S | I(t) = Y(t) @ V(t) for all t, Im V_0(0) = 0 (slack angle reference)}
    where Y(t) follows the fault schedule of Config.Y_at(t): the pre-disturbance matrix at t = 0,
    the faulted matrix while the fault is on, and the post-fault matrix (faulted line tripped)
    after clearing. For t >= 1 the loads are inside Y as constant impedances.
    C = {V, I, S, delta, omega |
        P_{min, i} <= P_i(0) <= P_{max, i},
        Q_{min, i} <= Q_i(0) <= Q_{max, i},
        |I_{kl}(0)| <= I_{kl}^{max} for all lines (k, l),
        |V_k(0)| <= V_{max, k} for all buses k,
        |delta_i(t) - delta_coi(t)| <= 100 deg for all t,
    }
    omega (the rotor speed deviation omega_abs - omega_s) is unconstrained by f, so its prox is the
    identity.

    The voltage upper bound is deliberately in f as well as in the per-bus projections: when the
    OPF optimum sits on V_max (which the fault contingency makes common), a splitting in which
    only the projections know the bound never reaches consensus, because f keeps pulling the
    voltage past the limit while the projections keep clamping it. With the bound on both sides
    the two proxes agree. The lower bound |V_k(0)| >= V_min is nonconvex, so it cannot be part of
    the convex f and remains in the projections only.

    prox_{rho, f}(z) = argmin_w f(w) + rho/2 ||w - z||^2 separates over time steps and, within a
    time step, over the independent variable groups, so it is evaluated block by block:

      (V_t, I_t), t >= 1 : orthogonal projection onto the subspace {I = Y(t) V} (rho cancels).
                           With x_t = [V_re; V_im; I_re; I_im] and A = [[G, -B, -I, 0], [B, G, 0, -I]]
                           the projection is x - Q Q^T x, Q an orthonormal basis of range(A^T)
                           (economic QR). One basis per fault phase is computed at setup and the
                           steps of each phase go through a single matrix product.
      (V_0, I_0)         : same with the slack row Im V_0(0) = 0 appended (basis Q_0). If the projected
                           point satisfies every line limit |I_kl(0)| <= I_max and every voltage bound
                           |V_k(0)| <= V_max it is also the projection onto the intersection with those
                           cones, so the cone program is only solved (small DPP cvxpy SOCP in 4 n_buses
                           variables, built lazily) when a limit is violated.
      S_0 at generators  : 1-D closed form: P = clip(rho z_P / (2 c_i + rho), P_min, P_max), Q = clip(z_Q,
                           Q_min, Q_max) (the clipped unconstrained minimiser is exact for a convex
                           1-D quadratic on an interval); every other S entry is the identity.
      delta_t            : identity when |delta_i - delta_coi| <= 100 deg already holds, otherwise a
                           tiny DPP cvxpy QP in n_gens variables (per violating time step).
      omega              : identity.

    Setup cost is one QR of a (4 n_buses) x (2 n_buses + 1) matrix; the per-call cost is a few dense
    matrix products, independent of rho except for the closed-form P update.

    Leverage metric (algorithms.metric.LeverageMetric, set_metric): the prox in the metric
    M = I - sum_i alpha_i d_i d_i^T is obtained from the Euclidean one by dualising the rank-one
    terms. Since -alpha y^2 = min_lambda (lambda y + lambda^2 / 4 alpha),

        prox^M_f(z) = w*(lambda*),  w*(lambda) = prox_f(z - 1/2 sum_i lambda_i d_i),
        lambda* = argmin Phi(lambda),  Phi convex in lambda (n_gens-dimensional),
        stationarity: lambda_i = -2 alpha_i d_i^T (w*(lambda) - z).

    The Euclidean prox is piecewise linear in z (subspace projections, clips; the cone / COI
    fallbacks are piecewise smooth), so a semi-smooth Newton step on lambda with the Jacobian
    D^T P D (P = the local Jacobian of prox_f, obtained by finite differences along the d_i) is
    exact in the common case and converges in a few steps otherwise. Cost: n_gens + 2 Euclidean
    prox calls per Newton step.
    """

    def __init__(self, config: Config, solver: str = "CLARABEL"):
        self.config = config
        self.solver = solver
        nb, ng, N = config.n_buses, config.n_gens, config.N
        self.n_buses, self.n_gens, self.N = nb, ng, N

        G = np.asarray(config.G, dtype=float)
        B = np.asarray(config.B, dtype=float)
        self.G, self.B = G, B

        # One network-subspace basis per fault phase. Steps sharing the same
        # constraint rows share a basis, so the projection of each phase is a
        # single matrix product. Phases: t = 0 (pre-fault, with the slack row),
        # 1 <= t <= n_clear (fault on: the faulted bus's network row is replaced
        # by V = 0 there), t > n_clear (post-fault, line tripped).
        slack_row = np.zeros((1, 4 * nb))
        slack_row[0, nb] = 1.0  # V_im[0]
        self.Q0 = self._range_basis(np.vstack([self._network_rows(config.Y_at(0)), slack_row]))
        self._phase_bases = {}
        self._phase_of_step = np.zeros(N, dtype=int)
        for n in range(1, N):
            phase = 1 if config.is_fault_step(n) else 2
            self._phase_of_step[n] = phase
            if phase not in self._phase_bases:
                self._phase_bases[phase] = self._range_basis(self._constraint_rows_at(n))

        lines = [(k, l, G[k, l], B[k, l]) for k in range(nb) for l in range(k + 1, nb)
                 if G[k, l] != 0 or B[k, l] != 0]
        self.line_k = np.array([k for k, _, _, _ in lines], dtype=int)
        self.line_l = np.array([l for _, l, _, _ in lines], dtype=int)
        self.line_G = np.array([g for _, _, g, _ in lines], dtype=float)
        self.line_B = np.array([b for _, _, _, b in lines], dtype=float)
        self.line_limit = float(config.line_flow_limits)
        self.V_max = np.asarray(config.V_max, dtype=float)

        self.costs = np.asarray(config.costs, dtype=float)
        self.P_min = np.asarray(config.P_min, dtype=float)
        self.P_max = np.asarray(config.P_max, dtype=float)
        self.Q_min = np.asarray(config.Q_min, dtype=float)
        self.Q_max = np.asarray(config.Q_max, dtype=float)

        M = np.asarray(config.M, dtype=float)
        self.coi_w = M / M.sum()
        self._coi_qp = self._build_coi_qp() if ng > 1 else None
        self._t0_socp = None  # built on the first line-limit or voltage-bound violation

        self.n_socp_fallbacks = 0
        self.n_coi_projections = 0

        self.metric = LeverageMetric.identity()
        self._metric_dirs: dict = {}
        self.n_metric_newton_steps: list = []  # Newton steps per metric prox call

    # ------------------------------------------------------------------
    # Leverage metric
    # ------------------------------------------------------------------

    def set_metric(self, metric: "LeverageMetric | None") -> None:
        """Installs a LeverageMetric (None = Euclidean). Directions are stored as full-width trajectories."""
        self.metric = metric if metric is not None else LeverageMetric.identity()
        nb, ng, N = self.n_buses, self.n_gens, self.N
        template = Trajectory({
            "v": np.zeros((N, nb), dtype=complex), "i": np.zeros((N, nb), dtype=complex),
            "s": np.zeros((N, nb), dtype=complex), "delta": np.zeros((N, ng)), "omega": np.zeros((N, ng)),
        })
        self._metric_dirs = {}
        for i, d in self.metric.directions.items():
            full = template.copy()
            full.at_bus[i] = d
            self._metric_dirs[i] = full

    def _prox_metric(self, z: Trajectory, rho: float, max_steps: int = 8, tol: float = 1e-10) -> Trajectory:
        gens = sorted(self._metric_dirs)
        D = [self._metric_dirs[i] for i in gens]
        alpha = np.array([self.metric.alphas[i] for i in gens])
        lam = np.zeros(len(gens))
        h = 1e-3

        def shifted(lam_):
            zp = z.copy()
            for l_, d in zip(lam_, D):
                if l_ != 0.0:
                    zp = zp - d * (0.5 * l_)
            return zp

        def residual(lam_, w_):
            return lam_ / (2.0 * alpha) + np.array([inner(d, w_ - z) for d in D])

        w = self._prox_euclid(z, rho)  # = w*(lambda = 0)
        steps = 0
        g = residual(lam, w)
        while np.abs(g).max() >= tol and steps < max_steps:
            # Local Jacobian J = D^T P D by finite differences of the Euclidean
            # prox at z'(lambda); w is already prox(z'(lambda)).
            zp = shifted(lam)
            J = np.empty((len(gens), len(gens)))
            for j, dj in enumerate(D):
                Pdj = (self._prox_euclid(zp + dj * h, rho) - w) * (1.0 / h)
                J[:, j] = [inner(di, Pdj) for di in D]
            H = np.diag(1.0 / (2.0 * alpha)) - 0.5 * J
            lam = lam - np.linalg.solve(H, g)
            w = self._prox_euclid(shifted(lam), rho)
            g = residual(lam, w)
            steps += 1
        if np.abs(g).max() >= tol:
            logger.warning(f"F metric prox: Newton on lambda stopped after {steps} steps, |g|={np.abs(g).max():.2e}")
        self.n_metric_newton_steps.append(steps)
        return w

    def _network_rows(self, Y: np.ndarray) -> np.ndarray:
        """
        Rows of I = Y V in the real coordinates x = [V_re; V_im; I_re; I_im]:
            [[G, -B, -I, 0], [B, G, 0, -I]]
        """
        nb = self.n_buses
        Y = np.asarray(Y)
        G = np.real(Y).astype(float)
        B = np.imag(Y).astype(float)
        I_nb = np.eye(nb)
        Z_nb = np.zeros((nb, nb))
        return np.block([[G, -B, -I_nb, Z_nb], [B, G, Z_nb, -I_nb]])

    def _constraint_rows_at(self, n: int) -> np.ndarray:
        """
        Linear network constraints of step n >= 1 as rows in x = [V_re; V_im; I_re; I_im]:
        I_k = (Y V)_k for every bus k of config.network_buses_at(n), plus, while
        the fault is on, V_re = V_im = 0 at the faulted bus (whose own network
        row is dropped: its device current is the free fault current). The row
        count is always 2 n_buses and the rows are independent.
        """
        nb = self.n_buses
        rows = self._network_rows(self.config.Y_at(n))
        if not self.config.is_fault_step(n):
            return rows
        k = self.config.fault_bus
        keep = [r for r in range(2 * nb) if r not in (k, nb + k)]
        pin = np.zeros((2, 4 * nb))
        pin[0, k] = 1.0        # V_re[k] = 0
        pin[1, nb + k] = 1.0   # V_im[k] = 0
        return np.vstack([rows[keep], pin])

    @staticmethod
    def _range_basis(A: np.ndarray) -> np.ndarray:
        """Orthonormal basis of range(A^T) (A has full row rank here)."""
        Q, _ = qr(A.T, mode="economic") # type: ignore
        return np.asarray(Q)

    @staticmethod
    def _project_network(X: np.ndarray, Q: np.ndarray) -> np.ndarray:
        """Projects the columns of X onto the null space of A, given Q = basis of range(A^T)."""
        return X - Q @ (Q.T @ X)

    def _line_currents(self, x0: np.ndarray) -> np.ndarray:
        nb = self.n_buses
        V_re, V_im = x0[:nb], x0[nb:2 * nb]
        dV_re = V_re[self.line_k] - V_re[self.line_l]
        dV_im = V_im[self.line_k] - V_im[self.line_l]
        I_re = -dV_re * self.line_G + dV_im * self.line_B
        I_im = -dV_re * self.line_B - dV_im * self.line_G
        return np.hypot(I_re, I_im)

    def _build_t0_socp(self):
        nb = self.n_buses
        G, B = self.G, self.B
        x = cp.Variable(4 * nb)
        z = cp.Parameter(4 * nb)
        V_re, V_im, I_re, I_im = x[:nb], x[nb:2 * nb], x[2 * nb:3 * nb], x[3 * nb:]
        constraints = [
            I_re == G @ V_re - B @ V_im,
            I_im == B @ V_re + G @ V_im,
            V_im[0] == 0,
            # Voltage upper bound |V_k(0)| <= V_max,k (second-order cones). The
            # lower bound is nonconvex and stays in the per-bus projections.
            cp.norm(cp.vstack([V_re, V_im]), 2, axis=0) <= self.V_max,
        ]
        for k, l, g, b in zip(self.line_k, self.line_l, self.line_G, self.line_B):
            I_re_kl = -(V_re[k] - V_re[l]) * g + (V_im[k] - V_im[l]) * b
            I_im_kl = -(V_re[k] - V_re[l]) * b - (V_im[k] - V_im[l]) * g
            constraints.append(cp.norm(cp.hstack([I_re_kl, I_im_kl])) <= self.line_limit)
        # ||x - z||^2 up to a constant, parameter-affine for DPP
        problem = cp.Problem(cp.Minimize(cp.sum_squares(x) - 2 * (z @ x)), constraints)
        return problem, x, z

    def _build_coi_qp(self):
        ng = self.n_gens
        d = cp.Variable(ng)
        z = cp.Parameter(ng)
        dev = d - self.coi_w @ d
        constraints = [dev <= DELTA_COI_MAX, dev >= -DELTA_COI_MAX]
        problem = cp.Problem(cp.Minimize(cp.sum_squares(d) - 2 * (z @ d)), constraints)
        return problem, d, z

    # The fallback problems are tiny, so solve them tightly (1e-12 is not
    # reliably reachable and makes Clarabel return "optimal_inaccurate");
    # these are Clarabel option names, other solvers get their defaults.
    _TIGHT = {"tol_gap_abs": 1e-10, "tol_gap_rel": 1e-10, "tol_feas": 1e-10, "max_iter": 500}

    def _solve_small(self, problem, what: str) -> None:
        opts = self._TIGHT if self.solver == "CLARABEL" else {}
        problem.solve(solver=self.solver, **opts)
        if problem.status not in ("optimal", "optimal_inaccurate"):
            raise ValueError(f"F.prox {what} did not converge: {problem.status}")

    def _voltage_magnitudes(self, x0: np.ndarray) -> np.ndarray:
        nb = self.n_buses
        return np.hypot(x0[:nb], x0[nb:2 * nb])

    def _project_t0(self, x0_raw: np.ndarray) -> np.ndarray:
        x0 = self._project_network(x0_raw[:, None], self.Q0)[:, 0]
        # The subspace projection is also the projection onto the intersection
        # with the line-flow and voltage cones whenever it already satisfies
        # them; otherwise the small SOCP is needed.
        lines_ok = not self.line_k.size or not np.any(
            self._line_currents(x0) > self.line_limit * (1 + 1e-9) + 1e-12
        )
        voltage_ok = not np.any(self._voltage_magnitudes(x0) > self.V_max * (1 + 1e-9) + 1e-12)
        if not (lines_ok and voltage_ok):
            if self._t0_socp is None:
                self._t0_socp = self._build_t0_socp()
            problem, x, z = self._t0_socp
            z.value = x0_raw
            self._solve_small(problem, "t=0 SOCP (line flows / voltage bound)")
            x0 = np.asarray(x.value, dtype=float)
            self.n_socp_fallbacks += 1
        return x0

    def _project_coi(self, delta: np.ndarray) -> np.ndarray:
        """delta: (n_gens, N). Projects each violating column onto the COI polytope."""
        dev = delta - self.coi_w @ delta
        violating = np.where(np.abs(dev).max(axis=0) > DELTA_COI_MAX + 1e-12)[0]
        if violating.size == 0:
            return delta
        out = delta.copy()
        problem, d, z = cast(tuple, self._coi_qp)
        for t in violating:
            z.value = delta[:, t]
            self._solve_small(problem, "COI projection")
            out[:, t] = np.asarray(d.value, dtype=float)
            self.n_coi_projections += 1
        return out

    def prox(self, z: Trajectory, rho: float = 1.0) -> Trajectory:
        if self._metric_dirs:
            return self._prox_metric(z, rho)
        return self._prox_euclid(z, rho)

    def _prox_euclid(self, z: Trajectory, rho: float = 1.0) -> Trajectory:
        nb, ng, N = self.n_buses, self.n_gens, self.N
        # Trajectory arrays are (N, width); work width-major here.
        V0 = z["v"].T
        I0 = z["i"].T
        X = np.vstack([np.real(V0), np.imag(V0), np.real(I0), np.imag(I0)])  # (4 nb, N)

        Xp = np.empty_like(X)
        # One projection per fault phase (the steps of a phase share Y).
        for phase, Q in self._phase_bases.items():
            cols = np.flatnonzero(self._phase_of_step == phase)
            if cols.size:
                Xp[:, cols] = self._project_network(X[:, cols], Q)
        Xp[:, 0] = self._project_t0(X[:, 0])

        V = Xp[:nb] + 1j * Xp[nb:2 * nb]
        I = Xp[2 * nb:3 * nb] + 1j * Xp[3 * nb:]

        S = np.array(z["s"].T, dtype=complex)
        zP = np.real(S[:ng, 0])
        zQ = np.imag(S[:ng, 0])
        P = np.clip(rho * zP / (2.0 * self.costs + rho), self.P_min, self.P_max)
        Q = np.clip(zQ, self.Q_min, self.Q_max)
        S[:ng, 0] = P + 1j * Q

        delta = np.array(z["delta"].T, dtype=float)
        if ng > 1:
            delta = self._project_coi(delta)

        return Trajectory({
            "v": V.T,
            "i": I.T,
            "s": S.T,
            "delta": delta.T,
            "omega": z["omega"].copy(),
        })


def _generator_metric_args(metric: LeverageMetric, i: int) -> tuple:
    """(direction, alpha) for Generator.set_metric of bus i (Euclidean if i is not in the metric)."""
    if i in metric.directions:
        return metric.directions[i], metric.alphas[i]
    return None, 0.0


class BusBehaviours(Proxable):
    """
    A class implementing projections onto the behaviours of a set of buses, each with its own behaviour (e.g., constant power load, generator, etc.)

    leverage_directions / set_metric support the variable-metric ADMM (algorithms.metric): the
    former measures each generator's response to a push on P(0), the latter installs the
    resulting metric in the generator projections.
    """
    def __init__(self, behaviours: list[Proxable]):
        self.behaviours = behaviours

    def prox(self, z: Trajectory, rho: float = 1.0) -> Trajectory:
        ret = z.copy()
        for i, behaviour in enumerate(self.behaviours):
            ret_i = behaviour.prox(ret.at_bus[i], rho)
            ret.at_bus[i] = ret_i
        return ret

    def leverage_directions(self, z: Trajectory, eps: float = 1e-2) -> dict:
        return {i: b.leverage_direction(z.at_bus[i], eps)
                for i, b in enumerate(self.behaviours) if isinstance(b, Generator)}

    def set_metric(self, metric: LeverageMetric) -> None:
        for i, b in enumerate(self.behaviours):
            if isinstance(b, Generator):
                b.set_metric(*_generator_metric_args(metric, i))

    def set_time_weights(self, w: np.ndarray) -> None:
        for b in self.behaviours:
            if isinstance(b, Generator):
                b.set_time_weights(w)


class _AffinityWorker(threading.Thread):
    """
    Persistent worker thread that owns a fixed subset of bus behaviours and
    evaluates their proxes for every trajectory submitted to it. Every call
    into a behaviour (prox, leverage_direction, set_metric) goes through this
    thread, see BusBehavioursParallel.
    """

    def __init__(self, behaviours: list[Proxable], indices: list[int]):
        super().__init__(daemon=True)
        self.behaviours = behaviours
        self.indices = indices
        self._tasks: "queue.Queue" = queue.Queue()
        self._results: "queue.Queue" = queue.Queue()

    def submit(self, z: Trajectory, rho: float) -> None:
        self._tasks.put(("prox", z, rho))

    def submit_leverage(self, z: Trajectory, eps: float) -> None:
        self._tasks.put(("leverage", z, eps))

    def submit_metric(self, metric: LeverageMetric) -> None:
        self._tasks.put(("set_metric", metric))

    def collect(self):
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
            try:
                if task[0] == "prox":
                    _, z, rho = task
                    self._results.put([(i, self.behaviours[i].prox(z.at_bus[i], rho)) for i in self.indices])
                elif task[0] == "leverage":
                    _, z, eps = task
                    self._results.put([(i, self.behaviours[i].leverage_direction(z.at_bus[i], eps))
                                       for i in self.indices if isinstance(self.behaviours[i], Generator)])
                elif task[0] == "set_metric":
                    _, metric = task
                    for i in self.indices:
                        if isinstance(self.behaviours[i], Generator):
                            self.behaviours[i].set_metric(*_generator_metric_args(metric, i))
                    self._results.put([])
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

    def leverage_directions(self, z: Trajectory, eps: float = 1e-2) -> dict:
        for worker in self._workers:
            worker.submit_leverage(z, eps)
        out = {}
        for worker in self._workers:
            out.update(dict(worker.collect()))
        return out

    def set_metric(self, metric: LeverageMetric) -> None:
        for worker in self._workers:
            worker.submit_metric(metric)
        for worker in self._workers:
            worker.collect()

    def close(self) -> None:
        for worker in self._workers:
            worker.stop()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _process_worker_main(conn, config: Config, indices: list[int]) -> None:
    """
    Entry point of a bus-projection worker process. Builds the behaviours for
    its own buses (casadi objects never cross the process boundary), then
    serves ("prox", [single-bus Trajectory, ...], rho), ("leverage", [slices],
    eps), ("set_metric", {bus: (direction, alpha)}) and ("stats",) requests
    until it receives None.
    """
    behaviours = {i: make_bus_behaviour(config, i) for i in indices}
    while True:
        msg = conn.recv()
        if msg is None:
            break
        try:
            if msg[0] == "prox":
                _, slices, rho = msg
                conn.send(("ok", [behaviours[i].prox(zb, rho) for i, zb in zip(indices, slices)]))
            elif msg[0] == "leverage":
                _, slices, eps = msg
                conn.send(("ok", {i: behaviours[i].leverage_direction(zb, eps)
                                  for i, zb in zip(indices, slices) if isinstance(behaviours[i], Generator)}))
            elif msg[0] == "set_metric":
                _, args = msg
                for i, b in behaviours.items():
                    if isinstance(b, Generator):
                        b.set_metric(*args.get(i, (None, 0.0)))
                conn.send(("ok", None))
            elif msg[0] == "stats":
                conn.send(("ok", {
                    i: {"n_failures": getattr(b, "n_failures", 0), "E": getattr(b, "E", None)}
                    for i, b in behaviours.items()
                }))
            else:
                conn.send(("error", RuntimeError(f"unknown request {msg[0]!r}")))
        except BaseException as exc:  # ship the failure to the parent instead of dying silently
            conn.send(("error", RuntimeError(f"{type(exc).__name__}: {exc}")))
    conn.close()


class _ProcessWorker:
    """
    One persistent worker process owning a fixed subset of buses, driven over
    a Pipe. Started with the "spawn" context (the only one on Windows), so the
    module must be importable in the child and scripts need the
    `if __name__ == "__main__"` guard.
    """

    def __init__(self, config: Config, indices: list[int]):
        ctx = multiprocessing.get_context("spawn")
        self.indices = indices
        # The per-bus problems are tiny; one BLAS/OpenMP thread per worker
        # process avoids oversubscribing the cores (inherited by the child).
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(var, "1")
        self._conn, child_conn = ctx.Pipe()
        self._proc = ctx.Process(target=_process_worker_main, args=(child_conn, config, indices), daemon=True)
        self._proc.start()
        child_conn.close()

    def submit(self, z: Trajectory, rho: float) -> None:
        self._conn.send(("prox", [z.at_bus[i] for i in self.indices], rho))

    def submit_leverage(self, z: Trajectory, eps: float) -> None:
        self._conn.send(("leverage", [z.at_bus[i] for i in self.indices], eps))

    def submit_metric(self, metric: LeverageMetric) -> None:
        self._conn.send(("set_metric", {i: _generator_metric_args(metric, i) for i in self.indices}))

    def _receive(self):
        status, payload = self._conn.recv()
        if status == "error":
            raise payload
        return payload

    def collect(self) -> list[tuple[int, Trajectory]]:
        return list(zip(self.indices, self._receive()))

    def collect_raw(self):
        return self._receive()

    def stats(self) -> dict:
        self._conn.send(("stats",))
        return self._receive()

    def stop(self) -> None:
        try:
            self._conn.send(None)
            self._conn.close()
        except (OSError, ValueError, BrokenPipeError):
            pass
        self._proc.join(timeout=5)


class BusBehavioursProcesses(Proxable):
    """
    Same as BusBehaviours, but every bus's projection runs in one of a pool of
    persistent worker processes (static round-robin assignment, so generator
    buses are spread evenly and each behaviour's warm start stays with the bus).
    Unlike threads this is not limited by the interpreter lock, and the casadi
    thread-affinity problem cannot arise because each process has one thread.
    Per iteration one message per worker goes out (the single-bus slices of z)
    and one comes back.
    """

    def __init__(self, config: Config, n_workers: int | None = None):
        self.config = config
        n_buses = config.n_buses
        n_workers = max(1, min(n_buses, n_workers or (os.cpu_count() or 1)))
        self.n_workers = n_workers
        self.assignments = [list(range(w, n_buses, n_workers)) for w in range(n_workers)]
        self._workers = [_ProcessWorker(config, indices) for indices in self.assignments]

    def prox(self, z: Trajectory, rho: float = 1.0) -> Trajectory:
        ret = z.copy()
        for worker in self._workers:
            worker.submit(z, rho)
        for worker in self._workers:
            for i, ret_i in worker.collect():
                ret.at_bus[i] = ret_i
        return ret

    def leverage_directions(self, z: Trajectory, eps: float = 1e-2) -> dict:
        for worker in self._workers:
            worker.submit_leverage(z, eps)
        out = {}
        for worker in self._workers:
            out.update(worker.collect_raw())
        return out

    def set_metric(self, metric: LeverageMetric) -> None:
        for worker in self._workers:
            worker.submit_metric(metric)
        for worker in self._workers:
            worker.collect_raw()

    def stats(self) -> dict:
        merged: dict = {}
        for worker in self._workers:
            merged.update(worker.stats())
        return dict(sorted(merged.items()))

    def close(self) -> None:
        for worker in self._workers:
            worker.stop()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def rho_heuristic(iteration, rho_prev, r, s, tau=2, mu=10):
    """
    Residual-balancing heuristic. Known not to work for this nonconvex
    splitting; kept for reference only. Every prox in the splitting is a
    projection except the small P(0) cost term, so the ADMM iterates are
    essentially independent of the value of rho (measured 2026-09-09: fixed
    rho in {5 .. 1000} gives the same residual plateau / limit cycle). What
    converges is the dual rescaling that admm.admm applies whenever rho
    changes, i.e. a ramp that never stops: see rho_geometric / rho_for_size.
    """
    if rho_prev == 0:
        return 2.0
    elif np.linalg.norm(r) > mu * np.linalg.norm(s):
        return tau * rho_prev
    elif np.linalg.norm(s) > mu * np.linalg.norm(r):
        return rho_prev / tau
    else:
        return rho_prev


def rho_geometric(rho_0: float = 2.0, growth: float = 1.007, rho_max: float = float("inf")):
    """
    Returns a rho schedule for admm.admm: rho_k = min(rho_0 * growth**k, rho_max),
    i.e. a geometric ramp from rho_0, independent of the residuals.

    The ramp is not about the size of rho: admm.admm rescales the dual variable
    by rho_{k-1}/rho_k whenever rho changes, and that damping of the dual
    memory is what breaks the limit cycles of the otherwise rho-invariant
    iteration. A cap that binds before the stop criterion is met switches the
    damping off and the residual drifts back into the cycle (measured: cap 200
    reached at iteration ~660 makes the 15-bus case stall at r ~ 2e-3, and
    holding rho once r < 1e-4 stalls too). Leave rho_max at infinity and let
    the stop rule (r and |dz| below the threshold) end the run.

    The growth rate trades speed for bias: the faster the dual is damped, the
    further the fixed point drifts from the true optimum. Use rho_for_size to
    pick it from the network size.
    """
    def schedule(iteration, rho_prev, r, s):
        return float(min(rho_0 * growth ** iteration, rho_max))
    return schedule


def rho_for_size(n_buses: int, rho_0: float = 2.0, max_growth: float = 0.007, growth_scale: float = 0.18):
    """
    Size-dependent geometric ramp: growth = 1 + min(max_growth, growth_scale / n_buses),
    uncapped. Empirical calibration (N = 10, flat start, dispatch error vs
    centralized once r < ~1e-5):

        buses   growth   iterations to r < 1e-4   dispatch error
          8     1.007            650                 0.003 %
         15     1.007           1660                 0.3 %
         15     1.02             590                 1.7 %
         15     1.05             560                12 %
         25     1.007           1310                 0.2 %
         45     1.007           1480                 2.1 %
         45     1.004           2560                 0.11 %

    Larger networks need slower damping to stay unbiased, hence the 1/n_buses
    rule (1.007 up to 25 buses, ~1.004 at 45, ~1.0013 at 135, ~1.0004 at 405);
    the 135+ values are extrapolated, not calibrated.
    """
    growth = 1.0 + min(max_growth, growth_scale / max(int(n_buses), 1))
    return rho_geometric(rho_0=rho_0, growth=growth, rho_max=float("inf"))


def make_bus_behaviour(config: Config, bus_index: int) -> Proxable:
    """
    Behaviour of a single bus: generators occupy the first n_gens indices,
    loads the next n_loads, the remaining buses are empty.
    """
    n_gens = config["n_gens"]
    n_loads = config["n_loads"]
    if bus_index < n_gens:
        return Generator(config, bus_index=bus_index, gen_index=bus_index)
    if bus_index < n_gens + n_loads:
        return ConstPowerLoad(config, bus_index=bus_index, load_index=bus_index - n_gens)
    return EmptyBus(config, bus_index=bus_index)


def make_bus_behaviours(config: Config, parallel: "bool | str" = False, n_workers: int | None = None) -> Proxable:
    """
    parallel: False / "sequential" -> BusBehaviours (one after another),
              True / "threads"     -> BusBehavioursParallel (pinned worker threads),
              "processes"          -> BusBehavioursProcesses (worker processes).
    """
    if isinstance(parallel, bool):
        mode = "threads" if parallel else "sequential"
    else:
        mode = parallel
    if mode == "processes":
        return BusBehavioursProcesses(config, n_workers)
    behaviours = [make_bus_behaviour(config, i) for i in range(config["n_buses"])]
    if mode == "threads":
        return BusBehavioursParallel(behaviours, n_workers)
    if mode == "sequential":
        return BusBehaviours(behaviours)
    raise ValueError(f"unknown parallel mode {parallel!r}; use False, True, 'sequential', 'threads' or 'processes'")


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
# Dynamic state   x = [delta, omega]      (rotor angle [rad], speed deviation
#                                          omega = omega_abs - omega_s [rad/s])
# Input           u = [P_0, P_e]          (mechanical power, electrical power)
#
# Continuous time:  x_dot = A x + B_u u + c      (c = 0 for the deviation state;
#                                                 kept so the helpers stay general)
#   delta_dot = omega
#   omega_dot = (P_0 - P_e - D omega) / M
# Using the deviation keeps every trajectory signal O(1) and makes the
# synchronous steady state x = 0 (omega_s never enters the equations).
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
    with omega the speed deviation from omega_s, so c = 0.
    Returns numeric (A (2x2), B_u (2x2), c (2,)).
    """
    D = float(config.D[gen_idx])
    M = float(config.M[gen_idx])
    A = np.array([[0.0, 1.0], [0.0, -D / M]])
    B_u = np.array([[0.0, 0.0], [1.0 / M, -1.0 / M]])
    c = np.zeros(2)
    return A, B_u, c


def gen_discrete_dynamics(config: Config, gen_idx: int, dt: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Exact zero-order-hold discretization of gen_continuous_matrices with step dt
    (defaults to config.dt):
        x_{n+1} = A_d x_n + B_d u_n + c_d
    with A_d = expm(A dt), Phi = int_0^dt expm(A s) ds, B_d = Phi B_u, c_d = Phi c
    (c_d = 0 for the speed-deviation state). A_d and Phi are read off a single
    matrix exponential of the augmented matrix [[A dt, I dt], [0, 0]].

    Note: with realistic inertia the zero-order hold needs dt ~ 0.01 s to track
    a fault transient; gen_discrete_dynamics_foh below is accurate at dt = 0.05
    and is what the solvers use. This function is kept for reference and tests.
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


def gen_discrete_dynamics_foh(config: Config, gen_idx: int, dt: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Exact first-order-hold (ramp-invariant) discretization of
    gen_continuous_matrices with step dt (defaults to config.dt):
        x_{n+1} = A_d x_n + B_0 u_n + B_1 u_{n+1}
    i.e. the input is interpolated linearly across the step instead of being
    held constant. The matrices come from one exponential of the augmented
    block matrix [[A dt, B dt, 0], [0, 0, I], [0, 0, 0]]:
    with expm(...) = [[A_d, G_1, G_2], ...], B_0 = G_1 - G_2 and B_1 = G_2.

    The electrical power P_e varies quickly during a fault transient, so the
    linear interpolation is far more accurate than a hold: at dt = 0.05 s the
    FOH matches an adaptive ODE solver within 1 degree of rotor angle where the
    ZOH is off by tens of degrees. The step stays affine in (x_n, u_n, u_{n+1}),
    so the optimisation problems keep the same structure.
    """
    if dt is None:
        dt = config.dt
    A, B_u, _ = gen_continuous_matrices(config, gen_idx)
    n, m = 2, 2
    aug = np.zeros((n + 2 * m, n + 2 * m))
    aug[:n, :n] = A * dt
    aug[:n, n:n + m] = B_u * dt
    aug[n:n + m, n + m:] = np.eye(m)
    aug_d = expm(aug)
    A_d = aug_d[:n, :n]
    G_1 = aug_d[:n, n:n + m]
    G_2 = aug_d[:n, n + m:]
    return A_d, G_1 - G_2, G_2


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
    x_dot = f(x, y, u). omega is the speed deviation omega_abs - omega_s [rad/s].
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
    where S_re is the electrical power held over the step, P_0 the mechanical
    power and omega the speed deviation omega_abs - omega_s [rad/s]. Works with
    floats and casadi symbols.
    """
    A_d, B_d, c_d = gen_discrete_dynamics(config, gen_idx, dt)
    A_d, B_d, c_d = A_d.tolist(), B_d.tolist(), c_d.tolist()

    return {
        "delta_next": A_d[0][0] * delta + A_d[0][1] * omega + B_d[0][0] * P_0 + B_d[0][1] * S_re + c_d[0],
        "omega_next": A_d[1][0] * delta + A_d[1][1] * omega + B_d[1][0] * P_0 + B_d[1][1] * S_re + c_d[1],
    }


def gen_discrete_step_foh(
        config: Config,
        gen_idx: int,
        delta: Scalar,
        omega: Scalar,
        S_re: Scalar,
        S_re_next: Scalar,
        P_0: Scalar,
        dt: float | None = None
    ) -> Dict[str, Scalar]:
    """
    One exact first-order-hold step of the generator swing dynamics:
        [delta_next, omega_next] = A_d [delta, omega] + B_0 [P_0, S_re] + B_1 [P_0, S_re_next]
    i.e. the electrical power is interpolated linearly from S_re at the start of
    the step to S_re_next at its end (the mechanical power P_0 is constant, so
    it appears in both terms). omega is the speed deviation omega_abs - omega_s
    [rad/s]. Works with floats and casadi symbols.

    This is what the solvers use: during a fault P_e changes far too quickly for
    a zero-order hold to track at dt = 0.05 s (see gen_discrete_dynamics_foh).
    """
    A_d, B_0, B_1 = gen_discrete_dynamics_foh(config, gen_idx, dt)
    A_d, B_0, B_1 = A_d.tolist(), B_0.tolist(), B_1.tolist()

    return {
        "delta_next": (A_d[0][0] * delta + A_d[0][1] * omega
                       + B_0[0][0] * P_0 + B_0[0][1] * S_re
                       + B_1[0][0] * P_0 + B_1[0][1] * S_re_next),
        "omega_next": (A_d[1][0] * delta + A_d[1][1] * omega
                       + B_0[1][0] * P_0 + B_0[1][1] * S_re
                       + B_1[1][0] * P_0 + B_1[1][1] * S_re_next),
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