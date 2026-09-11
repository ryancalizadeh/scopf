"""
Forward simulation of the discretized generator / network DAE from a dispatch.

Given a Config and the pre-disturbance generator dispatch (P(0), Q(0) at the
generator buses), `simulate_dispatch` rebuilds the trajectory that the TSC-OPF
models, using only the discretized dynamics and the network equations:

  1. pre-disturbance operating point: least-squares power flow with every bus
     power fixed (generators P+jQ, loads -S_load, empty buses I = 0) and bus 0
     as the angle reference. The system is over-determined by one equation,
     which absorbs the small consensus residual of an ADMM dispatch (zero for
     the centralized solution);
  2. E_i and delta_i(0) from E e^{j delta} = V + j Xd I, omega_i(0) = 0
     (omega is the speed deviation), mechanical power P_0 = P(0);
  3. for n >= 1: advance the rotors with the exact first-order-hold map
     (algorithms.common.gen_discrete_step_foh) and solve the network for V(n)
     with the admittance of config.Y_at(n) (pre-fault, faulted, or post-fault)
     and the generators represented by E e^{j delta(n)} behind Xd.

For n >= 1 the loads are constant impedances inside Y_at(n), so the network is
*linear* in V and is solved directly; only the pre-disturbance step needs the
casadi/IPOPT least-squares power flow. The first-order hold needs the electrical
power at the end of the step as well as at its start, so each step iterates
rotor update and network solve to a fixed point (a handful of iterations).

Note on load power: for n >= 1 the trajectory's S is zero at the load buses,
because the load current is carried by the admittance matrix rather than by a
device injection. The true consumption is |V(n)|^2 * conj(y_L); use
`load_consumption` below rather than reading S there.

The returned Trajectory has the same keys and conventions as the solvers'
trajectories ("v", "i", "s", "delta", "omega", "E").
"""
from typing import Dict, Tuple

import numpy as np
import casadi as ca

from Config import Config
from Trajectory import Trajectory
from algorithms.common import gen_discrete_step_foh, _IPOPT_QUIET


def load_consumption(config: Config, V: np.ndarray) -> np.ndarray:
    """
    Power drawn by the constant-impedance loads over a trajectory, |V|^2 conj(y_L),
    shape (N, n_loads). For n = 0 the loads are constant power, so the first row
    is the nominal load instead.
    """
    ng, nl = config.n_gens, config.n_loads
    V_load = np.asarray(V)[:, ng:ng + nl]
    consumption = np.abs(V_load) ** 2 * np.conj(config.load_admittance())
    consumption[0] = config.load_P[:nl] + 1j * config.load_Q[:nl]
    return consumption


class _InitialStateSolver:
    """
    Least-squares power flow for the pre-disturbance state: generator P and Q,
    load powers and empty-bus currents all fixed, Im V_0 = 0.
    """

    def __init__(self, config: Config):
        nb, ng, nl = config.n_buses, config.n_gens, config.n_loads
        self.nb, self.ng = nb, ng
        G = np.asarray(config.G, dtype=float)
        B = np.asarray(config.B, dtype=float)

        opti = ca.Opti()
        self.V_re = opti.variable(nb)
        self.V_im = opti.variable(nb)
        I_re = G @ self.V_re - B @ self.V_im
        I_im = B @ self.V_re + G @ self.V_im
        P = self.V_re * I_re + self.V_im * I_im
        Q = self.V_im * I_re - self.V_re * I_im

        self.P0 = opti.parameter(ng)
        self.Q0 = opti.parameter(ng)

        residuals = []
        for i in range(ng):
            residuals.append(P[i] - self.P0[i])
            residuals.append(Q[i] - self.Q0[i])
        for k in range(ng, ng + nl):
            residuals.append(P[k] + float(config.load_P[k - ng]))
            residuals.append(Q[k] + float(config.load_Q[k - ng]))
        for k in range(ng + nl, nb):
            residuals.append(I_re[k])
            residuals.append(I_im[k])
        self.residual = ca.vertcat(*residuals)
        opti.minimize(ca.sumsqr(self.residual))
        opti.subject_to(self.V_im[0] == 0)  # slack angle reference, as in the OPF
        opti.solver("ipopt", {**_IPOPT_QUIET, "ipopt.tol": 1e-12, "ipopt.max_iter": 500})
        self.opti = opti
        self.G, self.B = G, B

    def solve(self, P0: np.ndarray, Q0: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        self.opti.set_value(self.P0, P0)
        self.opti.set_value(self.Q0, Q0)
        self.opti.set_initial(self.V_re, np.ones(self.nb))
        self.opti.set_initial(self.V_im, np.zeros(self.nb))
        try:
            sol = self.opti.solve()
            value = sol.value
        except RuntimeError:
            value = self.opti.debug.value
        V = np.reshape(value(self.V_re), self.nb) + 1j * np.reshape(value(self.V_im), self.nb)
        I = (self.G + 1j * self.B) @ V
        mismatch = float(np.linalg.norm(np.reshape(value(self.residual), -1)))
        return V, I, mismatch


def _solve_network(config: Config, Y: np.ndarray, E: np.ndarray, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solves the transient network for V, given the generator internal voltages
    E e^{j delta}. With constant-impedance loads folded into Y the system is
    linear: I = Y V everywhere, and at a generator bus
        E e^{j delta} = V + j Xd I  =>  (j Xd Y + e_i^T) V = E e^{j delta}.
    Every other bus contributes its row of I - Y V = 0 (zero injection).
    """
    nb, ng = config.n_buses, config.n_gens
    Y = np.asarray(Y, dtype=complex)
    # Non-generator buses inject nothing, so their equation is (Y V)_k = 0.
    A = Y.copy()
    b = np.zeros(nb, dtype=complex)
    # Generator buses instead impose E e^{j delta} = V + j Xd (Y V).
    for i in range(ng):
        A[i, :] = 1j * float(config.Xd[i]) * Y[i, :]
        A[i, i] += 1.0
        b[i] = E[i] * np.exp(1j * delta[i])
    V = np.linalg.solve(A, b)
    return V, Y @ V


def simulate_dispatch(config: Config, P0: np.ndarray, Q0: np.ndarray, return_info: bool = False):
    """
    Forward-simulates the discretized DAE from the generator dispatch (P0, Q0)
    through the fault contingency of `config`. Returns a Trajectory (keys v, i,
    s, delta, omega, E); with return_info=True also a dict with the power-flow
    mismatch at t=0, the maximum network residual over the transient, the
    mechanical powers and the fixed-point iteration counts.
    """
    N, nb, ng = config.N, config.n_buses, config.n_gens
    P0 = np.asarray(P0, dtype=float).reshape(ng)
    Q0 = np.asarray(Q0, dtype=float).reshape(ng)
    Xd = np.asarray(config.Xd, dtype=float)

    V = np.zeros((N, nb), dtype=complex)
    I = np.zeros((N, nb), dtype=complex)
    delta = np.zeros((N, ng))
    omega = np.zeros((N, ng))

    # 1. pre-disturbance operating point
    V[0], I[0], mismatch0 = _InitialStateSolver(config).solve(P0, Q0)
    S0 = V[0] * np.conj(I[0])
    P_mech = np.real(S0[:ng])  # realised P(0): keeps t=0 an exact fixed point

    # 2. internal voltages
    E_phasor = V[0, :ng] + 1j * Xd * I[0, :ng]
    E = np.abs(E_phasor)
    delta[0] = np.angle(E_phasor)

    # 3. transient
    P_e = P_mech.copy()
    max_net_residual = 0.0
    max_fixed_point_iters = 0
    for n in range(N - 1):
        Y_next = np.asarray(config.Y_at(n + 1))
        P_e_next = P_e.copy()
        for iteration in range(100):
            for i in range(ng):
                step = gen_discrete_step_foh(config, i, delta[n, i], omega[n, i], P_e[i], P_e_next[i], P_mech[i])
                delta[n + 1, i] = step["delta_next"]
                omega[n + 1, i] = step["omega_next"]
            V[n + 1], I[n + 1] = _solve_network(config, Y_next, E, delta[n + 1])
            updated = np.real(V[n + 1, :ng] * np.conj(I[n + 1, :ng]))
            converged = np.abs(updated - P_e_next).max() < 1e-11
            P_e_next = updated
            if converged:
                break
        max_fixed_point_iters = max(max_fixed_point_iters, iteration + 1)
        residual = np.abs(I[n + 1] - Y_next @ V[n + 1]).max()
        max_net_residual = max(max_net_residual, float(residual))
        P_e = P_e_next

    traj = Trajectory({
        "v": V,
        "i": I,
        "s": V * np.conj(I),
        "delta": delta,
        "omega": omega,
        "E": np.tile(E, (N, 1)),
    })
    if not return_info:
        return traj
    info: Dict[str, object] = {
        "powerflow_mismatch": mismatch0,
        "max_network_residual": max_net_residual,
        "max_fixed_point_iters": max_fixed_point_iters,
        "P_mech": P_mech,
        "E": E,
    }
    return traj, info
