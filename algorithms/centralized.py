import time
import numpy as np
import casadi as ca
from Config import Config
from Trajectory import Trajectory
from algorithms.base import SolveResult
from algorithms.common import gen_alg_eqs, gen_discrete_step, gen_coi_angle


# Transient stability limit on the rotor angle relative to the centre of inertia
DELTA_COI_MAX = np.deg2rad(100.0)


def solve(config: Config) -> SolveResult:
    """
    Transient stability constrained OPF (TSC-OPF).

    Time step n = 0 is the pre-disturbance steady state and carries the usual
    OPF constraints (P/Q limits, voltage bounds, line flows, slack reference).
    For n >= 1 every load is scaled by config.load_step_factor and the system
    evolves according to the classical generator model (see algorithms.common),
    discretized exactly with a zero-order hold. The dispatch is required to
    keep |delta_i - delta_coi| <= 100 deg over the whole horizon.
    """
    N = config.N
    G = config.G
    B = config.B
    n_buses = config.n_buses
    n_gens = config.n_gens
    n_loads = config.n_loads
    costs = config.costs
    load_P = config.load_P
    load_Q = config.load_Q
    V_max = config.V_max
    V_min = config.V_min
    P_max = config.P_max
    P_min = config.P_min
    Q_max = config.Q_max
    Q_min = config.Q_min
    omega_s = config.omega_s
    load_step_factor = config.load_step_factor

    start = time.perf_counter()

    opti = ca.Opti()

    V_re = opti.variable(n_buses, N)
    V_im = opti.variable(n_buses, N)
    I_re = opti.variable(n_buses, N)
    I_im = opti.variable(n_buses, N)
    P = opti.variable(n_buses, N)
    Q = opti.variable(n_buses, N)

    delta = opti.variable(n_gens, N)
    omega = opti.variable(n_gens, N)
    E = opti.variable(n_gens)  # voltage behind transient reactance (constant over the horizon)

    opti.minimize(sum(costs[i] * P[i, 0]**2 for i in range(n_gens)))

    for n in range(N):
        # Network equations
        opti.subject_to(I_re[:, n] == G @ V_re[:, n] - B @ V_im[:, n])
        opti.subject_to(I_im[:, n] == B @ V_re[:, n] + G @ V_im[:, n])

        # Hyperbola constraints
        for k in range(n_buses):
            opti.subject_to(P[k, n] == V_re[k, n] * I_re[k, n] + V_im[k, n] * I_im[k, n])
            opti.subject_to(Q[k, n] == V_im[k, n] * I_re[k, n] - V_re[k, n] * I_im[k, n])

        # Load equality constraints (load step disturbance for n >= 1)
        load_scale = 1.0 if n == 0 else load_step_factor
        for k in range(n_gens, n_gens + n_loads):
            opti.subject_to(P[k, n] == -load_scale * load_P[k - n_gens])
            opti.subject_to(Q[k, n] == -load_scale * load_Q[k - n_gens])

        # Empty bus constraints
        for k in range(n_gens + n_loads, n_buses):
            opti.subject_to(I_re[k, n] == 0)
            opti.subject_to(I_im[k, n] == 0)

        # Generator algebraic equations: E e^{j delta} = V + j Xd I
        for i in range(n_gens):
            r = gen_alg_eqs(
                config, i,
                V_re[i, n], V_im[i, n], I_re[i, n], I_im[i, n],
                P[i, n], Q[i, n], delta[i, n], omega[i, n], E[i], P[i, 0],
            )
            opti.subject_to(r["r_re"] == 0)
            opti.subject_to(r["r_im"] == 0)

        # Transient stability constraint: |delta_i - delta_coi| <= 100 deg
        # (identically zero, hence skipped, with a single generator)
        if n_gens > 1:
            delta_coi = gen_coi_angle(config, delta[:, n])
            for i in range(n_gens):
                opti.subject_to(opti.bounded(-DELTA_COI_MAX, delta[i, n] - delta_coi, DELTA_COI_MAX))

    # Generator differential equations (exact ZOH discretization)
    # Mechanical power P_0 is the pre-disturbance dispatch P[i, 0]
    for n in range(N - 1):
        for i in range(n_gens):
            step = gen_discrete_step(config, i, delta[i, n], omega[i, n], P[i, n], P[i, 0])
            opti.subject_to(delta[i, n + 1] == step["delta_next"])
            opti.subject_to(omega[i, n + 1] == step["omega_next"])

    # Pre-disturbance steady state (delta[i, 0] follows from the algebraic equations)
    for i in range(n_gens):
        opti.subject_to(omega[i, 0] == omega_s)
        opti.subject_to(E[i] >= 0)

    # Reference (slack) bus angle: only for the steady state; afterwards the
    # generator DAE fixes every angle in the synchronous frame.
    opti.subject_to(V_im[0, 0] == 0)

    # Voltage magnitude constraints (for n=0)
    for k in range(n_buses):
        V_sq = V_re[k, 0]**2 + V_im[k, 0]**2
        opti.subject_to(V_sq >= V_min[k]**2)
        opti.subject_to(V_sq <= V_max[k]**2)

    # Power constraints (for n=0)
    for i in range(n_gens):
        opti.subject_to(P[i, 0] >= P_min[i])
        opti.subject_to(P[i, 0] <= P_max[i])
        opti.subject_to(Q[i, 0] >= Q_min[i])
        opti.subject_to(Q[i, 0] <= Q_max[i])

    # Line flow constraints (for n=0)
    # TODO check where/when line flow constraints become active
    for k in range(n_buses):
        for l in range(k + 1, n_buses):
            if G[k, l] == 0 and B[k, l] == 0:
                continue  # no line between k and l
            I_re_kl = -(V_re[k,0]-V_re[l,0])*G[k,l] + (V_im[k,0]-V_im[l,0])*B[k,l]
            I_im_kl = -(V_re[k,0]-V_re[l,0])*B[k,l] - (V_im[k,0]-V_im[l,0])*G[k,l]
            I_kl_sq = I_re_kl**2 + I_im_kl**2
            opti.subject_to(I_kl_sq <= config.line_flow_limits**2)

    opti.set_initial(V_re, np.ones((n_buses, N)))
    opti.set_initial(V_im, np.zeros((n_buses, N)))
    opti.set_initial(delta, np.zeros((n_gens, N)))
    opti.set_initial(omega, np.full((n_gens, N), omega_s))
    opti.set_initial(E, np.ones(n_gens))

    opti.solver('ipopt', {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'})

    sol = opti.solve()

    runtime = time.perf_counter() - start

    def val(x, shape):
        return np.reshape(np.asarray(sol.value(x), dtype=float), shape)

    V_re_sol = val(V_re, (n_buses, N))
    V_im_sol = val(V_im, (n_buses, N))
    I_re_sol = val(I_re, (n_buses, N))
    I_im_sol = val(I_im, (n_buses, N))
    P_sol = val(P, (n_buses, N))
    Q_sol = val(Q, (n_buses, N))
    delta_sol = val(delta, (n_gens, N))
    omega_sol = val(omega, (n_gens, N))
    E_sol = val(E, (n_gens,))

    dispatch = P_sol[:n_gens, 0]
    obj = float(np.sum(costs * dispatch ** 2))

    trajectory = Trajectory({
        "v": (V_re_sol + 1j * V_im_sol).T,
        "i": (I_re_sol + 1j * I_im_sol).T,
        "s": (P_sol + 1j * Q_sol).T,
        "delta": delta_sol.T,
        "omega": omega_sol.T,
        "E": np.tile(E_sol, (N, 1)),
    })

    return SolveResult(
        dispatch=dispatch,
        obj=obj,
        runtime=runtime,
        p_residual=None,
        s_residual=None,
        trajectory=trajectory,
    )
