import time
import numpy as np
import casadi as ca
from Config import Config
from algorithms.base import SolveResult


def solve(config: Config) -> SolveResult:
    G = config.G
    B = config.B
    n_buses = config.n_buses
    n_gens = config.n_gens
    costs = config.costs
    load_P = config.load_P
    load_Q = config.load_Q
    V_max = config.V_max
    V_min = config.V_min
    P_max = config.P_max
    P_min = config.P_min
    Q_max = config.Q_max
    Q_min = config.Q_min

    start = time.perf_counter()

    opti = ca.Opti()

    V_re = opti.variable(n_buses)
    V_im = opti.variable(n_buses)
    I_re = opti.variable(n_buses)
    I_im = opti.variable(n_buses)
    P = opti.variable(n_buses)
    Q = opti.variable(n_buses)

    opti.minimize(sum(costs[i] * P[i]**2 for i in range(n_gens)))

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

    opti.solver('ipopt', {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'})

    sol = opti.solve()

    runtime = time.perf_counter() - start

    dispatch = sol.value(P)[:n_gens]
    obj = float(np.sum(costs * dispatch ** 2))

    return SolveResult(
        dispatch=dispatch,
        obj=obj,
        runtime=runtime,
        p_residual=None,
        s_residual=None,
    )
