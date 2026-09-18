import time
import numpy as np
import cvxpy as cp
from Config import Config
from Trajectory import Trajectory
from algorithms.base import SolveResult


def solve(config: Config) -> SolveResult:
    """
    Centralized dynamic DC optimal power flow over a T-hour horizon.

    Solves one convex QP over all N = T/dt steps jointly (not a sequence of
    single-period OPFs): the storage and thermal states couple the steps, so the
    dispatch at t depends on the whole horizon.

    Index layout: buses are ordered [generators | loads | thermal | batteries],
    G = {0..n_g-1}, L, H and S the following blocks, V = G u L u H u S.

        min_{p, theta, q, T}  sum_{t=0}^{N-1} sum_{i in G} alpha_i p_{i,t}^2 + beta_i p_{i,t}

        s.t.  P_t = B theta_t,                               t = 0..N-1     (DC power flow)
              theta_{0,t} = 0                                               (angle reference)
              |B_ij (theta_{i,t} - theta_{j,t})| <= F_ij,     (i,j) in E     (line limits)

              P_{i,t} = p_{i,t}            i in G                           (generation)
              P_{i,t} = -P^load_{i,t}      i in L                           (fixed demand)
              P_{i,t} = -p_{i,t}           i in H                           (thermal draw)
              P_{i,t} = p_{i,t}            i in S                           (battery, discharge > 0)

              P^min_i <= p_{i,t} <= P^max_i                   i in G        (capacity)
              R^min_i <= p_{i,t+1} - p_{i,t} <= R^max_i       i in G        (ramping)

              q_{i,t+1} = q_{i,t} - dt p_{i,t}                i in S        (SOC dynamics)
              q^min_i <= q_{i,t} <= q^max_i,  q_{i,0} = q0_i,  q_{i,N} = qT_i
              p^min_i <= p_{i,t} <= p^max_i                   i in S

              T_{i,t+1} = (1 - mu_i/c_i) T_{i,t} + (eta_i/c_i) p_{i,t}
                                          + (mu_i/c_i) Tamb_{i,t}   i in H  (thermal dynamics)
              T^min_{i,t} <= T_{i,t} <= T^max_{i,t},  T_{i,0} = T0_i         (comfort band)
              p^min_i <= p_{i,t} <= p^max_i                   i in H

    Sign conventions: P is net injection, so load-like devices are negative. For
    thermal units p > 0 heats, and the unit draws that power from its bus.

    The objective is convex quadratic and every constraint is affine, so this is a
    QP solved with CLARABEL via cvxpy; the returned point is a global optimum.
    Convergence and residual fields of SolveResult are None, as they apply only to
    the ADMM solvers.
    """
    N = config.N
    dt = config.dt
    n_buses = config.n_buses
    n_gens = config.n_gens
    n_loads = config.n_loads
    n_thermal = config.n_thermal
    n_battery = config.n_battery

    # Bus index blocks: [gens | loads | thermal | batteries]
    load_slice = slice(n_gens, n_gens + n_loads)
    thermal_slice = slice(n_gens + n_loads, n_gens + n_loads + n_thermal)
    battery_slice = slice(n_gens + n_loads + n_thermal, n_buses)

    # DC power flow uses the susceptance Laplacian: P = B_dc theta with
    # B_dc = -B, so the off-diagonal -b_ij matches the line flow expression below.
    B_dc = -config.B

    theta = cp.Variable((n_buses, N), name="theta")
    P = cp.Variable((n_buses, N), name="P")
    p_gen = cp.Variable((n_gens, N), name="p_gen")
    p_thermal = cp.Variable((n_thermal, N), name="p_thermal")
    p_batt = cp.Variable((n_battery, N), name="p_batt")
    # States carry N+1 points so the terminal condition is expressible.
    temp = cp.Variable((n_thermal, N + 1), name="temp")
    soc = cp.Variable((n_battery, N + 1), name="soc")

    constraints = [
        theta[0, :] == 0,          # angle reference
        P == B_dc @ theta,         # DC power flow
        P[:n_gens, :] == p_gen,
        P[load_slice, :] == -config.load_P,
        P[thermal_slice, :] == -p_thermal,   # heating draws power from the bus
        P[battery_slice, :] == p_batt,       # discharge positive
    ]

    # Generator capacity and ramping (ramp couples consecutive steps only).
    constraints += [
        p_gen >= config.gen_P_min[:, None],
        p_gen <= config.gen_P_max[:, None],
        cp.diff(p_gen, axis=1) >= config.gen_R_min[:, None],
        cp.diff(p_gen, axis=1) <= config.gen_R_max[:, None],
    ]

    # Flexible-device power boxes.
    constraints += [
        p_thermal >= config.thermal_p_min[:, None],
        p_thermal <= config.thermal_p_max[:, None],
        p_batt >= config.battery_p_min[:, None],
        p_batt <= config.battery_p_max[:, None],
    ]

    # Battery: SOC dynamics, bounds, initial and terminal conditions.
    constraints += [
        soc[:, 1:] == soc[:, :-1] - dt * p_batt,
        soc >= config.battery_q_min[:, None],
        soc <= config.battery_q_max[:, None],
        soc[:, 0] == config.battery_q0,
        soc[:, N] == config.battery_qT,
    ]

    # Thermal: first-order envelope dynamics and the comfort band. The band is
    # imposed on the states the control can actually influence (t = 1..N), since
    # temp[:, 0] is pinned to T0.
    decay = 1.0 - config.thermal_mu / config.thermal_c
    gain = config.thermal_eta / config.thermal_c
    ambient = config.thermal_mu / config.thermal_c
    constraints += [
        temp[:, 1:] == (cp.multiply(decay[:, None], temp[:, :-1])
                        + cp.multiply(gain[:, None], p_thermal)
                        + cp.multiply(ambient[:, None], config.thermal_T_amb)),
        temp[:, 0] == config.thermal_T0,
        temp[:, 1:] >= config.thermal_T_min,
        temp[:, 1:] <= config.thermal_T_max,
    ]

    # Line flows: f_ij = -B_ij (theta_i - theta_j) = b_ij (theta_i - theta_j).
    for i, j in config.lines:
        flow = -config.B[i, j] * (theta[i, :] - theta[j, :])
        constraints.append(cp.abs(flow) <= config.line_flow_limits[i, j])

    cost = cp.sum(cp.multiply(config.gen_cost_alpha[:, None], cp.square(p_gen))
                  + cp.multiply(config.gen_cost_beta[:, None], p_gen))

    problem = cp.Problem(cp.Minimize(cost), constraints)

    start = time.perf_counter()
    problem.solve(solver=cp.CLARABEL)
    runtime = time.perf_counter() - start

    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(
            f"Centralized dynamic OPF did not solve: status {problem.status!r}. "
            "Check the config for an infeasible operating point (battery SOC "
            "bounds vs battery_q0/battery_qT, thermal comfort band reachability "
            "given thermal_eta/thermal_mu, and line_flow_limits vs peak demand)."
        )

    # States are solved on N+1 points. Store indices 1..N, the states the
    # constraints govern, so stored index t is the state *at* step t and lines
    # up with the per-step data (load_P, thermal_T_amb, the comfort band). Index
    # 0 is the given initial condition and carries no decision.
    trajectory = Trajectory({
        "p": np.asarray(P.value).T,
        "theta": np.asarray(theta.value).T,
        "soc": np.asarray(soc.value)[:, 1:].T,
        "temp": np.asarray(temp.value)[:, 1:].T,
    })

    return SolveResult(
        trajectory=trajectory,
        obj=float(problem.value), # type: ignore
        runtime=runtime,
        p_residual=None,
        s_residual=None,
        convergence=None,
    )
