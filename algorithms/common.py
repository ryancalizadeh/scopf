import os
import logging
import queue
import threading
import multiprocessing
import numpy as np
import cvxpy as cp
from scipy.linalg import cho_factor, cho_solve
from typing import Dict
from Config import Config
from Trajectory import Trajectory
from Proxable import Proxable

logger = logging.getLogger(__name__)


# A line flow this far over its limit after the analytic projection sends the
# timestep to the constrained QP instead.
_LINE_TOL = 1e-9

# Solver tolerances for the prox QPs. A projection's variable error grows like
# the square root of the objective gap, so CLARABEL's default 1e-8 leaves the
# projections accurate to only ~1e-4 (and not idempotent at that level);
# 1e-10 brings them to ~1e-6 for a few percent more time per solve.
_QP_TOL = 1e-10


def _solve_qp(problem: cp.Problem, what: str) -> None:
    problem.solve(solver=cp.CLARABEL, tol_gap_abs=_QP_TOL, tol_gap_rel=_QP_TOL, tol_feas=_QP_TOL)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"{what}: QP status {problem.status!r}")


def thermal_shift(config: Config) -> np.ndarray:
    """
    Offset added to z["temp"] to recover degrees C: thermal_T0 on the thermal
    columns, zero elsewhere. The stored state is shifted so the temperatures
    are of a size with p and soc under the single ADMM rho (a raw ~20 C state
    would dominate the residual norms).
    """
    shift = np.zeros(config.n_buses)
    start = config.n_gens + config.n_loads
    shift[start:start + config.n_thermal] = config.thermal_T0
    return shift


class Network(Proxable):
    """
    f(x) =
        ind_{P_t = B_dc theta_t, t = 0..N-1}(x) +
        ind_{|B_ij (theta_{i,t} - theta_{j,t})| <= F_ij,     (i,j) in E}(x)

    Indicator of an affine set intersected with a polyhedron, so the prox is a
    projection and rho plays no part. Projecting (p, theta) onto p = B_dc theta
    and eliminating p leaves an unconstrained least squares in theta,

        (B_dc^T B_dc + I) theta = B_dc p_z + theta_z,

    whose matrix is SPD (B_dc is a Laplacian, singular only on the constant
    angle, and the +I covers that), so one Cholesky factor serves every
    timestep and every ADMM iteration. When that projection satisfies the line
    limits it is also the projection onto the intersection; the few timesteps
    where it does not are re-solved with a one-timestep QP.

    The angle reference is deliberately left free. Pinning theta_0 = 0 inside a
    projection is not gauge-neutral: the proximity term pays for the pin by
    trading away the fit in p, so ADMM would converge to the wrong dispatch.
    soc and temp pass through untouched.
    """
    def __init__(self, config: Config):
        n = config.n_buses
        self.B_dc = -config.B
        self._cho = cho_factor(self.B_dc.T @ self.B_dc + np.eye(n))

        # Line flows f = A theta with rows -B_ij (e_i - e_j), as in centralized.py.
        self.A = np.zeros((len(config.lines), n))
        self.F = np.zeros(len(config.lines))
        for l, (i, j) in enumerate(config.lines):
            self.A[l, i] = -config.B[i, j]
            self.A[l, j] = config.B[i, j]
            self.F[l] = config.line_flow_limits[i, j]

        # Fallback QP in theta alone, parameterised so it is canonicalised once.
        self._qp_p_z = cp.Parameter(n)
        self._qp_theta_z = cp.Parameter(n)
        self._qp_theta = cp.Variable(n)
        self._qp = cp.Problem(
            cp.Minimize(cp.sum_squares(self.B_dc @ self._qp_theta - self._qp_p_z)
                        + cp.sum_squares(self._qp_theta - self._qp_theta_z)),
            [cp.abs(self.A @ self._qp_theta) <= self.F],
        )
        assert self._qp.is_dpp()
        self.n_qp_solves = 0

    def prox(self, z: Trajectory, rho: float = 1.0) -> Trajectory:
        rhs = z["p"] @ self.B_dc + z["theta"]                       # (N, n)
        theta = cho_solve(self._cho, rhs.T).T
        violated = np.flatnonzero((np.abs(theta @ self.A.T) > self.F + _LINE_TOL).any(axis=1))
        for t in violated:
            self._qp_p_z.value = z["p"][t]
            self._qp_theta_z.value = z["theta"][t]
            _solve_qp(self._qp, f"Network line-limit projection at t={t}")
            theta[t] = self._qp_theta.value
        self.n_qp_solves += len(violated)

        ret = z.copy()
        ret["theta"] = theta
        ret["p"] = theta @ self.B_dc
        return ret


class Generator(Proxable):
    """
    for generator with key i
    f(x) =
        sum_{t in T}(c_i(P_i(t))) +
        ind_{P^min_i <= p_{i,t} <= P^max_i}(x) +
        ind_{R^min_i <= p_{i,t+1} - p_{i,t} <= R^max_i}(x)

    The one prox that is not a projection: it carries the quadratic cost, so
    rho matters. The prox term is written rho/2 ||p||^2 - <rho z, p> with rho
    and rho*z as the parameters: a parameter times a parameter-free expression
    is DPP, so cvxpy canonicalises the QP once and every call is a data update.
    (rho/2 ||p - z||^2 with both as parameters is parameter times parameter and
    would be re-canonicalised on every call.)
    """

    def __init__(self, config: Config, bus_index: int, gen_index: int):
        self.bus_index = bus_index
        N, k = config.N, gen_index
        p = cp.Variable(N)
        self._rho = cp.Parameter(nonneg=True)
        self._rho_z = cp.Parameter(N)
        objective = (float(config.gen_cost_alpha[k]) * cp.sum_squares(p)
                     + float(config.gen_cost_beta[k]) * cp.sum(p)
                     + 0.5 * self._rho * cp.sum_squares(p) - self._rho_z @ p)
        constraints = [
            p >= config.gen_P_min[k],
            p <= config.gen_P_max[k],
            cp.diff(p) >= config.gen_R_min[k],
            cp.diff(p) <= config.gen_R_max[k],
        ]
        self._p = p
        self._prob = cp.Problem(cp.Minimize(objective), constraints)
        assert self._prob.is_dpp()

    def prox(self, z: Trajectory, rho: float = 1.0) -> Trajectory:
        self._rho.value = rho
        self._rho_z.value = rho * z["p"][:, 0]
        _solve_qp(self._prob, f"Generator prox at bus {self.bus_index}")
        ret = z.copy()
        ret["p"][:, 0] = self._p.value
        return ret


class Load(Proxable):
    """
    for load with key i (positive p is power draw)
    f(x) = ind_{P_{i,t} = -P^load_{i,t}}(x)
    Barely even a projection. Just set the power values to the value in P^load.
    """

    def __init__(self, config: Config, bus_index: int, load_index: int):
        self.bus_index = bus_index
        self._p = -config.load_P[load_index, :]

    def prox(self, z: Trajectory, rho: float = 1.0) -> Trajectory:
        ret = z.copy()
        ret["p"][:, 0] = self._p
        return ret


class Battery(Proxable):
    """
    for battery with key i
    f(x) =
        ind_{q_{i,t+1} = q_{i,t} - dt p_{i,t}}(x) +
        ind_{q^min_i <= q_{i,t} <= q^max_i}(x) +
        ind_{q_{i,0} = q0_i}(x) +
        ind_{q_{i,N} = qT_i}(x) +
        ind_{p^min_i <= p_{i,t} <= p^max_i}(x)

    Projection in joint (p, q) space onto the SOC dynamics (affine) and the
    boxes on q and p. The boxes keep it from being a matrix solve, so it is a
    parameterised QP in 2N variables; rho is ignored. Stored q[t] is the SOC
    at step t, i.e. after p[t] is applied, with q0 the given initial state.
    """

    def __init__(self, config: Config, bus_index: int, battery_index: int):
        self.bus_index = bus_index
        N, dt, k = config.N, config.dt, battery_index
        p = cp.Variable(N)
        q = cp.Variable(N)
        self._p_z = cp.Parameter(N)
        self._q_z = cp.Parameter(N)
        constraints = [
            q[0] == config.battery_q0[k] - dt * p[0],
            q[1:] == q[:-1] - dt * p[1:],
            q >= config.battery_q_min[k],
            q <= config.battery_q_max[k],
            q[N - 1] == config.battery_qT[k],
            p >= config.battery_p_min[k],
            p <= config.battery_p_max[k],
        ]
        self._p, self._q = p, q
        self._prob = cp.Problem(
            cp.Minimize(cp.sum_squares(p - self._p_z) + cp.sum_squares(q - self._q_z)),
            constraints,
        )
        assert self._prob.is_dpp()

    def prox(self, z: Trajectory, rho: float = 1.0) -> Trajectory:
        self._p_z.value = z["p"][:, 0]
        self._q_z.value = z["soc"][:, 0]
        _solve_qp(self._prob, f"Battery prox at bus {self.bus_index}")
        ret = z.copy()
        ret["p"][:, 0] = self._p.value
        ret["soc"][:, 0] = self._q.value
        return ret


class Thermal(Proxable):
    """
    for thermal with key i (positive p is power draw)
    f(x) =
        ind_{T_{i,t+1} = (1 - mu_i/c_i) T_{i,t} + (eta_i/c_i) p_{i,t} + (mu_i/c_i) Tamb_{i,t}}(x) +
        ind_{T^min_{i,t} <= T_{i,t} <= T^max_{i,t}}(x) +
        ind_{T_{i,0} = T0_i}(x) +
        ind_{p^min_i <= p_{i,t} <= p^max_i}(x)

    Same shape as Battery, in joint (p, T) space. Two conventions to keep
    straight:
      - the bus sees P = -p (heating draws power), so p is negated on the way
        in and on the way out. The reflection is an isometry, so projecting in
        device coordinates and negating back is the projection in bus
        coordinates; drop either negation and the heater cools.
      - the stored state is T - T0 (see thermal_shift), so the dynamics carry
        a constant (decay - 1) T0 and start from zero.
    """

    def __init__(self, config: Config, bus_index: int, thermal_index: int):
        self.bus_index = bus_index
        N, k = config.N, thermal_index
        decay = 1.0 - config.thermal_mu[k] / config.thermal_c[k]
        gain = config.thermal_eta[k] / config.thermal_c[k]
        ambient = config.thermal_mu[k] / config.thermal_c[k]
        T0 = float(config.thermal_T0[k])
        drive = ambient * config.thermal_T_amb[k] + (decay - 1.0) * T0        # (N,)

        p = cp.Variable(N)
        T = cp.Variable(N)
        self._p_z = cp.Parameter(N)
        self._T_z = cp.Parameter(N)
        constraints = [
            T[0] == gain * p[0] + drive[0],
            T[1:] == decay * T[:-1] + gain * p[1:] + drive[1:],
            T >= config.thermal_T_min[k] - T0,
            T <= config.thermal_T_max[k] - T0,
            p >= config.thermal_p_min[k],
            p <= config.thermal_p_max[k],
        ]
        self._p, self._T = p, T
        self._prob = cp.Problem(
            cp.Minimize(cp.sum_squares(p - self._p_z) + cp.sum_squares(T - self._T_z)),
            constraints,
        )
        assert self._prob.is_dpp()

    def prox(self, z: Trajectory, rho: float = 1.0) -> Trajectory:
        self._p_z.value = -z["p"][:, 0]
        self._T_z.value = z["temp"][:, 0]
        _solve_qp(self._prob, f"Thermal prox at bus {self.bus_index}")
        ret = z.copy()
        ret["p"][:, 0] = -self._p.value # type: ignore
        ret["temp"][:, 0] = self._T.value
        return ret

class BusBehaviours(Proxable):
    """
    A class implementing projections onto the behaviours of a set of buses, each with its own behaviour (e.g., generator, load, etc.)
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


def _process_worker_main(conn, config: Config, indices: list[int]) -> None:
    """
    Entry point of a bus-projection worker process. Builds the behaviours for
    its own buses (casadi objects never cross the process boundary), then
    serves ("prox", [single-bus Trajectory, ...], rho) requests
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


def rho_fixed(value: float = 2.0):
    """
    Constant rho, the setting ADMM's convergence theory covers. On the dynamic
    DC-OPF split (Network / BusBehaviours) rho = 2 converges at every size
    tried, n_buses = 5..75, within ~2x of the best fixed value for that size:
        buses   iterations   (best fixed rho / its)
          5        197          1 / 129
          8        820          5 / 362
         15        136          2 / 136
         45        781          2 / 781    (1: 1168, 3: 873, 5: 819, 10: 1528, 20+: none)
         75        172          2 / 172    (5: 476, 10: 941)
    Above the best value iterations grow about linearly with rho: r converges
    quickly but s = rho |dz| does not, since the cost's pull on z fades as
    1/rho, and at rho = 50 the dispatch is still 0.18 off after 2000
    iterations while the objective gap reads 1e-5.

    Residual balancing (rho_heuristic) helps where the best rho is not 2
    (5, 15, 75 buses: one or two early changes to rho = 1) and fails on the
    45-bus config, where rho = 2 is already balanced and the residuals
    oscillate with a ~40-iteration period whose s/r swing exceeds mu = 10:
    every trigger is a false positive, each halving kicks r up 5x, and the
    rule locks onto the oscillation. mu >= 50, or the rule on smoothed
    residuals, never triggers there and reduces to fixed 2; tau = 1.5 is
    worse. The oscillation itself comes from the one binding line (26,34),
    a bridge behind which sit two loads and three thermal units and nothing
    else: during the 6:30-8:00 pre-heat their aggregate draw is rationed to
    the line limit, a coupling among three g-blocks that only the dual
    (bus prices) can enforce, and the thermals' price response is zero at
    their box bound and bang-bang off it. The slow mode is the pocket price
    (u at bus 26) while every primal quantity converges early; with the
    limits relaxed the same config converges 3x faster (253 vs 892 its),
    while removing the cost's flat directions with a small quadratic on
    flexible power does not help (787 its).

    Geometric ramps (rho_geometric, rho_for_size) capped at 20-100 freeze the
    iterate before the dual converges (small r, large s, dispatch 0.1-0.2 off)
    and uncapped ones never meet the stopping rule.
    """
    def schedule(iteration, rho_prev, r, s):
        return float(value)
    return schedule


def rho_heuristic(iteration, rho_prev, r, s, tau=2, mu=10):
    """
    Residual-balancing heuristic.
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
    Geometric ramp: rho = min(rho_0 * growth ** iteration, rho_max).
    """
    def schedule(iteration, rho_prev, r, s):
        return float(min(rho_0 * growth ** iteration, rho_max))
    return schedule


def rho_for_size(n_buses: int, rho_0: float = 2.0, max_growth: float = 0.007, growth_scale: float = 0.18):
    """
    Size-dependent geometric ramp: growth = 1 + min(max_growth, growth_scale / n_buses),
    uncapped.
        buses   growth   iterations to r < 1e-4   dispatch error
          8     1.007            650                 0.003 %
         15     1.007           1660                 0.3 %
         15     1.02             590                 1.7 %
         15     1.05             560                12 %
         25     1.007           1310                 0.2 %
         45     1.007           1480                 2.1 %
         45     1.004           2560                 0.11 %

    Larger networks suspected to need slower damping to stay unbiased, hence the 1/n_buses
    rule
    """
    growth = 1.0 + min(max_growth, growth_scale / max(int(n_buses), 1))
    return rho_geometric(rho_0=rho_0, growth=growth, rho_max=float("inf"))


def make_bus_behaviour(config: Config, bus_index: int) -> Proxable:
    # TODO make this more flexible (e.g., config["bus_types"] = ["gen", "load", "load", "empty", ...])
    """
    Behaviour of a single bus in the [gens | loads | thermal | batteries]
    layout of centralized.py. Config sizes n_battery as the remainder, so the
    four blocks tile every bus and there is no empty case.
    """
    n_gens, n_loads, n_thermal = config.n_gens, config.n_loads, config.n_thermal
    if bus_index < n_gens:
        return Generator(config, bus_index=bus_index, gen_index=bus_index)
    if bus_index < n_gens + n_loads:
        return Load(config, bus_index=bus_index, load_index=bus_index - n_gens)
    if bus_index < n_gens + n_loads + n_thermal:
        return Thermal(config, bus_index=bus_index, thermal_index=bus_index - n_gens - n_loads)
    return Battery(config, bus_index=bus_index, battery_index=bus_index - n_gens - n_loads - n_thermal)


def make_bus_behaviours(config: Config, parallel: "bool | str" = False, n_workers: int | None = None) -> Proxable:
    # TODO make this more flexible (e.g., config["bus_types"] = ["gen", "load", "load", "empty", ...])
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


def _excess(values: np.ndarray, lower, upper) -> float:
    """Largest amount by which values leave [lower, upper]; 0.0 when inside."""
    if values.size == 0:
        return 0.0
    return float(max(0.0, np.max(lower - values), np.max(values - upper)))


def _mismatch(a: np.ndarray, b) -> float:
    return float(np.abs(a - b).max()) if a.size else 0.0


def check_solution(z: Trajectory, config: Config) -> Dict[str, float]:
    """
    Objective and constraint violations of a trajectory in the base-trajectory
    layout (every key of width n_buses, temp shifted by thermal_T0). The checks
    mirror test_centralized.check_solution but return magnitudes, 0.0 meaning
    satisfied, so an ADMM iterate can be judged without raising.
    """
    dt = config.dt
    n_gens, n_loads, n_thermal = config.n_gens, config.n_loads, config.n_thermal
    gen_cols = slice(0, n_gens)
    load_cols = slice(n_gens, n_gens + n_loads)
    thermal_cols = slice(n_gens + n_loads, n_gens + n_loads + n_thermal)
    battery_cols = slice(n_gens + n_loads + n_thermal, config.n_buses)

    p, theta = z["p"], z["theta"]
    p_gen = p[:, gen_cols]
    p_thermal = -p[:, thermal_cols]           # heating draws power from the bus
    p_batt = p[:, battery_cols]
    soc = z["soc"][:, battery_cols]
    temp = z["temp"][:, thermal_cols] + config.thermal_T0

    checks: Dict[str, float] = {}
    checks["objective"] = float(np.sum(config.gen_cost_alpha * p_gen ** 2 + config.gen_cost_beta * p_gen))

    # Network: DC power flow and (implied) lossless balance.
    B_dc = -config.B
    checks["network_residual"] = _mismatch(p, theta @ B_dc)
    checks["power_balance_residual"] = float(np.abs(p.sum(axis=1)).max())

    # Generators: capacity and ramping.
    checks["gen_capacity"] = _excess(p_gen, config.gen_P_min, config.gen_P_max)
    checks["gen_ramp"] = _excess(np.diff(p_gen, axis=0), config.gen_R_min, config.gen_R_max)

    # Fixed loads served exactly.
    checks["load_mismatch"] = _mismatch(-p[:, load_cols], config.load_P.T)

    # Thermal: dynamics (stored index t is the state at step t, driven by the
    # control and ambient at t; index 0 follows from T0), comfort band, power box.
    decay = 1.0 - config.thermal_mu / config.thermal_c
    gain = config.thermal_eta / config.thermal_c
    ambient = config.thermal_mu / config.thermal_c
    T_amb = config.thermal_T_amb.T
    if n_thermal:
        predicted = np.vstack([
            decay * config.thermal_T0 + gain * p_thermal[0] + ambient * T_amb[0],
            decay * temp[:-1] + gain * p_thermal[1:] + ambient * T_amb[1:],
        ])
        checks["thermal_dynamics"] = _mismatch(temp, predicted)
    else:
        checks["thermal_dynamics"] = 0.0
    checks["thermal_band"] = _excess(temp, config.thermal_T_min.T, config.thermal_T_max.T)
    checks["thermal_box"] = _excess(p_thermal, config.thermal_p_min, config.thermal_p_max)

    # Battery: dynamics (same indexing), SOC bounds, terminal state, power box.
    if config.n_battery:
        predicted_soc = np.vstack([
            config.battery_q0 - dt * p_batt[0],
            soc[:-1] - dt * p_batt[1:],
        ])
        checks["soc_dynamics"] = _mismatch(soc, predicted_soc)
        checks["soc_terminal"] = _mismatch(soc[-1], config.battery_qT)
    else:
        checks["soc_dynamics"] = 0.0
        checks["soc_terminal"] = 0.0
    checks["soc_bounds"] = _excess(soc, config.battery_q_min, config.battery_q_max)
    checks["battery_box"] = _excess(p_batt, config.battery_p_min, config.battery_p_max)

    # Line flows against their limits.
    worst_excess, worst_util = 0.0, 0.0
    for i, j in config.lines:
        flow = np.abs(-config.B[i, j] * (theta[:, i] - theta[:, j])).max()
        limit = config.line_flow_limits[i, j]
        worst_excess = max(worst_excess, flow - limit)
        worst_util = max(worst_util, flow / limit)
    checks["line_limit"] = float(max(0.0, worst_excess))
    checks["line_utilization"] = float(worst_util)

    return checks
