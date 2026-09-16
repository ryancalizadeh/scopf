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

# (MX / SX / DM); the arithmetic inside them is +, *, ca.cos and ca.sin only.
Scalar = Any
from Trajectory import Trajectory
from Proxable import Proxable

logger = logging.getLogger(__name__)


_IPOPT_QUIET = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}


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
    # TODO Update this with the new problem

    def __init__(self, config: Config, bus_index: int, load_index: int, max_iter=100, tol=1e-8):
        raise NotImplementedError("ConstPowerLoad.prox is not implemented yet")

    def prox(self, z: Trajectory, rho: float = 1.0) -> Trajectory:
        raise NotImplementedError("ConstPowerLoad.prox is not implemented yet.")

class Generator(Proxable):
    # TODO Update this with the new problem
    """
    A class implementing the prox operator on the indicator function of the behaviour set of a generator (i.e. projection onto the behaviour set).
    """

    def __init__(self, config: Config, bus_index: int, gen_index: int, max_iter=500, tol=1e-8):
        raise NotImplementedError("Generator.prox is not implemented yet")

    def prox(self, z: Trajectory, rho: float = 1.0) -> Trajectory:
        raise NotImplementedError("Generator.prox is not implemented yet. It should project onto the behaviour set of a generator.")


class EmptyBus(Proxable):
    # TODO Update this with the new problem

    def __init__(self, config: Config, bus_index: int):
        raise NotImplementedError("EmptyBus.prox is not implemented yet")

    def prox(self, z: Trajectory, rho: float = 1.0) -> Trajectory:
        raise NotImplementedError("EmptyBus.prox is not implemented yet. It should project onto the behaviour set of an empty bus.")

class F(Proxable):
    # TODO Update this with the new problem

    def __init__(self, config: Config, solver: str = "CLARABEL"):
        raise NotImplementedError("F.prox is not implemented yet")

    def prox(self, z: Trajectory, rho: float = 1.0) -> Trajectory:
        raise NotImplementedError("F.prox is not implemented yet. ")


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


def check_solution(z: Trajectory, config: Config) -> Dict[str, float]:
    """
    TODO implement
    """
    raise NotImplementedError("check_solution is not implemented yet")
