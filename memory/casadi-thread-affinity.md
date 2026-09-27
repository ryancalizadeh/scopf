---
name: casadi-thread-affinity
description: "casadi Opti/IPOPT problems segfault when the same instance is solved from different OS threads over time; the thread executor pins buses to threads and the process executor (BusBehavioursProcesses, default for admm_parallel since 2026-09-08) sidesteps it"
metadata: 
  node_type: memory
  type: project
  originSessionId: cceed7c1-45e2-4f97-be5e-cdbd279bdfc6
  modified: 2026-09-08T20:41:37.153Z
---

Observed 2026-09-06 with casadi 3.8.0 (bundled IPOPT/MUMPS) on Windows: the `Generator` projection (Opti with ~160 vars, trig constraints) crashes the interpreter (exit 139) after a few calls when a `ThreadPoolExecutor` hands the same behaviour to varying threads. Serialising the calls with a lock does NOT help; `OMP_NUM_THREADS=1` does NOT help; a single worker thread or a fixed bus-to-thread assignment (concurrent, different Opti per thread) works.

**Why:** the failure is thread-affinity of the solver instance, not concurrency of solves, so any casadi-based parallel executor must keep each Opti on one thread, or use processes with per-process construction.

**How to apply:** `algorithms/common.py` has three executors selected by `make_bus_behaviours(config, parallel=...)`: `False` sequential, `True`/`"threads"` = `BusBehavioursParallel` (persistent `_AffinityWorker` threads, round-robin buses), `"processes"` = `BusBehavioursProcesses` (spawned worker processes, one Pipe each, buses built inside the worker via `make_bus_behaviour`; scripts need the `__main__` guard). `admm_parallel` uses processes. Measured ms/iteration with the closed-form F (2026-09-08, 8 cores): 15-bus N=20: seq 33 / threads 19 / proc 33; 45-bus N=20: 94 / 59 / 54; 135-bus N=8: 296 / 144 / 75. Processes win from ~45 buses up; per-iteration time is bounded by the slowest generator projection, so the gain grows with buses per worker. Threads only ever gave 1.5-2x (interpreter lock). See [[random-configs-infeasible]] and [[rho-schedule-preference]].
