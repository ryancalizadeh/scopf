# Work report: transient-stability-constrained OPF (centralized + ADMM)

Covers the sessions from 2026-09-04 to 2026-09-11. Each section gives the task as posed, the
obstacles met, and what was done about them. The last section reviews the codebase as it stands
(uncommitted changes included) and lists what is still open.

## 1. Tasks, obstacles, resolutions

### 1.1 Centralized TSC-OPF (`algorithms/centralized.py`)

**Task.** Complete the centralized transient-stability-constrained OPF: include the generator DAE
from `gen_diff_eqs` / `gen_alg_eqs`, discretize it exactly (the differential equations are linear
in the states), set up the internal EMF `E`, and add `|delta_i - delta_coi| <= 100 deg`. Put the
discretization in reusable helpers.

**Obstacles and resolutions.**
- `gen_diff_eqs` had a damping-sign typo (`+D/M`, unstable) and `gen_alg_eqs` referenced an
  undefined `E`. Fixed the sign to `-D/M`; `E` became a per-generator variable, constant over the
  horizon, constrained `E >= 0` to rule out the `(-E, delta + pi)` mirror solution.
- Nothing disturbed the system, so the trajectory was flat and the constraint vacuous. A load-step
  disturbance was added first (later replaced by a line fault, section 1.7).
- The uncommitted line-flow loop built a constant constraint for every non-adjacent bus pair, which
  casadi rejects; pairs without a line are skipped.
- The random `Config` generator produced OPF cases that were infeasible even statically: total load
  exceeded total `P_max`, and line reactances of 0.1-0.2 p.u. on a tree-like network exhausted the
  generators' reactive limits under the 0.95-1.05 voltage band. Fixed by dividing line impedances by
  about three and scaling loads to `load_fraction = 0.3` of capacity; the user later widened the
  voltage band to 0.85-1.15.

Verified: the exact ZOH map matched the closed form and one-step ODE integration to 1e-14;
forward-simulating the discrete map reproduced the solved trajectories to 1e-12.

### 1.2 ADMM building blocks (`ConstPowerLoad`, `Generator`, `EmptyBus`, `F`)

**Task.** Check that the docstring splitting reproduces the centralized problem, comment on
scalability, implement the four classes, keep `main.py` runnable.

**Obstacles and resolutions.**
- The split was exact only after four additions: the disturbance in the load prox, `omega(0) = 0`,
  the slack angle reference in `F` (otherwise the solution set is a rotation manifold), and voltage
  bounds on load and empty buses (the docstrings had them on generators only).
- `E` is local to the generator projection, not a consensus variable.
- **Thread executor crashed the interpreter.** casadi/IPOPT segfaults when the same `Opti` is solved
  from changing OS threads, even under a lock. `BusBehavioursParallel` pins each bus to a persistent
  worker thread.
- **The residual-balancing rho heuristic fails** on this splitting (limit cycles). The user stated
  it is known not to work here; it was replaced by a geometric ramp (section 1.4).

### 1.3 Speed deviation, separable `F`, process executor

**Tasks.** (a) Switch the consensus variable from absolute rotor speed to the deviation
`omega - omega_s`; (b) exploit the time-separability of `F`; (c) run the bus projections in
processes; (d) choose the executor by size.

**Resolutions.**
- (a) The constant terms vanish (`c = 0`), the steady state is `x = 0`, and the iteration-0 residual
  fell from ~2384 (pure offset) to ~7.7. No change to solutions.
- (b) The cvxpy `F` compiled in 19 s at 45 buses and would have taken about an hour at 405. The new
  `F` projects each fault phase onto the network subspace with a precomputed QR basis (one matrix
  product per phase), keeps only a small `t = 0` SOCP that is solved lazily when a line limit or
  voltage bound is violated, and uses closed forms for `P(0)`, `Q(0)` and the COI polytope. Setup
  5 ms at 45 buses, 0.8 s at 405; the old class is kept as `FCvxpy` for verification.
- (c) `BusBehavioursProcesses`: spawned workers over pipes, each building only its own buses;
  bit-identical to the sequential executor. Per-iteration times (135 buses, N = 8): sequential 296 ms,
  threads 144 ms, processes 75 ms.
- (d) `admm_parallel` uses threads below 40 buses and processes at or above.

### 1.4 rho schedule and stop rule

**Task.** ADMM stalled at 15 buses; the user hypothesised that larger problems need a larger rho.

**Finding.** The level of rho is irrelevant: every prox in the split is a projection except the
small `P(0)` cost term, so fixed rho in {5 ... 1000} gives the same plateau or limit cycle. What
converges is the dual rescaling `u *= rho_old / rho_new` that `admm.py` applies on every rho
change, i.e. an *uncapped* geometric ramp; a cap that binds before the stop rule fires reinstates
the cycle. Larger networks need slower growth to stay unbiased (45 buses: 2.1 % error at growth
1.007, 0.11 % at 1.004).

**Resolution.** `rho_for_size(n_buses)`: uncapped ramp with growth `1 + min(0.007, 0.18 / n_buses)`;
`admm.py` stops on `r < thr` and the unscaled dual step `|dz| = s / rho < thr` (the scaled `s` can
never drop below the threshold with rho ~ 1e5); the threshold is `1e-3 * sqrt(n_buses / 8)`, which
holds the per-entry RMS mismatch constant across sizes (user's request). With this, dispatch errors
were 0.05-0.3 % from 8 to 135 buses on the load-step problem.

### 1.5 Time-domain plots (`plot_dynamics.py`, `simulate.py`)

**Task.** Standalone script reading the latest results and overlaying centralized and ADMM
frequency and angle dynamics with the COI bounds. The user asked that trajectories not be stored;
instead re-simulate from the dispatch.

**Resolution.** `ExperimentResult.dispatch_Q` was added (filled by `experiment_runner`);
`simulate.py` rebuilds the transient from `(P(0), Q(0))` via a least-squares power flow and the
discretized dynamics, reproducing the solvers' trajectories to 1e-9. The plot shades the fault
window and includes a zoomed angle inset.

### 1.6 Line fault contingency

**Task.** Replace the load step with a random line fault at t = 0 cleared at t_clear; would that be
enough to provoke the COI constraint?

**Finding: not on its own.** Three model changes were needed, each confirmed by simulation:
- The machines were far too heavy (`M` 0.12-0.49, i.e. H of 23-92 s). Realistic H of 2-6 s was
  adopted.
- Constant-power loads cannot be served at fault voltages. Loads are constant impedances folded
  into the transient admittance for t >= 1 (constant power at t = 0). Their current is unchanged;
  what is zero for t >= 1 is only the residual device injection, and the trajectory's `S` is zero
  there (user's choice; the true consumption is `|V|^2 conj(y_L)`, see `simulate.load_consumption`).
- The zero-order hold was badly wrong at realistic inertia (84 deg vs 31 deg from an adaptive ODE
  reference at dt = 0.05). An exact first-order hold (`gen_discrete_step_foh`) matches the reference
  within 1-2 deg at dt = 0.05 and is still affine in the decision variables.

Also: faults far from generators barely move the angles, and tripping an arbitrary line usually
islands the network, so the faulted line is drawn from generator-incident non-bridge lines. With
t_clear = 0.6 s (user's choice; 0.7 s is infeasible at 15 buses) the bound is active at 45 and 105
buses and slack at 15.

### 1.7 Voltage bound in `F`; the ADMM/centralized discrepancy on fault cases

**Task.** ADMM no longer converged on fault cases. Add the voltage upper bound to `F`. The user then
suspected the load/shunt absorption into Y made ADMM solve a different problem.

**Resolutions and findings (2026-09-11).**
- `F` and `FCvxpy` now carry `|V_k(0)| <= V_max` (the lower bound is nonconvex and stays in the
  projections). This removed the residual stall; ADMM stops by its criterion again.
- The suspicion was right, though the culprit was the **fault shunt**, not the loads: an audit of the
  ADMM iterate showed every bus-local constraint satisfied to 1e-10 but `I = Y(n) V` violated by
  0.11 p.u., because the 1e3 shunt inside `Y_fault` turns a 1.7e-4 voltage disagreement into a 0.1
  current disagreement. The load admittance (~0.3) is benign. The fault is now **exact**: `V = 0` at
  the faulted bus during the fault, that bus's network row dropped, its device current the
  generator's fault current `E e^{j delta} / (j Xd)`. The fault-step network residual of the ADMM
  iterate fell from 0.11 to 7e-3, the same as the pre-fault steps.
- Scaling `delta`/`omega` in the consensus metric (approved as a remedy for the leverage of `P(0)`
  on the angles under light inertia) was tested and **reverted**: s = 0.3 changed nothing, s <= 0.1
  diverged. `Generator(angle_weight=1.0)` keeps the knob with the negative result in a comment.
- **Remaining cause, unresolved:** the flat (zero-generation) start. The faulted generator's `P(0)`
  is set in the early small-rho iterations by building up ~5 p.u. fault currents from nothing, and the
  rho ramp then freezes it; the dispatch drifts monotonically toward the optimum but stalls at ~50 %
  error with the stop rule satisfied. A straight, feasible, cost-decreasing path to the optimum
  exists, so it is not a local optimum. A warm start from the static economic dispatch fixes the
  15-bus case (0.5 % error, 906 iterations) but not the 45-bus case, whose static dispatch is
  transiently unstable (peak 418 deg): the first projections overshoot and ADMM settles with the
  faulted generator switched off, 13 % over the centralized cost.

### 1.8 ADMM from a flat start: investigation and fix (2026-09-11)

**Task.** From a flat start ADMM converged on fault cases but to a dispatch ~50 % from centralized,
at 20 % higher cost, with the faulted generator's power far too low. Investigate and solve.

**Mechanism, measured rather than inferred.** Pushing the generator prox's target `P(0)` by +0.1
moves its output by 0.003: the projection keeps **3 %** of an economic push. Shifting dispatch
between generators changes every machine's transient by about `T^2 / 2M` (~100x) times the shift,
so the feasible direction that moves `P(0)` is enormous in the rotor states and a Euclidean
projection barely moves along it (IPOPT does not care: Newton steps are curvature-aware; a
first-order splitting sees an ill-conditioned direction). The two observed behaviours follow: at
fixed rho the nonconvex iteration cycles; under any ramp it converges, but the P-consensus dual
`u_P = -2cP/rho -> 0`, so the limit is a pure feasibility method and `P(0)` freezes wherever the
first few hundred iterations put it (from `delta = omega = 0` the prox prefers a small swing, i.e.
a small `P(0)` for the faulted machine). Anderson acceleration converged faster to the *same*
point, which confirms it is a genuine fixed point of the ramped iteration, not a slow transient.

**Levers tried (15-bus flat start; dispatch error vs centralized).**

| lever | outcome |
|---|---|
| start the ramp lower, rho_0 = 0.2 / 0.02 / 0.002 (= cost x10 / x100 / x1000) | 47 % / 50 % / diverges |
| hold rho = 2 for 1500 iterations, then ramp | 46 % (cycles during the hold) |
| slower ramp, growth 1.002 | 51 % |
| scale delta/omega in the consensus metric, s = 0.3 / 0.1 / 0.03 | 50 % / diverges / diverges (reverted) |
| Anderson acceleration, m = 5, safeguarded | 52 %, 13 % fewer iterations |
| fixed rho with proximal damping, (rho, tau) = (2,1), (10,2), (50,5) | cycles, 39-50 % |
| ramp to a cap of 50 or 200, then hold with damping | cycles at r ~ 2e-2, 51-53 % (passed through the optimum once, did not stay) |
| warm start from the static dispatch | 15-bus 0.6 % (start stable); 45-bus start unstable (peak 418 deg), overshoots, faulted gen shut off, +13 % cost |
| **warm start from the static dispatch, stabilised** | **15-bus +0.2 % cost; 45-bus +0.2 %; 105-bus -0.3 %; all feasible** |

**Fix adopted.** `admm_vanilla.warm_start`: (1) ADMM on the static problem (the same split with
`N = 1`, ~110-300 iterations) for the economic dispatch, so the method stays distributed;
(2) `simulate.stabilise_dispatch` bisects the fraction of the faulted generator's output moved to
the others until the simulated peak COI angle is at most 90 deg; (3) the simulated trajectory of
that dispatch is the start. `solve(config, parallel, warm=True)`; `admm_parallel` inherits it.

| case | static it / shift alpha | ADMM it | runtime | cost vs centralized | peak |
|---|---|---|---|---|---|
| 15-bus | 111 / 0 | 810 | 15 s (threads) | +0.22 % | 76.5 deg |
| 45-bus | 111 / 0.27 | 1290 | 62 s (processes) | +0.21 % | 90.6 deg |
| 105-bus | 309 / 0.15 | 2495 | 515 s | -0.30 % | 91.4 deg |

The remaining per-generator discrepancy (45-bus faulted generator 0.30 vs 0.36) lies along the
direction with a tiny cost gradient, which is why the cost gap is 0.2 %; compare the algorithms on
cost and feasibility rather than per-generator dispatch. The 105-bus ADMM point being 0.3 %
*cheaper* than centralized while feasible on the bound is within its 3.6e-3 consensus tolerance and
should be read as "within 0.3 %", not as a better optimum.

## 2. Current state of the codebase

Branch `main`, last commit `8e1a34d synch`. Uncommitted: `Config.py`, `algorithms/centralized.py`,
`algorithms/common.py`, `simulate.py`, `generate_configs.py`, `visualize.py`, `configs.pkl`
(regenerated with the exact fault; the older pickles carried shunt matrices).

| file | role |
|---|---|
| `Config.py` (264 lines) | Random network (seed 42), realistic inertia `M = 2H/omega_s`, loads scaled to `load_fraction` of capacity, fault setup: eligible lines, `fault_line`/`fault_bus`, `Y_transient`/`Y_fault`/`Y_post`, `Y_at(n)`, `is_fault_step(n)`, `network_buses_at(n)`, `load_admittance()`. Defaults `T=2.0, dt=0.1, t_clear=0.6`; `generate_configs.py` uses `T=1.2, dt=0.1` for 5/15/45/105 buses. `load_step_factor` is a legacy no-op (1.0). |
| `algorithms/common.py` (1337) | Bus proxes (`ConstPowerLoad`, `Generator`, `EmptyBus`), `F` (separable, closed-form with lazy SOCP/QP fallbacks, voltage bound included) and `FCvxpy` (reference), executors (`BusBehaviours`, thread-pinned `BusBehavioursParallel`, `BusBehavioursProcesses`), rho schedules (`rho_geometric`, `rho_for_size`, legacy `rho_heuristic`), `make_bus_behaviour(s)`, `check_solution` (still t=0 only, still compares squared voltage to unsquared bounds), DAE helpers (`gen_continuous_matrices`, ZOH and FOH discretizations and steps, `gen_alg_eqs`, `gen_coi_angle`). |
| `algorithms/centralized.py` (207) | Full TSC-OPF in casadi/IPOPT: network per `Y_at(n)` with the faulted bus pinned to `V = 0` during the fault, loads constant-power at t=0 only, FOH dynamics with `P_0 = P(0)`, `E >= 0`, `omega(0) = 0`, COI bound, t=0 limits. Returns the trajectory. |
| `algorithms/admm_vanilla.py` / `admm_parallel.py` | `warm_start` (static ADMM + `stabilise_dispatch` + simulation) then the transient ADMM with `rho_for_size`, threshold `1e-3 sqrt(n/8)`, 10000-iteration cap; `warm=False` gives the old flat start. Parallel driver picks threads/processes at 40 buses. `admm_parallel_relaxed.py` still raises `NotImplementedError` (skipped by the runner). |
| `admm.py` (65) | The user's loop with the `|dz|` stop rule and dual rescaling on rho changes. |
| `simulate.py` | Forward simulation from `(P(0), Q(0))`: least-squares power flow at t=0, linear network solves (exact fault) with an FOH fixed-point iteration; `peak_coi_angle_deg`, `stabilise_dispatch`. |
| `plot_dynamics.py` (194) | Standalone overlay of frequency and COI-relative angle per algorithm from the latest results pickle. |
| `Trajectory.py`, `Proxable.py`, `ExperimentResult.py` (`dispatch_Q`), `experiment_runner.py`, `main.py` (n_runs=1, exp1 only), `visualize.py` | Support and pipeline; unchanged in substance. |

Verified as of the last run: centralized satisfies every constraint to 1e-11 with `V = 0` and
`I = E/Xd` at the faulted bus during the fault; `simulate_dispatch` reproduces it to 1e-10; `F`
matches `FCvxpy` to 6e-5 (15 buses) and 4e-4 (45 buses, reference solver tolerance); the bus
proxes and `F` are idempotent on the centralized solution; the FOH peak is within 2 deg of an RK45
reference at 15 buses and 6 deg at 45 buses at dt = 0.1 (2 deg at dt = 0.05).

## 3. Open items

1. ~~ADMM start point on fault cases~~ resolved by the stabilised warm start (section 1.8). What
   remains is a property of the splitting, not a bug: per-generator dispatch can differ from
   centralized along the low-cost-gradient direction (up to ~10 % on the faulted generator) while
   the cost agrees to ~0.2 %.
2. `check_solution` still compares `|V|^2` against unsquared bounds and checks t=0 only; the reported
   `voltage_residual` is misleading (pre-existing TODO).
3. `admm_parallel_relaxed` is not implemented.
4. The 405-bus config was never verified for feasibility; the sweep currently stops at 105.
5. Scratch verification scripts live in the session scratchpad, not in the repo; the checks listed
   above are not part of a test suite.
