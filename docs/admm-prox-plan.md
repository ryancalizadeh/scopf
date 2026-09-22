# Implement the device/network proxes and wire them into ADMM

## Context

The 2026-09-15 nuke (commit `172c24e`) removed the TSCOPF implementations, and the
following `f358d1a` commit added a working centralized dynamic DC-OPF
([algorithms/centralized.py](../algorithms/centralized.py)) but left the distributed path
as stubs. Today nothing in the ADMM path runs:

- `Network`, `Generator`, `Load`, `Battery`, `Thermal` in
  [algorithms/common.py](../algorithms/common.py) are all `raise NotImplementedError`.
- `check_solution` ([common.py:416](../algorithms/common.py#L416)) is a stub.
- `Config.make_base_trajectory` ([Config.py:231](../Config.py#L231)) is a stub.
- `make_bus_behaviour` ([common.py:380](../algorithms/common.py#L380)) still references
  `ConstPowerLoad` / `EmptyBus` — dead names from the nuked problem; it would `NameError`.
- [admm_vanilla.py:21](../algorithms/admm_vanilla.py#L21) calls an undefined `F(config)`.

Goal: implement the five proxes against the same problem `centralized.py` already solves,
then wire them into ADMM with **f = `Network`** (power flow + line limits) and
**g = `BusBehaviours`** (per-bus device behaviour, carrying the generator cost). Success
means `admm_vanilla.solve` reproduces `centralized.solve` on the 8-bus config.

## Findings that shape the design (verified, not assumed)

1. **Trajectory keys must all have width `n_buses`.** `Trajectory.at_bus[i]` slices *every*
   key at column `i`, and with ragged widths it **silently returns a width-0 array instead
   of raising** (verified). `centralized.py` emits `soc` at width `n_battery` and `temp` at
   width `n_thermal`; reusing that layout would hand a battery bus an empty `soc` with no error.

2. **`theta_0 = 0` must NOT be imposed inside `Network.prox`.** Verified on the 8-bus config:
   pinning it changes the projected `p` by **0.055**, while the mean-zero gauge changes it by
   8e-14. In a *projection* the reference is not gauge-neutral — the proximity term
   `½||theta - theta_z||²` pays for the pin and trades away `p`-fit — so ADMM would converge
   to the wrong answer. `centralized.py:78` can pin it safely only because there is no
   objective on `theta` there. **Leave the angle free; the `+I` term makes the system
   nonsingular anyway.**

3. **Line limits are inactive at small n.** Max line utilization from the centralized
   optimum: n=5 → 0.25, n=8 → 0.67, n=15 → 0.67, n=45 → **1.000**. So the analytic fast path
   covers n ≤ 15 entirely and the QP fallback is genuinely rare.

4. **Scales differ by ~250×**: `temp` ≈ 17, `p` ≈ 2.7, `soc` ≈ 0.4, `theta` ≈ 0.07–0.8.
   `Trajectory.norm()` concatenates everything under one scalar `rho`, so `temp` would
   dominate the residual. **Decision: store `temp` shifted by −21 °C** (`thermal_T0`).

5. **There are no empty buses** — `Config.__init__:103` makes `n_battery` the remainder, so
   the four blocks tile `n_buses` exactly. No `EmptyBus` class is needed.

6. `battery_qT == battery_q0` ([Config.py:165](../Config.py#L165)), so the SOC set is nonempty
   (`p ≡ 0` is feasible) and the projection is always well-defined.

## Decisions taken (confirmed with the user)

- Store `temp` shifted by −21 °C; shift back in `check_solution` and when comparing.
- `admm_vanilla` returns **`zs[-1]`** (device-feasible) so the reported objective is meaningful.
- `SolveResult` gains an optional `checks: dict | None = None`; `p_residual`/`s_residual`
  become the true ADMM residuals `rs[-1]` / `ss[-1]`.

## Trajectory schema

Four keys, each `(N, n_buses)`:

| key | meaning |
|---|---|
| `p` | bus net injection `P_{i,t}` (sign convention exactly as `centralized.py:80-83`) |
| `theta` | bus voltage angle |
| `soc` | battery SOC; **zeros** at non-battery columns |
| `temp` | thermal state **minus 21 °C**; **zeros** at non-thermal columns |

The shared per-bus coordinate is the **bus net injection**, not a device-local variable —
`Network` constrains `P`, so `P` must be the consensus variable. Each device maps internally:
gen `P_i = p_gen`, load `P_i = -load_P`, thermal `P_i = -p_thermal` (**negate on read and
write**), battery `P_i = p_batt` (discharge positive).

States are stored on indices 1..N as `centralized.py:152-157` already does, matching the
existing memory note.

## Implementation

### 1. `Config.make_base_trajectory` ([Config.py:231](../Config.py#L231))

Return the 4-key schema: loads at their exact `-load_P.T`, generators splitting total load
evenly (clipped to `[P_min, P_max]`), `soc` = `battery_q0`, `temp` = 0 (i.e. `T0` shifted),
`theta` = zeros. Do **not** use all-zeros for `soc`/`temp` — that starts both states far
outside their feasible sets.

### 2. `Network.prox` — f, the analytic projection

Per timestep, projecting `(p, theta)` jointly and eliminating `p = B_dc theta` gives an
unconstrained least-squares in `theta`:

```
M theta = B_dc p_z + theta_z,      M := B_dc^T B_dc + I     (B_dc symmetric)
```

`M` is SPD (min eigenvalue exactly 1). **One `scipy.linalg.cho_factor(M)` in `__init__`,
reused for every timestep and every ADMM iteration.** Verified against a joint-space cvxpy
solve to 1e-14 and verified idempotent to 1e-14. Vectorized over the full horizon:

```python
rhs   = z["p"] @ self.B_dc + z["theta"]        # (N, n)
theta = cho_solve(self._cho, rhs.T).T
p     = theta @ self.B_dc
```

Then check line flows `theta @ A.T` against `F`; for the rare timesteps that violate, fall
back to a **single DPP-parameterized one-timestep cvxpy QP** built once in `__init__`
(parameters `p_z`, `theta_z`; `rho` does not appear — the prox of an indicator is a
projection, so `rho` is ignored). Rows of `A`: `-B[i,j]*(e_i - e_j)` for each
`(i,j) in config.lines`; `F_l = line_flow_limits[i,j]`.

`soc` and `temp` are **untouched** (`f` does not constrain them): `ret = z.copy()`, then
overwrite only `p` and `theta`. Do not zero them.

### 3. Device proxes — g

All receive a **width-1** Trajectory from `at_bus[i]`, so every array is `(N,1)` and the
column index is `0`, not `bus_index`. Each leaves the columns it does not own untouched.

- **`Load`** — no solver: set `p[:,0] = -config.load_P[load_index,:]`. `rho` ignored.
- **`Generator`** — the only non-projection (it carries the cost). Small QP over `N=96` vars,
  tridiagonal via ramping. `rho` **matters**, so follow the old `FCvxpy` DPP pattern: write
  the prox term parameter-affine as `rho/2·||p||² − <rho·z_p, p>` with `rho` a
  `cp.Parameter(nonneg=True)` and `rho*z_p` a parameter. Writing
  `rho/2*cp.sum_squares(p - z)` directly is **not DPP** and re-canonicalises every call.
  Solver CLARABEL (matches `centralized.py`; ~1 ms/solve). Not casadi/IPOPT — that is the
  source of the documented thread-affinity segfault ([common.py:176-183](../algorithms/common.py#L176-L183)).
- **`Battery`** — projection onto SOC dynamics + boxes + `q_{N-1} = qT`, in joint `(p,q)`
  space. Boxes make it non-analytic → small parameterized cvxpy QP (2N vars). `rho` ignored.
  Indexing per [test_centralized.py:92-98](../test_centralized.py#L92-L98): `soc[0] = q0 - dt*p[0]`,
  `soc[t] = soc[t-1] - dt*p[t]`.
- **`Thermal`** — same shape, joint `(p, T)`. **The sign is the trap**: read
  `p_th = -z["p"][:,0]`, write back `ret["p"][:,0] = -p_th`. Because `P = -p_th` is an
  isometry (a reflection), projecting in device coordinates and negating back equals
  projecting in bus coordinates — but only if *both* negations are applied. Omitting either
  gives a heater that cools, and it still converges, to a wrong answer. Dynamics per
  [test_centralized.py:80-86](../test_centralized.py#L80-L86) with `decay/gain/ambient`. Fold the
  −21 °C shift into the band, `T0`, and the constant `(decay-1)*21` term once in `__init__`.

### 4. `make_bus_behaviour` rewrite ([common.py:380](../algorithms/common.py#L380))

Four branches over the `[gens | loads | thermal | batteries]` blocks, with
`Thermal` before `Battery`. Delete the `ConstPowerLoad` / `EmptyBus` references — no empty
branch is reachable.

### 5. `check_solution(z, config) -> Dict[str, float]`

Port [test_centralized.py:44-108](../test_centralized.py#L44-L108) from asserts to magnitudes
(`0.0` = satisfied), unshifting `temp` by +21 first. Must supply the keys
`admm_vanilla` consumes — `objective`, `network_residual`, `power_balance_residual` — plus
per-constraint violations: gen min/max/ramp, load mismatch, soc dynamics/bounds/terminal,
thermal dynamics/band/box, battery box, line limits, max line utilization.

### 6. Wire into [admm_vanilla.py](../algorithms/admm_vanilla.py)

`f = Network(config)` (the import on line 8 is already right); drop the `# @claude` comment.
Return `zs[-1]`; set `p_residual=rs[-1]`, `s_residual=ss[-1]`, and pass `checks` through the
new `SolveResult.checks` field ([base.py](../algorithms/base.py)). The existing `close()`
handling (lines 34-37) is already correct and needs no change — `Network` has no workers, so
do not give it a `close`. Leave `admm_parallel_relaxed` as its stub;
[experiment_runner.py:49](../experiment_runner.py#L49) already skips `NotImplementedError`.

## Build order

Each step is testable before the next, and step 2 settles the question everything depends on.

1. `make_base_trajectory` + the schema.
2. `Network`, analytic path only — unit-test projection/idempotence/`rho`-invariance now.
3. `Load` → `Battery` → `Generator` (establishes the DPP `rho` pattern) → `Thermal` (hardest
   signs, by then the pattern is routine).
4. `make_bus_behaviour` + `check_solution`.
5. Wire `admm_vanilla`; run n=5 end-to-end with **fixed `rho = 2.0`** first. Do not enable
   `rho_for_size` until a fixed-`rho` run matches centralized.
6. `Network` line-limit branch; validate at n=45.
7. Tune the `rho` schedule; confirm the proxes survive the thread and process executors.

## Convergence assessment

The existing memory note (*ADMM voltage-bound stall*) warns that ADMM stalled when the
optimum sat on `V_max` because `f` lacked a constraint the bus projections enforced. **That
risk does not carry over here**, and the reason matters: the old `g` was **nonconvex** (the
voltage annulus `V_min <= |V| <= V_max` plus the `S = V conj(I)` hyperbola), and nonconvex
ADMM has no convergence guarantee. Constraint *placement* alone does not cause a stall for
convex operators.

Here every set is convex and polyhedral — `f` is an affine subspace intersected with a
polyhedron, `g` is a separable product of convex quadratics over polyhedra, and the
intersection is nonempty (the centralized QP solves at every size tested). Standard ADMM
theory applies. Duplicating the line limits into `g` is also impossible in principle: line
flow depends on `theta_i - theta_j`, which no per-bus projection can express. The split is clean.

The genuine risks, in order: (1) `rho_for_size` was tuned for the old problem and is
unbounded, so late iterations freeze `z` and shrink `s` artificially — hence testing with
fixed `rho` first; (2) coordinate scaling, addressed by the temp shift; (3) `g` is the
identity on `theta`, so `u["theta"]` should go to zero — if it grows, `Network.prox` has the
`theta_0 = 0` bug from finding 2.

## Verification

New `test_admm.py`, mirroring `test_centralized.py`'s structure:

- **Unit, `Network`**: `||p - B_dc theta||` ≈ 0; idempotence `prox(prox(z)) == prox(z)`;
  `rho`-invariance `prox(z,1) == prox(z,100)`; agreement with a joint-space cvxpy reference
  on random draws at n=5,8,45, including draws scaled up to force line violations so the QP
  branch is exercised.
- **Unit, devices**: idempotence and `rho`-invariance for `Load`/`Battery`/`Thermal`;
  `Generator` checked against a direct cvxpy solve at `rho ∈ {0.5, 2, 50}` (it is neither
  idempotent nor `rho`-invariant). Dedicated `Thermal` sign test: feed `z["p"] = -0.5*ones`
  and assert the returned `p` stays negative. Assert foreign columns are bit-identical in/out.
- **Integration, n=5 and n=8**: objective gap `|obj_admm - obj_cent| / obj_cent < 1e-3`;
  `p` distance (expect ~1e-3, per the `rho_for_size` docstring's own 0.003 % at 8 buses);
  `theta` compared **only after removing the per-timestep mean from both** (finding 2);
  `soc`/`temp` compared on owning columns only, since the two layouts differ in width.
- **Integration, n=45**: same, slower tier — the only size exercising the line-limit QP.
- **Regression gate**: reuse `test_centralized.py`'s `check_solution` verbatim on the ADMM
  result, after repacking the trajectory into the centralized width layout.

Run: `./venv/Scripts/python.exe test_admm.py`, and `python test_centralized.py` to confirm
the centralized path still passes unchanged.

## Outcome (2026-09-21)

Implemented as above with these deviations, each forced by a measurement:

- **`rho` is fixed at 2.0, not `rho_for_size`.** Under the ramp `s = rho·|dz|` stalls
  (7.2e-3 at n=5 while `r` reaches 1e-9) because `|dz|` shrinks only as `1/rho`; the run
  never stops and `rho` grows until CLARABEL fails at ~1e14. A `|dz|` stop rule was
  considered and rejected: it is not the ADMM dual-feasibility criterion, and the ramp
  itself has no convergence guarantee. Sweep of fixed `rho ∈ {1, 2, 5, 10, 20}` and residual
  balancing at n = 5/8/15/45 (table in `common.rho_fixed`): fixed 2 converges everywhere
  within ~2× of the best fixed value; balancing wins at n=15 but fails at n=45 after a single
  `rho` halving. `rho_for_size` / `rho_geometric` remain available via `solve(rho=...)`; the
  docstring table on `rho_for_size` describes the old problem.
- **Only the objective and generator dispatch are asserted against centralized.** The
  cost is strictly convex in `p_gen` alone, so the battery split (and with it `theta`,
  `soc`, thermal timing) is non-unique: at n=5 the two batteries differ from centralized
  by 0.06 each while their sum agrees to 3e-3. Those distances are printed, not asserted.
- **Prox QPs solve at CLARABEL tolerance 1e-10** (`common._QP_TOL`). A projection's
  variable error scales as √(objective gap), so the default 1e-8 left the device
  projections accurate to only ~1e-4 and the line-limit QP non-idempotent at 6e-5.
- **`test_centralized.check_solution` gained `imbalance_tol`** (default unchanged) so it can
  gate the ADMM `z`, whose power balance holds to the primal residual, not 1e-6.

Results, fixed `rho = 2`, threshold `1e-3·sqrt(n/8)`: n=5 197 its / 1.9 s / gap 7.5e-9;
n=8 820 its / 11.7 s / gap 1.6e-5; n=45 781 its / 70 s / gap 2.2e-6 with line utilization
1.000. Thread and process executors reproduce the sequential run bit-for-bit.

### rho-schedule follow-up at n = 45 and 75 (2026-09-21, 22 runs, 2000-iteration budget)

Question: can tuned residual balancing, a larger fixed `rho`, or a capped ramp beat fixed 2
on the larger networks, and why does balancing fail at n=45 when it worked below?

- **Balancing fails at n=45 for config-specific reasons, not size.** (i) `rho = 2` is already
  the best fixed value there (1: 1168, 2: 781, 3: 873, 5: 819, 10: 1528, ≥20: none), and
  `s/r` averages 1–2 in that run. (ii) The residuals oscillate with a ~40-iteration period
  whose `s/r` swing exceeds `mu = 10` at the troughs, so every trigger is a false positive;
  each halving (with the `u` rescale) kicks `r` up 5×, and the rule locks onto the
  oscillation (changes come in pairs 20 iterations apart). `tau = 1.5` is worse (45
  changes); `mu ≥ 50`, the rule every 50 iterations, or the rule on 40-iteration running
  means never trigger and reduce to fixed 2. (iii) The oscillation comes from the single
  binding line (26, 34); see the mechanism below. With the limits relaxed 1.5× the same
  config converges 3× faster. At n=75 (two binding lines, not bridges into a rigid pocket)
  balancing converges in 149 vs 172 fixed with one early change to `rho = 1`, as at n=5/15.
- **Larger fixed `rho` is monotonically worse at both sizes**, iterations ~linear in `rho`
  (n=75: 2→172, 5→476, 10→941). `r` converges fast, `s = rho·|dz|` does not; at `rho = 50`
  the dispatch is 0.18 off after 2000 iterations while the objective gap reads 8e-6.
- **Capped ramps (growth 1.02/1.05) freeze the iterate**: caps 20 and 100 fail at n=45
  (small `r`, `s` 0.01–0.25, dispatch 0.01–0.19 off); cap 10 converges in 1021 (1.3×
  fixed 2); cap 20 converges at n=75 in 1605 (9× fixed 2).

Conclusion: fixed `rho = 2` stays the default. The n=45 cost is structural (below).

### Mechanism of the n=45 slow mode (2026-09-21, current cost coefficients)

Line (26,34) is a **bridge**; behind it is a radial pocket {17, 20, 24, 26, 31} holding two
fixed loads and three thermal units, no generation or storage. During the 6:30–8:00 pre-heat
(t = 26–32) the pocket's demand at full heating exceeds the line's 4.445, so the optimum
rations the three heaters: pocket net demand equals the limit at all seven steps, the heat is
shared over time (26 at 1.0 for six steps, 24 ramping 0.44→1.0, 31 ramping 1.0→0.39), and the
line multiplier is large (mean 3.7, max 7.0 vs β ≈ 6). That ration is an implicit coupling
constraint among three devices that live in separate `g` blocks; only the dual (bus prices)
can enforce it, so ADMM runs a price-adjustment loop on the pocket. The devices' price
response is degenerate — zero at a box bound (bus 26 sits at `p_max` six of seven steps),
bang-bang off it and lagged through the decay-0.9 dynamics — so clearing needs large price
moves for small quantity moves: an ill-conditioned fixed point with lag, i.e. a slowly damped
oscillation (`cos θ_F` near 1 in Douglas–Rachford terms).

Evidence: every primal quantity on the line converges by iteration ~120 (flow = F, `p26_z =
−1.0000`, `p34 = 0.0414`, active set 7 steps every iteration, no chattering) while the pocket
price `u26` is still oscillating around 3.63 at iteration 890; `u34` across the bridge settles
at 3.2966 by iteration 126. Asymptotic rate: base a decade per 751 iterations (892 to
converge); limits relaxed 1.5× a decade per 91 (253); the same config with `ε·Σ(p_th² +
p_batt²)` added to both solvers, which removes every flat direction of the cost (optimal face
dimension ≥ 11 → 0), a decade per 489 (787) — so the earlier "flat direction cut by the
limit" hypothesis is wrong; a quadratic on `p` adds no elasticity at an active bound.

Levers that follow: re-partition the split at the bridge (pocket + bridge constraint as one
`g` block, the multi-area-OPF remedy); a larger per-block `rho` on pocket buses; over-relaxation
(damping only, constant factor).
