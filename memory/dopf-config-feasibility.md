---
name: dopf-config-feasibility
description: Three Config.py parameters that must stay tuned or the dynamic OPF goes infeasible; verified 2026-09-18 across n_buses 5-105
metadata: 
  node_type: memory
  type: project
  originSessionId: 8ffca1ac-3d5e-427e-8e75-36c7df0b8cdb
  modified: 2026-09-18T17:42:47.623Z
---

The dynamic OPF (`algorithms/centralized.py`) is infeasible unless three `Config.py`
parameters stay tuned. All three were fixed on 2026-09-18 and verified feasible for
`n_buses` in {5, 8, 15, 45, 105}.

1. **`battery_q0` must lie inside `[q_min, q_max]`.** It was `zeros()` while
   `q_min = uniform(0.1, 0.3)`, infeasible at t=0. Now `0.5*(q_min + q_max)`.
2. **`thermal_eta` is scaled x4.** Steady-state lift above ambient at full heat is
   `eta*p_max/mu`. Unscaled gives ~5C of lift, but the occupied band (20-23C) sits
   ~15C above an 8-15C ambient. x3 clears the median draw but leaves some seeds
   short (n=15 and n=105 fail); x4 clears all.
3. **`line_flow_limits` is `1.5 * gen_P_max.max()`, not `ones_like(B)`.** The
   minimum feasible limit depends on the random topology (measured 1.0-3.5 p.u.
   across sizes), not on bus count, so no simple demand-based formula works.

**Why:** these interact — the thermal sign convention makes heating a real load, so
raising `eta` raises peak demand, which is what forces the line limits up.

**How to apply:** after changing any of these, re-check feasibility across sizes
rather than trusting one case. `min`/`max` slack per thermal bus is quickest found
with a per-bus min-violation LP (minimize deadband slack subject to the thermal
dynamics); the coupled problem then confirms. See [[dopf-thermal-sign]].

**Update 2026-09-24:** at `n_buses=500` (default avg_degree 2.2), seeds 0-9: seed 7 is
infeasible (cause not diagnosed); the other nine solve in ~50-65 s each with CLARABEL.
Line limits bind on only 0-16 of 550 lines, so they are rarely the bottleneck there.
