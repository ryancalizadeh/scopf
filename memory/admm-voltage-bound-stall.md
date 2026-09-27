---
name: admm-voltage-bound-stall
description: "HISTORY (AC TSCOPF, removed 2026-09-15; the DC-OPF has no voltage): ADMM stalled (r ~ 2e-2) when the optimum sat on V_max because F had no voltage constraint and the bus projections clamped there. Same splitting-failure class as n45-binding-line-oscillation"
metadata:
  type: project
---

**History only.** This is from the AC TSCOPF problem that was removed on 2026-09-15. The current dynamic DC-OPF has no voltage variables, so the V_max fix below no longer applies. The general lesson still holds: when an optimum sits on a constraint that only one side of the split can see, ADMM stalls or crawls. See [[n45-binding-line-oscillation]].

Found 2026-09-10 when the contingency became a line fault. On the 15-bus case the centralized optimum puts 2 of 15 buses exactly at V_max = 1.15. ADMM then stalls: the primal residual plateaus around 2e-2 (threshold 1.4e-3), the gap is spread evenly across all buses in v and i (~5e-3 each) while delta/omega agree to 1e-6, and the dispatch lands 50% away from centralized at a different feasible point (more conservative: 9 deg peak swing vs 78 deg, 20% higher cost).

**Why:** the voltage bound lives only in the per-bus projections (`Generator`, `ConstPowerLoad`, `EmptyBus` clamp |V(0)| into [V_min, V_max]); `F` constrains the network, P/Q, line flows and COI but not voltage. When the optimum is interior to the bound both sides agree; when it sits *on* the bound, F keeps pulling past it and the projections keep clamping, so x and z never meet. Warm-starting ADMM at the centralized solution keeps it there (dispatch matches to 3 decimals, r ~ 3e-3), which proves the implementation is right and the issue is the splitting, not a bug.

**How to apply:** if ADMM stalls at a residual plateau, check whether |V(0)| is at V_max in the centralized solution before suspecting the rho schedule or the projections. Fixes to consider: add the voltage bound to F as well (it is a convex norm constraint on each bus, so F's t=0 block would become a small SOCP always rather than only on line-limit violations), or relax V_max so the optimum is interior. See [[rho-schedule-preference]] and [[casadi-thread-affinity]].
