---
name: rho-schedule-preference
description: user chose fixed rho = 2 for the dynamic DC-OPF ADMM (2026-09-21); geometric ramps and the |dz| stop rule rejected as unsupported by ADMM theory; residual balancing fails at n=45; old rho_for_size preference applied to the nuked TSCOPF problem only
metadata: 
  node_type: memory
  type: feedback
  originSessionId: d572b9b3-4e88-4a62-be9b-c702550029c3
  modified: 2026-09-21T17:53:45.630Z
---

User stance (2026-09-21, dynamic DC-OPF split f=Network / g=BusBehaviours): use a **fixed rho** (2.0) — "drop the ramping rho approach if fixed rho works better", and it does. The user explicitly questioned the `|dz| = s/rho` stop rule as unsupported by ADMM theory; agreed: the dual residual is `s = rho·|dz|` (Boyd §3.3), and `|dz| → 0` under a ramp only means the iterate froze. Do not reintroduce the ramp or the `|dz|` rule.

Measured: fixed 2 converges at n=5/8/15/45 (197/820/136/781 its) within ~2× of the best fixed value; best fixed value moves 1→5→2→2 with size, so no single value is optimal. Residual balancing (`rho_heuristic`) wins at n=15 but fails at n=45 after one halving (u-rescale set r back 2× and it never recovered). `rho_for_size` ramp: `s` stalls (|dz| ∝ 1/rho), rho grows to 1e14, CLARABEL fails.

Historical (2026-09-09, nuked TSCOPF problem): there every prox was a projection except a P(0) cost term, fixed rho limit-cycled, and the uncapped ramp with `|dz|` stop was the working choice. That reasoning does not transfer to the new problem, where the generator cost gives g real curvature.

**How to apply:** `admm_vanilla.solve(config, rho=...)` defaults to `common.rho_fixed(2.0)`; other schedules stay available for experiments. `admm.py` stops on `r < thr and s < thr` with `s = rho·|dz|`, threshold `1e-3·sqrt(n_buses/8)`. Only the objective and generator dispatch are unique — the battery split is a flat direction, so compare those, not full `p`.
