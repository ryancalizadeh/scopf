---
name: n45-binding-line-oscillation
description: "the 45-bus config is slow for ADMM because line (26,34) is a bridge into a pocket of two loads and three thermals that is rationed during the 6:30-8:00 pre-heat; the ration couples three g-blocks that only the dual can enforce, and the thermals' price response is zero/bang-bang; NOT a flat-direction issue (regularising flexible power does not help)"
metadata: 
  node_type: memory
  type: project
  originSessionId: d572b9b3-4e88-4a62-be9b-c702550029c3
  modified: 2026-09-23T19:51:30.494Z
---

Measured 2026-09-21 on the dynamic DC-OPF ADMM (f = Network, g = BusBehaviours), n=45, current cost coefficients (alpha 0.01-0.08, beta 5-7).

- Line (26,34) is a bridge; behind it {17, 20, 24, 26, 31} = 2 loads + 3 thermals, no gen/battery. At t=26..32 the pocket's demand at full pre-heat exceeds the limit 4.445; the optimum rations the heaters (pocket net demand = F at all 7 steps, multiplier mean 3.7 / max 7.0).
- The ration is an implicit coupling among three separate g-blocks, enforced only through the dual (prices). Measured elasticity (dp_out/dp_in under a sustained 7-step nudge, 1 = follows the price): pocket thermals 0.01-0.04, batteries 0.26-0.47, interior generator 0.95 = rho/(2*alpha+rho) (theory confirmed). Crucially bus 24's heater is NOT at p_max (0.44-0.48) and is still 0.01: the rigidity is the comfort band + decay-0.9 dynamics pinning the INTEGRAL of pre-heat needed to reach 20 C by 8am, not the power box. The ration constrains exactly that aggregate, so the price loop is near-vertical-supply-curve and converges slowly. Slow mode = u26 (pocket price), still moving at it 890 while all primal quantities and the active set (7 steps, no chattering) settle by ~120; u34 across the bridge settles at it 126.
- Fix that works: soft comfort band (slacks penalised sigma*s^2, sigma=20) -> 614 its, decade/364, for a 0.147 C worst violation and 0.05% objective change. An eps-quadratic on flexible power does NOT help (787 its) because it adds no elasticity at an active constraint.
- Rates (decade per N its): base 751 (892 its); limits x1.5 → 91 (253); eps-quadratic on p_th, p_batt in both solvers (removes all flat directions, face dim >= 11) → 489 (787) — flat directions are NOT the cause.
- Residual balancing fails there only because rho = 2 is already balanced on average and the pocket oscillation's s/r swing exceeds mu=10 (false triggers); works at n=5/15/75. Larger fixed rho and capped ramps are worse at n=45 and 75.

**How to apply:** don't expect any rho schedule or regularisation to fix n=45. Levers that address the mechanism: re-partition the split at the bridge (pocket + bridge constraint as one g block, multi-area-OPF style), per-block rho on the pocket buses, over-relaxation for damping. Write-up: docs/admm-prox-plan.md "Mechanism of the n=45 slow mode". See [[rho-schedule-preference]].
