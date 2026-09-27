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

Re-verified end-to-end 2026-09-22 on seed=42 at rho=2 (892 its). This was a separate session that reached the same diagnosis; its extra details:
- Signature at convergence: every non-pocket bus prices at exactly 3.2966. All five pocket buses share one elevated price that rises 3.633 → 6.795 over the congested window (congestion rent). The optimum shares heat over time: bus 26 runs at 1.0 for six steps, bus 24 ramps 0.48→1.0, bus 31 ramps 1.0→0.39, and all three reach exactly 20.0 C at 8 am.
- `u26` ringing decays slowly: half-range 0.122 over its 200-300, 0.011 over 790-890. Flow = F by iteration 56; the active set is a steady 7 steps from iteration 88.
- Shifting the offered pocket price uniformly by ±0.4 moves the heaters' aggregate congested-window heat by only ~0.005 (pass-through ≈ 0.03), so clearing needs price moves ~30× the quantity error.
- That session's eps-regulariser was ε·Σp² on all flexible power, and it gave 892 its, *identical* to baseline. The 787 above used a different variant (p_th, p_batt in both solvers). Both support the same conclusion.

**How to apply:** don't expect any rho schedule or regularisation to fix n=45. When any DOPF config is unexpectedly slow, first check for a binding line that is a *bridge* into a pocket with no local generation: that is a splitting problem, not a tuning one. Levers that address the mechanism: re-partition the split at the bridge (pocket + bridge constraint as one g block, multi-area-OPF style), per-block rho on the pocket buses, over-relaxation for damping. Write-up: docs/admm-prox-plan.md "Mechanism of the n=45 slow mode". See [[rho-schedule-preference]], [[admm-binding-lines-scaling]] (iterations don't scale with the binding-line count) and [[admm-voltage-bound-stall]] (the same splitting-failure class in the old AC problem).
