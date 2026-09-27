---
name: admm-binding-lines-scaling
description: "ADMM iterations do not scale with the number of binding line limits (n=20 seed=42 sweep, 2026-09-25): a step when the first meshed line binds, then 2-6 lines are all 220-840 its with no trend; radial bridges to a lone generator are free"
metadata:
  node_type: memory
  type: project
  originSessionId: a0137d10-2ae0-4a33-97ac-2501e2a3bb68
  modified: 2026-09-25T21:01:23.994Z
---

Experiment 2026-09-25 (`line_limit_sweep.py`, results `results/line_limit_sweep_n20_seed42_20260925_135928.{json,png}`): uniform line limit L on every line of n=20 seed=42 (22 lines), swept 2.80 -> 1.28 (min feasible ~1.2777), admm_parallel at rho_fixed(2), cap 5000. All 37 points converged; iteration counts are deterministic (identical across two runs).

- 0-1 binding lines: 215-224 its. The one line, (2,5), is a bridge whose pocket is only generator 2, so binding it just caps an elastic (quadratic-cost) device: free, even as its dual grows 20x.
- 2+ binding lines (first meshed/cycle line (5,18) at L=2.04): 224-836 its. Within >=2 lines, Spearman rho = 0.17 (p=0.36) with line count, 0.38 with (line, step) pair count. 2 lines alone span 224-700; 6 lines give ~455.
- Two humps in L (1.98-1.88 and 1.74-1.62, peak 836 at 1.62) sit next to binding-set changes, with a ~270-its valley between at the same 2 lines. Nearly-binding count (slack < 1%) does not predict (rho ~ 0); what tracks iterations is how long ADMM's Network active set flickers (100-235 changes, still changing within a few its of the stop in slow runs).

**Why:** consistent with [[n45-binding-line-oscillation]]: cost comes from couplings only the dual can enforce (loop flows over meshed lines, inelastic pockets), not from constraint count.

**How to apply:** don't model ADMM cost as a function of #binding constraints. Open follow-ups: repeat over seeds to test the radial-vs-meshed split; a per-line tightening design to isolate count from which line. The stop rule leaves generator dispatch up to 0.045 p.u. off centralized (obj gap <= 3.4e-5), so "iterations" means iterations to the r/s threshold, not to an accurate dispatch.
