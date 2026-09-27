---
name: mpc-closed-loop
description: "Receding-horizon closed loop (run_mpc.py) built 2026-09-24; user's roadmap is soft state constraints next, then nonlinear battery/thermal plant models; rolling-H results must be read with midnight SOC"
metadata:
  node_type: memory
  type: project
  originSessionId: 224204b9-e820-426e-b8f7-5eb18c8276f2
  modified: 2026-09-24T21:12:29.298Z
---

Closed-loop MPC of the dynamic OPF (`mpc.py`, `plant.py`, `run_mpc.py`, `test_mpc.py`) was built 2026-09-24 with the
user choosing: a rolling fixed-length window (default 24 h), battery SOC = qT at the end of *every* window,
a nominal plant (same linear models as the MPC), generator participation factors (by P_max) for balancing, and an
infeasible window ending the run (for now).

**User's roadmap (in order):** soft state constraints in the MPC (comfort band, SOC bounds, terminal SOC; will need
matching ADMM Thermal/Battery prox changes) is the next step. Later: nonlinear plant models for battery and/or
thermal loads, plugged in through the per-device `step(state, p_cmd, k) -> (next_state, p_realized)` classes.
Forecast error and AC power flow were not requested.

**Findings (n=8, seed 42):**
- Rolling 24 h closed loop costs 0.68% *less* than the day-ahead optimum only because the moving terminal
  constraint lets batteries end the day ~0.2 SOC below qT (deferring the recharge past midnight). Always read the
  cost gap together with the midnight SOC deviation.
- CLARABEL at default tolerances pins generator dispatch only to ~1e-4 (the day-ahead solve moves 1.3e-4 when
  tightened to 1e-10), so the shrinking-horizon Bellman check matches dispatch to 5.9e-4 and cost to 3.5e-10 rel.
  A dispatch gap near 1e-4 is solver accuracy, not a loop bug.

**How to apply:** when adding soft constraints or plant mismatch, rerun `python test_mpc.py` (Bellman check on a
nominal plant) first to keep the loop honest. See [[dopf-thermal-sign]] for the index convention the loop relies on,
and [[dopf-config-feasibility]] for why the lookahead data comes from a separate RNG stream (day-1 data must not change).
