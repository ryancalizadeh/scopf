---
name: random-configs-infeasible
description: "HISTORY (pre-2026-09-15 TSCOPF nuke; load_fraction and sandbox.py no longer exist): Config's random network generator originally produced statically infeasible AC OPFs; fixed 2026-09-06 (impedances /3, loads scaled to load_fraction of P_max). Current feasibility rules: dopf-config-feasibility"
metadata: 
  node_type: memory
  type: project
  originSessionId: cceed7c1-45e2-4f97-be5e-cdbd279bdfc6
  modified: 2026-09-07T06:19:46.033Z
---

**History only.** This is from the AC TSCOPF problem that was removed on 2026-09-15; `load_fraction` and `sandbox.make_config()` no longer exist. For the current dynamic DC-OPF see [[dopf-config-feasibility]]. The one lesson that carries over is the Why below.

Until 2026-09-06 `Config(n_buses, gen_ratio=0.2, load_ratio=0.6, avg_degree=2.0)` gave sum(load_P) > sum(P_max) and, with X ~ 0.1-0.2 p.u. lines on a tree-like topology, reactive losses exhausted the generator Q limits, so the static OPF was infeasible at every size (loosening V bounds alone did not help beyond ~20% loading).

Fix applied (with the user's approval): line R ~ U(0.003, 0.01), X ~ U(0.03, 0.07); loads rescaled after P_max is drawn so sum(load_P) = `load_fraction` * sum(P_max) (ctor param, default 0.3); V bounds are 0.85/1.15 (user's choice). `configs.pkl` was regenerated. Static feasibility verified for 5/15/45/135 buses and all exp2 configs; the 405-bus config was not checked.

**Why:** IPOPT's "Infeasible_Problem_Detected" from `centralized.solve` looked like a formulation bug but was the data.

**How to apply:** If a new random config is infeasible, first check sum(load_P) vs sum(P_max) and the line reactances before touching the solver. The 3-bus `sandbox.make_config()` case is always feasible. 50% loading is infeasible at 135 buses even with impedances /10 (probably the 1.0 p.u. line-flow limit).
