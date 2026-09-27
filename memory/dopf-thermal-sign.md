---
name: dopf-thermal-sign
description: Thermal dynamics sign was flipped 2026-09-18 so p>0 heats; also the state-indexing convention in Trajectory
metadata: 
  node_type: memory
  type: project
  originSessionId: 8ffca1ac-3d5e-427e-8e75-36c7df0b8cdb
  modified: 2026-09-18T17:42:56.925Z
---

`Config.py` originally documented the thermal dynamics with a minus on the control
term (`- eta/c*p`), making positive `p` *cool*. On 2026-09-18 the user decided to
flip it, so the implemented and documented dynamics are now:

    T(t+1) = (1-mu/c)*T(t) + eta/c*p(t) + mu/c*T_amb(t)    # p > 0 heats

The thermal bus injection is `-p_t`: a heating unit draws power, like any load.
Sanity check: mean thermal `p` should be **positive** over a Vancouver fall day.

**Why:** the flip makes thermal units genuine consumers, which roughly doubles peak
demand and was what forced the line limits up (see [[dopf-config-feasibility]]).

**How to apply:** state trajectories (`temp`, `soc`) are solved on N+1 points but
stored as indices **1..N**, so stored index `t` is the state *at* step `t` and lines
up with `load_P`, `thermal_T_amb` and the comfort band. Storing `0..N-1` instead
puts the deadband check off by one step — that bug was caught by the constraint
assertions in `test_centralized.py`, which are worth keeping in that form.
