---
name: working-feedback-style
description: "user wants to be kept updated during long tasks — what is being checked, what problems come up, what was tried, and how it went — rather than a silent run and a final summary"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: d572b9b3-4e88-4a62-be9b-c702550029c3
  modified: 2026-09-21T17:54:00.271Z
---

While working through a multi-step task, narrate progress between tool calls: what is being verified and why, each problem as it appears, the fix attempted, and the measured outcome. Stated 2026-09-21 ("Keep me updated as you work, let me know what problems come up, what you try to overcome them, and how that goes").

**Why:** the user makes design calls mid-task (e.g. rejected the `|dz|` stop rule and the rho ramp on theoretical grounds as soon as the stall was reported, see [[rho-schedule-preference]]); they need the evidence as it arrives, not after.

**How to apply:** short status messages with numbers before and after each investigation step; surface any deviation from a plan or from theory immediately and let the user decide rather than quietly working around it.
