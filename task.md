# Main goal
- Test admm n_buses vs runtime, convergence, and compare time-domain solutions with centralized
- 3 plots for TSCOPF and FSCOPF:
    1) n_buses vs runtime with 4 data series: [centralized, admm sequential, admm parallel, admm parallel+relaxation]
    2) convergence of admm
    3) time domain contingency solutions, for admm vs centralized, showing they look very similar and constraints are respected

# Sub goal
- flesh out main logic