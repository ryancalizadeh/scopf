# Main goal
- Test admm n_buses vs runtime, convergence, and compare time-domain solutions with centralized
- 3 plots for TSCOPF and FSCOPF:
    1) n_buses vs runtime with 4 data series: [centralized, admm sequential, admm parallel, admm parallel+relaxation]
    2) convergence of admm
    3) time domain contingency solutions, for admm vs centralized, showing they look very similar and constraints are respected

# Sub goal
1) fix up centralized to reflect the new problem (for now, lets do TSC-OPF + partial relaxation. FSC-OPF can come later).
2) Figure out exact splitting (what goes in f and g)
3) implement f
4) implement generator prox, constpowerload prox, and empty prox