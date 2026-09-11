"""
Standalone time-domain plots of the transient that follows each dispatch found
by the centralized solver and by ADMM. Reads the most recent raw results pickle
written by main.py (results/<sweep>_raw_<timestamp>.pkl), re-simulates the
discretized dynamics from the stored dispatch (P(0), Q(0)) with simulate.py,
and overlays the algorithms. Not called by main.py; run it afterwards:

    python plot_dynamics.py                      # largest case in the latest results
    python plot_dynamics.py --n-buses 15         # a specific case
    python plot_dynamics.py --algorithms centralized admm_vanilla --no-show

Top panel: rotor speed deviation of every generator (in Hz).
Bottom panel: rotor angle relative to the centre of inertia with the
+/- 100 deg transient stability bounds used by the optimisation.
Centralized results are drawn solid, ADMM results dashed/dotted, one colour
per generator.

Results written before ExperimentResult gained dispatch_Q cannot be plotted;
the script says so and exits.
"""
import argparse
import glob
import os
import pickle
import sys

import numpy as np
import matplotlib.pyplot as plt

from Config import Config  # noqa: F401  (needed to unpickle the results)
from ExperimentResult import ExperimentResult  # noqa: F401
from simulate import simulate_dispatch

DELTA_COI_MAX_DEG = 100.0
LINESTYLES = {
    "centralized": "-",
    "admm_vanilla": "--",
    "admm_parallel": ":",
    "admm_parallel_relaxed": "-.",
}


def latest_raw_results(results_dir: str) -> str:
    files = sorted(glob.glob(os.path.join(results_dir, "*_raw_*.pkl")), key=os.path.getmtime)
    if not files:
        sys.exit(f"No *_raw_*.pkl found in {results_dir!r}; run main.py first.")
    return files[-1]


def pick_case(raw: dict, n_buses):
    """raw maps case key -> {algorithm: [ExperimentResult, ...]}."""
    sizes = {}
    for key, runs_by_alg in raw.items():
        for runs in runs_by_alg.values():
            if runs:
                sizes[key] = runs[0].config.n_buses
                break
    if not sizes:
        sys.exit("The results file contains no runs.")
    if n_buses is None:
        key = max(sizes, key=sizes.get)
    else:
        matches = [k for k, n in sizes.items() if n == n_buses]
        if not matches:
            sys.exit(f"No case with {n_buses} buses; available: {sizes}")
        key = matches[0]
    return key, raw[key]


def coi_relative_angles_deg(traj, config) -> np.ndarray:
    """delta_i(t) - delta_coi(t) in degrees, shape (N, n_gens)."""
    delta = np.asarray(traj["delta"], dtype=float)
    M = np.asarray(config.M, dtype=float)
    delta_coi = delta @ (M / M.sum())
    return np.rad2deg(delta - delta_coi[:, None])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--n-buses", type=int, default=None, help="case to plot (default: the largest)")
    parser.add_argument("--algorithms", nargs="+", default=None,
                        help="algorithms to overlay (default: all present)")
    parser.add_argument("--no-show", action="store_true", help="only save the figure")
    args = parser.parse_args()

    path = latest_raw_results(args.results_dir)
    with open(path, "rb") as fh:
        raw = pickle.load(fh)
    key, runs_by_alg = pick_case(raw, args.n_buses)
    print(f"results file: {path}")

    order = args.algorithms or ([a for a in LINESTYLES if a in runs_by_alg] + [a for a in runs_by_alg if a not in LINESTYLES])
    sims = {}
    for alg in order:
        if alg not in runs_by_alg or not runs_by_alg[alg]:
            print(f"  {alg}: not in the results, skipped")
            continue
        res = runs_by_alg[alg][0]  # first run of each algorithm
        Q0 = getattr(res, "dispatch_Q", None)
        if Q0 is None:
            print(f"  {alg}: no dispatch_Q stored (results predate it), skipped")
            continue
        traj, info = simulate_dispatch(res.config, res.dispatch, Q0, return_info=True)
        sims[alg] = (res, traj, info)
    if not sims:
        sys.exit("Nothing to plot: the results carry no dispatch_Q. Re-run main.py.")

    config = next(iter(sims.values()))[0].config
    N, dt, n_gens = config.N, config.dt, config.n_gens
    t = np.arange(N) * dt
    cmap = plt.get_cmap("tab20" if n_gens > 10 else "tab10")
    colors = [cmap(i % cmap.N) for i in range(n_gens)]
    fault_line = getattr(config, "fault_line", None)
    t_clear = getattr(config, "t_clear", None)
    if fault_line is not None:
        contingency = (f"fault on line {tuple(int(b) for b in fault_line)} at bus {int(config.fault_bus)}, "
                       f"cleared at t={t_clear:g} s")
    else:  # older configs used a load step
        contingency = f"load step x{config.load_step_factor} at t={dt:g} s"
    print(f"case: {key} ({config.n_buses} buses, {n_gens} generators, N={N}, dt={dt}, {contingency})")

    fig, (ax_f, ax_d) = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
    ref_alg = next(iter(sims))
    ref_freq = np.asarray(sims[ref_alg][1]["omega"], dtype=float) / (2 * np.pi)
    ref_dev = coi_relative_angles_deg(sims[ref_alg][1], config)
    for alg, (res, traj, info) in sims.items():
        ls = LINESTYLES.get(alg, "-")
        freq_hz = np.asarray(traj["omega"], dtype=float) / (2 * np.pi)
        dev_deg = coi_relative_angles_deg(traj, config)
        for i in range(n_gens):
            ax_f.plot(t, freq_hz[:, i], ls, color=colors[i], lw=1.3)
            ax_d.plot(t, dev_deg[:, i], ls, color=colors[i], lw=1.3)
        max_dev = np.abs(dev_deg).max()
        line = (f"  {alg:<22} obj={res.obj:.5f}  max|delta-delta_coi|={max_dev:6.2f} deg "
                f"({'within' if max_dev <= DELTA_COI_MAX_DEG else 'VIOLATES'} bound)  "
                f"powerflow mismatch={info['powerflow_mismatch']:.1e} network residual={info['max_network_residual']:.1e}")
        if alg != ref_alg:
            line += (f"\n{'':26}vs {ref_alg}: max|dfreq|={np.abs(freq_hz - ref_freq).max():.2e} Hz, "
                     f"max|dangle|={np.abs(dev_deg - ref_dev).max():.2e} deg, "
                     f"max|dP0|={np.abs(res.dispatch - sims[ref_alg][0].dispatch).max():.2e}")
        print(line)

    for ax in (ax_f, ax_d):
        if t_clear is not None:
            ax.axvspan(0.0, t_clear, color="0.85", alpha=0.6, zorder=0)
            ax.axvline(t_clear, color="0.4", lw=1.0, ls="--")
        else:
            ax.axvline(dt, color="0.6", lw=0.8, ls="--")
        ax.grid(True, alpha=0.3)
    if t_clear is not None:
        ax_f.text(t_clear / 2, ax_f.get_ylim()[1], "fault on ", color="0.35", fontsize=8,
                  ha="center", va="top")
    ax_f.set_ylabel("frequency deviation [Hz]")
    ax_f.set_title(f"{key}: {config.n_buses} buses, {n_gens} generators, {contingency}")
    ax_d.axhline(DELTA_COI_MAX_DEG, color="k", ls="--", lw=1.2)
    ax_d.axhline(-DELTA_COI_MAX_DEG, color="k", ls="--", lw=1.2)
    ax_d.set_ylabel("rotor angle relative to COI [deg]")
    ax_d.set_xlabel("time [s]")
    ax_d.set_ylim(-1.15 * DELTA_COI_MAX_DEG, 1.15 * DELTA_COI_MAX_DEG)

    # The bounds dwarf the actual swing, so repeat the angle traces in a zoomed inset.
    all_dev = np.concatenate([coi_relative_angles_deg(traj, config).ravel() for _, traj, _ in sims.values()])
    span = max(np.abs(all_dev).max(), 1e-3)
    # if span < 0.5 * DELTA_COI_MAX_DEG:
    #     ax_in = ax_d.inset_axes([0.07, 0.56, 0.55, 0.38])  # upper half: the main traces stay near 0
    #     for alg, (res, traj, info) in sims.items():
    #         dev_deg = coi_relative_angles_deg(traj, config)
    #         for i in range(n_gens):
    #             ax_in.plot(t, dev_deg[:, i], LINESTYLES.get(alg, "-"), color=colors[i], lw=1.1)
    #     ax_in.set_ylim(-1.15 * span, 1.15 * span)
    #     ax_in.set_title(f"zoom: max |delta - delta_coi| = {span:.2f} deg", fontsize=8)
    #     ax_in.tick_params(labelsize=7)
    #     ax_in.grid(True, alpha=0.3)
    #     ax_in.axvline(dt, color="0.6", lw=0.8, ls="--")

    style_handles = [plt.Line2D([], [], color="k", ls=LINESTYLES.get(alg, "-"), label=alg) for alg in sims]
    style_handles.append(plt.Line2D([], [], color="k", ls="--", lw=1.2, label=f"COI bound ±{DELTA_COI_MAX_DEG:g}°"))
    ax_d.legend(handles=style_handles, loc="lower right", fontsize=9)
    if n_gens <= 12:
        gen_handles = [plt.Line2D([], [], color=colors[i], label=f"gen {i}") for i in range(n_gens)]
        ax_f.legend(handles=gen_handles, loc="upper right", fontsize=8, ncol=2)
    fig.tight_layout()

    stamp = os.path.basename(path).rsplit("_raw_", 1)[-1].replace(".pkl", "")
    out = os.path.join(args.results_dir, f"dynamics_{key}_{stamp}.png")
    fig.savefig(out, dpi=150)
    print(f"figure saved: {out}")
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
