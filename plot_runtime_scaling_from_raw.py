"""
Plots runtime-vs-n_buses scaling from one or more saved raw results pickles.

    python plot_runtime_scaling_from_raw.py exp1_raw_20260923_115102.pkl
    python plot_runtime_scaling_from_raw.py exp1_raw_20260921_115407.pkl exp1_raw_20260922_173304.pkl

Raw pickles are written by main.py._run_sweep as nested
{sweep_key: {algorithm: [ExperimentResult]}} dicts (one sweep_key per config
swept over, e.g. "exp1_n_buses_15"). Runs are grouped by each result's own
config.n_buses (not by sweep_key, so files/keys can be combined freely) and
aggregated per algorithm before plotting. Passing several files merges their
runs for any n_buses they share, so a handful of narrow sweeps (e.g. one file
per n_buses value) can be combined into a single scaling curve.
"""
import argparse
import os
import pickle
import sys
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

from ExperimentResult import AggregatedResult, ExperimentResult, aggregate
from visualize import plot_runtime_scaling

RESULTS_DIR = "results"


def _resolve_path(pickle_file: str) -> str:
    if os.path.isfile(pickle_file):
        return pickle_file
    candidate = os.path.join(RESULTS_DIR, pickle_file)
    if os.path.isfile(candidate):
        return candidate
    sys.exit(f"Could not find {pickle_file!r} (looked in '.' and '{RESULTS_DIR}/')")


def _collect_runs(paths: List[str]) -> Dict[float, Dict[str, List[ExperimentResult]]]:
    """Groups every ExperimentResult across all pickles by (n_buses, algorithm)."""
    by_n_buses: Dict[float, Dict[str, List[ExperimentResult]]] = defaultdict(lambda: defaultdict(list))
    for path in paths:
        with open(path, "rb") as f:
            data = pickle.load(f)
        for sweep_key, by_algorithm in data.items():
            for algorithm, runs in by_algorithm.items():
                for run in runs:
                    by_n_buses[float(run.config.n_buses)][algorithm].append(run)
    return by_n_buses


def _aggregate_all(by_n_buses: Dict[float, Dict[str, List[ExperimentResult]]]
                    ) -> Dict[float, Dict[str, AggregatedResult]]:
    return {
        n_buses: {algorithm: aggregate(runs) for algorithm, runs in by_algorithm.items()}
        for n_buses, by_algorithm in by_n_buses.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pickle_files", nargs="+", help="Raw results pickle(s) (looked up in results/ if not found as-is).")
    parser.add_argument("--title", default="Runtime scaling", help="Plot title.")
    parser.add_argument("--out", default=None, help="Output PNG path; defaults to results/runtime_scaling_<timestamp>.png.")
    args = parser.parse_args()

    paths = [_resolve_path(p) for p in args.pickle_files]
    by_n_buses = _collect_runs(paths)
    if not by_n_buses:
        sys.exit("No runs found in the given pickle(s).")

    aggregated = _aggregate_all(by_n_buses)
    fig = plot_runtime_scaling(aggregated, xlabel="n_buses", title=args.title)

    out_path = args.out
    if out_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        out_path = os.path.join(RESULTS_DIR, f"runtime_scaling_{timestamp}.png")
    fig.savefig(out_path)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
