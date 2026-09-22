"""
Plots a time-domain trajectory comparison from a saved raw results pickle.

    python plot_trajectory.py exp1_raw_20260921_115407.pkl
    python plot_trajectory.py exp1_raw_20260921_115407.pkl --key exp1_n_buses_15

Raw pickles are written by main.py._run_sweep as nested
{sweep_key: {algorithm: [ExperimentResult]}} dicts (one sweep_key per config
swept over, e.g. "exp1_n_buses_15"). --key picks which one to plot; it can be
omitted if the pickle only has one.
"""
import argparse
import os
import pickle
import sys

from visualize_trajectory import save_trajectory_comparison

RESULTS_DIR = "results"


def _resolve_path(pickle_file: str) -> str:
    if os.path.isfile(pickle_file):
        return pickle_file
    candidate = os.path.join(RESULTS_DIR, pickle_file)
    if os.path.isfile(candidate):
        return candidate
    sys.exit(f"Could not find {pickle_file!r} (looked in '.' and '{RESULTS_DIR}/')")


def _resolve_key(data: dict, key: "str | None") -> str:
    if key is not None:
        if key not in data:
            available = ", ".join(sorted(data.keys()))
            sys.exit(f"Unknown key {key!r}. Available: {available}")
        return key
    if len(data) == 1:
        return next(iter(data))
    available = ", ".join(sorted(data.keys()))
    sys.exit(f"Pickle has {len(data)} sweep keys; pass one with --key. Available: {available}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pickle_file", help="Filename of a raw results pickle (looked up in results/ if not found as-is).")
    parser.add_argument("--key", default=None, help="Sweep key to plot; required if the pickle has more than one.")
    args = parser.parse_args()

    path = _resolve_path(args.pickle_file)
    with open(path, "rb") as f:
        data = pickle.load(f)

    key = _resolve_key(data, args.key)
    results = data[key]

    run_name = f"{os.path.splitext(os.path.basename(path))[0]}_{key}"
    for out_path in save_trajectory_comparison(results, results_dir=RESULTS_DIR, run_name=run_name):
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
