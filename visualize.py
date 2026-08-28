from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt
from ExperimentResult import AggregatedResult
from algorithms.base import ConvergenceHistory


def plot_runtime_scaling(aggregated_by_config: Dict[float, Dict[str, AggregatedResult]],
                          xlabel: str, title: str = "Runtime scaling"):
    """
    aggregated_by_config: maps a sweep x-value (e.g. n_buses or avg_degree) to a dict of
    algorithm name -> AggregatedResult for that config.
    """
    x_values = sorted(aggregated_by_config.keys())
    algorithms = sorted({name for agg in aggregated_by_config.values() for name in agg.keys()})

    fig, ax = plt.subplots()
    for name in algorithms:
        means = []
        stds = []
        xs = []
        for x in x_values:
            agg = aggregated_by_config[x].get(name)
            if agg is None:
                continue
            xs.append(x)
            means.append(agg.runtime_mean)
            stds.append(agg.runtime_std)
        if xs:
            ax.errorbar(xs, means, yerr=stds, label=name, marker="o", capsize=3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("runtime (s)")
    ax.set_title(title)
    ax.legend()
    return fig


def plot_convergence(history: ConvergenceHistory, title: str = "ADMM convergence"):
    fig, ax = plt.subplots()
    ax.plot(history.r, label="r (primal residual)")
    ax.plot(history.s, label="s (dual residual)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("residual")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend()
    return fig


def print_dispatch_comparison(aggregated: Dict[str, AggregatedResult], reference: str = "centralized"):
    if reference not in aggregated:
        print(f"No '{reference}' result available for comparison.")
        return

    ref = aggregated[reference]
    print(f"Dispatch/objective deviation vs '{reference}':")
    for name, agg in aggregated.items():
        if name == reference:
            continue
        dispatch_diff = float(np.linalg.norm(agg.dispatch_ref - ref.dispatch_ref))
        obj_diff = abs(agg.obj_mean - ref.obj_mean)
        print(f"  {name}: ||dispatch_diff||={dispatch_diff:.6g}, |obj_diff|={obj_diff:.6g}")
