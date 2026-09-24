from typing import Dict, List
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ExperimentResult import AggregatedResult
from algorithms.base import ConvergenceHistory
import logging

logger = logging.getLogger(__name__)


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
    fig.savefig(f"results/{title.replace(' ', '_')}.png")
    return fig

