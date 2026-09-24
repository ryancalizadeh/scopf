from typing import Optional
from Config import Config
import numpy as np
from dataclasses import dataclass
from Trajectory import Trajectory
from algorithms.base import ConvergenceHistory


@dataclass
class ExperimentResult:
    config: Config
    algorithm: str
    trajectory: Trajectory
    obj: float
    runtime: float
    p_residual: Optional[float]
    s_residual: Optional[float]
    convergence: Optional[ConvergenceHistory]

    def __eq__(self, other):
        if not isinstance(other, ExperimentResult):
            return NotImplemented
        return (self.trajectory - other.trajectory).norm() < 1e-6 and abs(self.obj - other.obj) < 1e-6


@dataclass
class AggregatedResult:
    algorithm: str
    runtime_mean: float
    runtime_std: float
    obj_mean: float
    obj_std: float
    p_residual_mean: Optional[float]
    s_residual_mean: Optional[float]


def aggregate(results: list[ExperimentResult]) -> AggregatedResult:
    if not results:
        raise ValueError("Cannot aggregate an empty list of results.")

    algorithm = results[0].algorithm
    runtimes = np.array([r.runtime for r in results])
    objs = np.array([r.obj for r in results])

    p_residuals = [r.p_residual for r in results if r.p_residual is not None]
    s_residuals = [r.s_residual for r in results if r.s_residual is not None]

    return AggregatedResult(
        algorithm=algorithm,
        runtime_mean=float(np.mean(runtimes)),
        runtime_std=float(np.std(runtimes)),
        obj_mean=float(np.mean(objs)),
        obj_std=float(np.std(objs)),
        p_residual_mean=float(np.mean(p_residuals)) if p_residuals else None,
        s_residual_mean=float(np.mean(s_residuals)) if s_residuals else None,
    )
