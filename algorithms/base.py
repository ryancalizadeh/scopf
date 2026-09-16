from dataclasses import dataclass
from typing import Optional
import numpy as np
from Trajectory import Trajectory


@dataclass
class ConvergenceHistory:
    r: list[float]
    s: list[float]
    rho: list[float]


@dataclass
class SolveResult:
    trajectory: Trajectory
    obj: float
    runtime: float
    p_residual: Optional[float]
    s_residual: Optional[float]
    convergence: Optional[ConvergenceHistory]
    
