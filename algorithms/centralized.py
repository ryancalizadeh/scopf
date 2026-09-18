import time
import numpy as np
import casadi as ca
from Config import Config
from Trajectory import Trajectory
from algorithms.base import SolveResult
# from algorithms.common import gen_alg_eqs, gen_discrete_step_foh, gen_coi_angle, DELTA_COI_MAX

def solve(config: Config) -> SolveResult:
    raise NotImplementedError("Centralized solver is not implemented yet")
