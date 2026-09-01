from abc import ABC, abstractmethod
from Trajectory import Trajectory

class Proxable(ABC):
    """
    Abstract class implementing the solution to
    Prox_{rho, f}(z) = min_x f(x) + rho/2||x - z||^2
    """
    @abstractmethod
    def prox(self, z: Trajectory, rho: float) -> Trajectory:
        pass