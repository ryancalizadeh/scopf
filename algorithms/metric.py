"""
Rank-one "leverage" metric for the variable-metric ADMM.

ADMM in a metric M replaces every prox term rho/2 ||w - z||^2 by rho/2 ||w - z||_M^2 with the
same M in both proxes (it is plain ADMM on the rescaled variable M^{1/2} w, so the fixed points
are unchanged for any M > 0; only the conditioning changes).

Why: the Euclidean generator projection keeps ~3 % of a push on P(0), because a P(0) change
drags the whole transient (delta, omega, and through E e^{j delta} = V + j Xd I also V, I, S for
t >= 1) along by ~T^2 / 2M, and the Euclidean prox term charges all of it. The pathology is one
*coupled* direction per generator, so a diagonal rescaling cannot fix it (down-weighting
delta/omega alone was tested and diverges, see report.md). The metric here is cheap along that
direction only:

    M_i = I - alpha_i d_i d_i^T          on generator i's consensus block (V_i, I_i, S_i, delta_i, omega_i)
    M   = I                              on load and empty buses

with d_i the (unit) response of the Euclidean generator projection to a push on P_i(0), and
alpha_i = 1 - r_P, r_P the P(0) pass-through of that response. Then the physical direction
D_i = r_i / r_P ("P(0) moves by 1 with its transient in tow") has ||D_i||_M = 1, i.e. it costs
the same as a P(0) move with no transient at all. The d_i have disjoint supports (bus i's
columns and generator i's rotor states), so M is block diagonal per bus and the bus projections
stay separable. M^{-1} = I + beta_i d_i d_i^T with beta_i = alpha_i / (1 - alpha_i).

Coordinates: a Trajectory's complex entries are pairs of real coordinates; the inner product is
the real one, <a, b> = sum Re(conj(a) b), so that Trajectory.norm is the induced norm.
"""
from typing import Dict, Optional

import numpy as np

from Trajectory import Trajectory


def inner(a: Trajectory, b: Trajectory) -> float:
    """Real inner product on the (re, im) coordinates of two same-shaped trajectories."""
    return float(sum(np.real(np.vdot(a[k], b[k])) for k in a.keys()))


class LeverageMetric:
    """
    directions: {generator index -> unit single-bus Trajectory d_i (width 1, keys v, i, s, delta,
    omega)}, alphas: {generator index -> alpha_i in [0, 1)}. An empty metric is the identity.
    """

    def __init__(self, directions: Optional[Dict[int, Trajectory]] = None,
                 alphas: Optional[Dict[int, float]] = None):
        self.directions = dict(directions or {})
        self.alphas = {i: float(a) for i, a in (alphas or {}).items()}
        for i, d in self.directions.items():
            if i not in self.alphas:
                raise ValueError(f"no alpha for generator {i}")
            if not (0.0 <= self.alphas[i] < 1.0):
                raise ValueError(f"alpha must be in [0, 1), got {self.alphas[i]} for generator {i}")
            nrm = np.sqrt(inner(d, d))
            if abs(nrm - 1.0) > 1e-8:
                raise ValueError(f"direction of generator {i} is not normalised (norm {nrm})")

    @classmethod
    def identity(cls) -> "LeverageMetric":
        return cls()

    @classmethod
    def from_responses(cls, responses: Dict[int, "tuple[Trajectory, float]"],
                       alpha_max: float = 0.99) -> "LeverageMetric":
        """
        responses: {generator -> (r_i, r_P)} with r_i the response of the Euclidean generator
        projection to a unit push on P_i(0) (single-bus Trajectory) and r_P its P(0) entry (the
        pass-through, = ||r_i||^2 for an orthogonal projection). alpha_i = clip(1 - r_P, 0,
        alpha_max); generators with alpha_i = 0 or a degenerate response are left Euclidean.
        """
        directions, alphas = {}, {}
        for i, (r, r_P) in responses.items():
            nrm = np.sqrt(inner(r, r))
            alpha = float(np.clip(1.0 - r_P, 0.0, alpha_max))
            if nrm < 1e-12 or alpha <= 0.0:
                continue
            directions[i] = r * (1.0 / nrm)
            alphas[i] = alpha
        return cls(directions, alphas)

    @property
    def is_identity(self) -> bool:
        return not self.directions

    def beta(self, i: int) -> float:
        a = self.alphas[i]
        return a / (1.0 - a)

    def _rank_one(self, t: Trajectory, coeffs: Dict[int, float]) -> Trajectory:
        """t + sum_i coeffs[i] * <d_i, t_i> d_i over the generator blocks."""
        ret = t.copy()
        for i, d in self.directions.items():
            ti = t.at_bus[i]
            c = coeffs[i] * inner(d, ti)
            ret.at_bus[i] = ti + d * c
        return ret

    def apply(self, t: Trajectory) -> Trajectory:
        """M t."""
        return self._rank_one(t, {i: -a for i, a in self.alphas.items()})

    def apply_inv(self, t: Trajectory) -> Trajectory:
        """M^{-1} t."""
        return self._rank_one(t, {i: self.beta(i) for i in self.alphas})

    def norm(self, t: Trajectory) -> float:
        """||t||_M = sqrt(<t, M t>)."""
        return float(np.sqrt(max(inner(t, self.apply(t)), 0.0)))

    def summary(self) -> str:
        if self.is_identity:
            return "identity"
        return ", ".join(f"g{i}: alpha={a:.4f}" for i, a in sorted(self.alphas.items()))
