import logging
from Proxable import Proxable
from Trajectory import Trajectory

logger = logging.getLogger(__name__)

def admm(f: Proxable,
         g: Proxable,
         z0: Trajectory,
         rho=lambda i, prev, r, s: 2.0,
         threshold=1e-3,
         max_iterations=1000,
         callback=None,
         metric_update=None):
    """
    Minimizes a constrained optimization problem using the Alternating Direction Method of Multipliers (ADMM).

    Parameters
    ----------
    f : Proxable
        The (possibly constrained) objective function to be minimized.
    g : Proxable
        The projection operator representing the constraints.
    z0 : Trajectory
        The initial guess for the solution.
    callback : callable, optional
        Called as callback(iteration, x, z, u) at the end of each iteration.
    metric_update : callable, optional
        Called as metric_update(iteration, z, u) at the start of each iteration; returns a
        new algorithms.metric.LeverageMetric to switch to, or None to keep the current one.
        On a switch the metric is installed in both f and g (set_metric) and the scaled dual
        is rescaled, u <- M_new^{-1} M_old u, so that the multiplier lambda = rho M u stays
        continuous (the same idea as the rescaling on a rho change below). With a metric M the
        iteration is plain ADMM on M^{1/2} w, i.e. the proxes are taken in the M-norm.
    """

    # Initialize x0, z0, mu0
    zs: list[Trajectory] = [z0.copy()]
    xs: list[Trajectory] = [z0.zeroslike()]
    us: list[Trajectory] = [z0.zeroslike()]

    rs = [(xs[-1] - zs[-1]).norm()]
    ss = [(zs[-1] - zs[-1]).norm()]  # Initialize ss with zero since there's no previous z
    rhos = [2.0]
    metric = None

    for iteration in range(max_iterations-1):
        new_rho = rho(iteration, rhos[-1], rs[-1], ss[-1])
        rhos.append(new_rho)
        # Rescale u when rho changes to keep λ = rho*u continuous
        if new_rho != rhos[-2]:
            us[-1] = us[-1] * (rhos[-2] / new_rho)

        if metric_update is not None:
            new_metric = metric_update(iteration, zs[-1], us[-1])
            if new_metric is not None:
                if metric is not None:
                    us[-1] = metric.apply(us[-1])
                us[-1] = new_metric.apply_inv(us[-1])
                f.set_metric(new_metric)
                g.set_metric(new_metric)
                metric = new_metric
                logger.info(f"ADMM iteration {iteration}: metric switched ({new_metric.summary()})")

        xs.append(f.prox(zs[-1] - us[-1], rhos[-1]))
        zs.append(g.prox(xs[-1] + us[-1], rhos[-1]))
        us.append(us[-1] + (xs[-1] - zs[-1]))

        rs.append((xs[-1] - zs[-1]).norm())
        ss.append(rhos[-1] * (zs[-1] - zs[-2]).norm())

        dz = ss[-1] / rhos[-1]  # unscaled dual step ||z_k - z_{k-1}||

        if iteration % 10 == 0:
            logger.debug(f"ADMM iteration {iteration} / {max_iterations-1}: r={rs[-1]:.6g}, s={ss[-1]:.6g}, |dz|={dz:.6g}, rho={rhos[-1]:.6g}")

        if callback is not None:
            callback(iteration, xs[-1], zs[-1], us[-1], rs[-1], ss[-1])

        # Stop on the primal residual and the unscaled dual step: with an
        # uncapped rho ramp (rho reaches 1e5+) the scaled dual residual
        # s = rho*|dz| can never fall below the threshold.
        if rs[-1] < threshold and dz < threshold:
            break

    return xs, zs, us, rs, ss, rhos