from Proxable import Proxable
from ZDict import ZDict

def admm(f: Proxable,
         g: Proxable,
         z0: ZDict,
         rho=lambda i, prev, r, s: 2.0,
         threshold=1e-3,
         max_iterations=1000,
         callback=None):
    """
    Minimizes a constrained optimization problem using the Alternating Direction Method of Multipliers (ADMM).

    Parameters
    ----------
    f : Proxable
        The (possibly constrained) objective function to be minimized.
    g : Proxable
        The projection operator representing the constraints.
    z0 : ZDict
        The initial guess for the solution.
    callback : callable, optional
        Called as callback(iteration, x, z, u) at the end of each iteration.
    """

    # Initialize x0, z0, mu0
    zs: list[ZDict] = [z0.copy()]
    xs: list[ZDict] = [z0.zeroslike()]
    us: list[ZDict] = [z0.zeroslike()]

    rs = [(xs[-1] - zs[-1]).norm()]
    ss = [(zs[-1] - zs[-1]).norm()]  # Initialize ss with zero since there's no previous z
    rhos = [2.0]

    for iteration in range(max_iterations-1):
        new_rho = rho(iteration, rhos[-1], rs[-1], ss[-1])
        rhos.append(new_rho)
        # Rescale u when rho changes to keep λ = rho*u continuous
        if new_rho != rhos[-2]:
            us[-1] = us[-1] * (rhos[-2] / new_rho)

        xs.append(f.prox(zs[-1] - us[-1], rhos[-1]))
        zs.append(g.prox(xs[-1] + us[-1], rhos[-1]))
        us.append(us[-1] + (xs[-1] - zs[-1]))

        rs.append((xs[-1] - zs[-1]).norm())
        ss.append(rhos[-1] * (zs[-1] - zs[-2]).norm())

        if callback is not None:
            callback(iteration, xs[-1], zs[-1], us[-1], rs[-1], ss[-1])

        if rs[-1] < threshold and ss[-1] < threshold:
            break

    return xs, zs, us, rs, ss, rhos