"""K01 isolation: fit a whole arm trajectory to an interpolated measured response grid.

This is an offline numerical experiment, NOT anatomical IK. The grid interpolant must
be checked against fresh Core/render results. Temporal penalties do not certify a
natural spatial route, and residuals must be reported alongside continuity.
"""
import numpy as np


class ResponseGrid:
    def __init__(self, entries):
        self.shoulders = np.array(sorted({e['shoulder'] for e in entries}), dtype=float)
        self.swings = np.array(sorted({e['swing'] for e in entries}), dtype=float)
        lookup = {(e['shoulder'], e['swing']): e['hand'] for e in entries}
        self.points = np.array([[lookup[(s, w)] for w in self.swings] for s in self.shoulders])
        if min(len(self.shoulders), len(self.swings)) < 2:
            raise ValueError('Need a complete response grid with at least two points per axis')
        self.lower = np.array([self.shoulders[0], self.swings[0]])
        self.span = np.array([np.ptp(self.shoulders), np.ptp(self.swings)])

    def forward(self, q):
        """q in [0,1]^2; return predicted hand position and derivative wrt q."""
        angles = self.lower + q * self.span
        i = np.clip(np.searchsorted(self.shoulders, angles[:, 0], side='right') - 1,
                    0, len(self.shoulders) - 2)
        j = np.clip(np.searchsorted(self.swings, angles[:, 1], side='right') - 1,
                    0, len(self.swings) - 2)
        ds = self.shoulders[i + 1] - self.shoulders[i]
        dw = self.swings[j + 1] - self.swings[j]
        u = ((angles[:, 0] - self.shoulders[i]) / ds)[:, None]
        v = ((angles[:, 1] - self.swings[j]) / dw)[:, None]
        a, b = self.points[i, j], self.points[i + 1, j]
        c, d = self.points[i, j + 1], self.points[i + 1, j + 1]
        value = (1-u)*(1-v)*a + u*(1-v)*b + (1-u)*v*c + u*v*d
        du = ((1-v)*(b-a) + v*(d-c)) * (self.span[0] / ds)[:, None]
        dv = ((1-u)*(c-a) + u*(d-b)) * (self.span[1] / dw)[:, None]
        return value, np.stack((du, dv), axis=2)


def fit(targets, grid, scale, iterations=40, velocity_weight=0.4, acceleration_weight=1.5):
    """Bounded whole-sequence least squares, with fixed adjacent-sample penalties.

    Weights apply only to the frozen 30Hz experiment, not universal human speed limits.
    Starts at model-neutral parameter values; does not filter a jumping output curve.
    """
    targets = np.asarray(targets, dtype=float)
    if targets.ndim != 2 or targets.shape[1] != 2 or len(targets) < 3:
        raise ValueError('Need at least three 2D targets')
    if not np.all(np.isfinite(targets)) or scale <= 0:
        raise ValueError('Targets must be finite and scale positive')
    n = len(targets)
    identity = np.eye(n * 2)
    first = identity[2:] - identity[:-2]
    second = first[2:] - first[:-2]
    regularizer = velocity_weight * first.T @ first + acceleration_weight * second.T @ second
    q = np.tile(np.clip(-grid.lower / grid.span, 0, 1), (n, 1))

    def objective(x):
        error = (grid.forward(x)[0] - targets) / scale
        flat = x.ravel()
        return float(np.sum(error ** 2) + flat @ regularizer @ flat)

    history = [objective(q)]
    for _ in range(iterations):
        predicted, jac = grid.forward(q)
        jac = jac / scale
        error = (predicted - targets) / scale
        gradient = np.einsum('nki,nk->ni', jac, error).ravel() + regularizer @ q.ravel()
        hessian = regularizer.copy()
        blocks = np.einsum('nki,nkj->nij', jac, jac)
        for index in range(n):
            hessian[2*index:2*index+2, 2*index:2*index+2] += blocks[index]
        hessian += np.eye(2*n) * 1e-5
        step = np.linalg.solve(hessian, -gradient).reshape(n, 2)
        accepted = False
        for alpha in (1, .5, .25, .125, .0625, .03125, .015625):
            trial = np.clip(q + alpha * step, 0, 1)
            cost = objective(trial)
            if cost < history[-1] - 1e-10:
                q = trial
                history.append(cost)
                accepted = True
                break
        if not accepted:
            break
    predicted = grid.forward(q)[0]
    angles = grid.lower + q * grid.span
    return angles, {'objective': history, 'predictedHand': predicted.tolist(),
                    'predictedMeanError': float(np.mean(np.linalg.norm(predicted-targets, axis=1))),
                    'velocityWeight': velocity_weight, 'accelerationWeight': acceleration_weight,
                    'scope': '30Hz interpolated grid; not anatomical or visual quality certification'}
