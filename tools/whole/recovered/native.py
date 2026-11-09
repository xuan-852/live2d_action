"""Two execution backends behind one skeleton interface.

Both backends answer the same question -- "can you draw this skeleton, and if not,
what exactly is missing?" -- and return the same record shape. They do not share
degrees of freedom, and the caller must not assume they do.

  NativeArm   the model's own arm channels, measured through Cubism Core.
              Rigid two-link planar chain: it matches *directions* exactly and
              cannot foreshorten a segment (measured: forearm drawn length is
              constant to 0.2% across the whole envelope).
  PatchArm    the isolated mesh postprocess on the other arm. It can shorten a
              segment within measured bounds, so it also carries the source's
              projected ratio -- and reports when the source needs more shortening
              than those bounds allow.

Targets carry *ratios*, not lengths, so each backend converts them with its own
measured bone lengths. Nothing here writes to the model.
"""
import math

import numpy as np


def signed_degrees(v):
    return ((v + 180.) % 360.) - 180.


def wrap_continuous(values):
    """Unwrap a sequence of signed angles so each step is the shortest one."""
    out, previous = [], None
    for value in values:
        if previous is None:
            out.append(float(value))
        else:
            out.append(float(value + 360. * round((previous - value) / 360.)))
        previous = out[-1]
    return out


def interp_inverse(grid, values, target):
    """Invert a monotonic measured map; returns (parameter, achieved, clamped)."""
    grid = np.asarray(grid, dtype=float)
    values = np.asarray(values, dtype=float)
    if target <= values[0]:
        return float(grid[0]), float(values[0]), True
    if target >= values[-1]:
        return float(grid[-1]), float(values[-1]), True
    return float(np.interp(target, values, grid)), float(target), False


def unit2(heading_deg):
    a = math.radians(heading_deg)
    return np.array([math.cos(a), math.sin(a)])


def common_residuals(target, drawn_upper, drawn_fore, up_h, fore_h, bend, palm_rel,
                     wrist, tol_deg, wrist_target):
    up_err = abs(signed_degrees(up_h - target['upperHeadingDeg']))
    fore_err = abs(signed_degrees(fore_h - target['forearmHeadingDeg']))
    bend_err = abs(signed_degrees(bend - signed_degrees(target['bendUnwrappedDeg'])))
    palm_err = (abs(signed_degrees(palm_rel - target['palmRelativeDeg']))
                if target.get('palmRelativeDeg') is not None else None)
    residuals = dict(upperDirectionDeg=float(up_err), forearmDirectionDeg=float(fore_err),
                     bendDeg=float(bend_err),
                     palmRelativeDeg=(None if palm_err is None else float(palm_err)),
                     drawnVsSourceLength=float(drawn_upper + drawn_fore),
                     wristProxyPosition=float(np.linalg.norm(np.asarray(wrist) - np.asarray(wrist_target))))
    reasons = []
    if up_err > tol_deg:
        reasons.append('upper-direction-residual')
    if fore_err > tol_deg:
        reasons.append('forearm-direction-residual')
    if palm_err is not None and palm_err > tol_deg * 1.5:
        reasons.append('palm-heading-residual')
    return residuals, reasons


class NativeArm:
    """The model's own arm, solved over all four arm channels.

    Param94 rotates the upper arm, Param95 foreshortens it, Param97 bends the elbow,
    Param98 foreshortens the forearm. All four are measured tables in the
    calibration, so the solver starts from a closed-form seed and then refines
    against the actual Core readback. Nothing here is a fitted coefficient that was
    tuned to a clip.
    """

    name = 'native'
    side = 'anatomicalLeft'

    def __init__(self, model, anchors, calibration):
        self.model = model
        self.anchors = anchors
        self.cal = calibration
        maps = calibration['controlMaps']
        perspective = calibration['perspective']
        self.grid94 = np.asarray(perspective['grids']['param94'], dtype=float)
        self.grid95 = np.asarray(perspective['grids']['param95'], dtype=float)
        self.grid97 = np.asarray(perspective['grids']['param97'], dtype=float)
        self.grid98 = np.asarray(perspective['grids']['param98'], dtype=float)
        # table[q] is a list over the partner grid of [lengthRatio, headingOrBend]
        self.upper_table = {float(q): np.asarray(rows, dtype=float)
                            for q, rows in perspective['upper']['table'].items()}
        self.fore_table = {float(q): np.asarray(rows, dtype=float)
                           for q, rows in perspective['forearm']['table'].items()}
        self.tol_deg = float(calibration['precision']['directionToleranceDegrees'])
        self.tol_ratio = float(calibration['precision']['foreshorteningTolerance'])
        self.lengths = calibration['lengths']
        self.rest_shoulder = np.asarray(calibration['neutralLandmarks']['shoulder'], dtype=float)
        # Solver bounds sit a little inside the hard channel limits. The margin is
        # measured, not stylistic: on the exact limit the drawn bend is folded, so a
        # sub-degree change in another channel flips it by more than ten degrees.
        margin = {'Param94': .5, 'Param95': .5, 'Param97': .2, 'Param98': .5}
        self.limits = {k: [float(model.lo[model.idx[k]]) + margin[k],
                           float(model.hi[model.idx[k]]) - margin[k]]
                       for k in ('Param94', 'Param95', 'Param97', 'Param98')}
        self.hard_limits = {k: [float(model.lo[model.idx[k]]), float(model.hi[model.idx[k]])]
                            for k in ('Param94', 'Param95', 'Param97', 'Param98')}
        self.margin = margin
        self.unreachable_arcs = calibration['bendSurface']['surface2d']['unreachableArcs']

    # -- measured table access ---------------------------------------------
    def _row(self, table, key_grid, key):
        keys = sorted(table)
        key = min(max(key, keys[0]), keys[-1])
        if key <= keys[0]:
            return table[keys[0]]
        if key >= keys[-1]:
            return table[keys[-1]]
        hi = next(i for i, k in enumerate(keys) if k >= key)
        lo = hi - 1
        w = (key - keys[lo]) / (keys[hi] - keys[lo])
        return table[keys[lo]] * (1 - w) + table[keys[hi]] * w

    def sample_upper(self, q94, q95):
        """(drawn upper length ratio, upper heading) at one channel pair."""
        row = self._row(self.upper_table, self.grid95, q95)
        return (float(np.interp(q94, self.grid94, row[:, 0])),
                float(np.interp(q94, self.grid94, row[:, 1])))

    def sample_fore(self, q97, q98):
        """(drawn forearm length ratio, signed bend) at one channel pair."""
        row = self._row(self.fore_table, self.grid98, q98)
        return (float(np.interp(q97, self.grid97, row[:, 0])),
                float(np.interp(q97, self.grid97, row[:, 1])))

    # -- actual measurement -------------------------------------------------
    def measure(self, pose):
        self.model.set(pose)
        p = self.anchors.read()
        u, f, h = p['elbow'] - p['shoulder'], p['wrist'] - p['elbow'], p['palm'] - p['wrist']
        up_h = math.degrees(math.atan2(u[1], u[0]))
        fore_h = math.degrees(math.atan2(f[1], f[0]))
        return dict(points=p, upper=u, forearm=f, palm=h,
                    upperHeadingDeg=up_h, forearmHeadingDeg=fore_h,
                    bendDeg=signed_degrees(fore_h - up_h),
                    palmRelativeDeg=signed_degrees(math.degrees(math.atan2(h[1], h[0])) - fore_h),
                    upperRatio=float(np.linalg.norm(u)) / self.lengths['upper'],
                    forearmRatio=float(np.linalg.norm(f)) / self.lengths['forearm'])

    def expected_wrist(self, target):
        up = self.lengths['upper'] * target['upperProjectedRatio']
        fore = self.lengths['forearm'] * target['forearmProjectedRatio']
        return (self.rest_shoulder + up * unit2(target['upperHeadingDeg'])
                + fore * unit2(target['forearmHeadingDeg']))

    def in_unreachable_arc(self, bend):
        for lo, hi in self.unreachable_arcs:
            if lo <= bend <= hi:
                return True
        return False

    # -- solve --------------------------------------------------------------
    def _seed_upper(self, target, previous):
        q95 = float(previous['Param95']) if previous else 0.
        q94 = float(previous['Param94']) if previous else 0.
        for _ in range(4):
            heading_row = np.array([self.sample_upper(q, q95)[1] for q in self.grid94])
            q94, _, _ = interp_inverse(self.grid94, heading_row, target['upperHeadingDeg'])
            ratio_col = np.array([self.sample_upper(q94, q)[0] for q in self.grid95])
            # Ratio falls as Param95 rises, so invert on the flipped axis.
            q95, _, _ = interp_inverse(self.grid95[::-1], ratio_col[::-1],
                                       target['upperProjectedRatio'])
        return q94, q95

    def _seed_bend(self, target, previous):
        want = float(target['bendUnwrappedDeg'])
        q98 = float(previous['Param98']) if previous else 0.
        q97 = float(previous['Param97']) if previous else 0.
        for _ in range(4):
            row = np.array([self.sample_fore(q, q98)[1] for q in self.grid97])
            unwrapped = np.asarray(wrap_continuous(row.tolist()), dtype=float)
            # The bend circle has a small unreachable arc, so the two ends are not the
            # same configuration. Both copies are candidates; continuity decides.
            candidates = []
            for turns in (-1, 0, 1):
                shifted = want + 360. * turns
                if unwrapped[0] - 360. <= shifted <= unwrapped[-1] + 360.:
                    value = min(max(shifted, unwrapped[0]), unwrapped[-1])
                    q = float(np.interp(value, unwrapped, self.grid97))
                    error = abs(signed_degrees(value - want))
                    if not any(abs(q - c[0]) < 1e-6 for c in candidates):
                        candidates.append((q, error))
            inside = [c for c in candidates if c[1] <= self.tol_deg]
            pool = inside or candidates
            q97 = (min(pool, key=lambda c: abs(c[0] - q97))[0] if previous is not None
                   else min(pool, key=lambda c: c[1])[0])
            ratio_col = np.array([self.sample_fore(q97, q)[0] for q in self.grid98])
            q98, _, _ = interp_inverse(self.grid98, ratio_col, target['forearmProjectedRatio'])
        return q97, q98

    def _residual(self, q, target):
        pose = {'Param94': q[0], 'Param95': q[1], 'Param97': q[2], 'Param98': q[3]}
        m = self.measure(pose)
        return np.array([
            signed_degrees(m['upperHeadingDeg'] - target['upperHeadingDeg']) / self.tol_deg,
            signed_degrees(m['forearmHeadingDeg'] - target['forearmHeadingDeg']) / self.tol_deg,
            (m['upperRatio'] - target['upperProjectedRatio']) / self.tol_ratio,
            (m['forearmRatio'] - target['forearmProjectedRatio']) / self.tol_ratio])

    def solve(self, target, previous=None):
        q94, q95 = self._seed_upper(target, previous)
        q97, q98 = self._seed_bend(target, previous)
        q = np.array([q94, q95, q97, q98], dtype=float)
        lo = np.array([self.limits[k][0] for k in ('Param94', 'Param95', 'Param97', 'Param98')])
        hi = np.array([self.limits[k][1] for k in ('Param94', 'Param95', 'Param97', 'Param98')])
        q = np.clip(q, lo, hi)
        # Damped Gauss-Newton on the actual Core, so the result is measured, not
        # extrapolated from a table. Bounds are the margin-shrunk solver bounds.
        r = self._residual(q, target)
        damping = 1e-3
        for _ in range(14):
            jac = np.empty((len(r), 4))
            for j in range(4):
                probe = q.copy()
                probe[j] = min(hi[j], q[j] + .5) if q[j] + .5 <= hi[j] else q[j] - .5
                jac[:, j] = (self._residual(probe, target) - r) / (probe[j] - q[j])
            try:
                step = np.linalg.solve(jac.T @ jac + damping * np.eye(4), -jac.T @ r)
            except np.linalg.LinAlgError:
                break
            nxt = np.clip(q + np.clip(step, -6., 6.), lo, hi)
            nr = self._residual(nxt, target)
            if nr @ nr < r @ r:
                q, r = nxt, nr
                damping = max(1e-9, damping * .3)
            else:
                damping = min(1e3, damping * 5.)
            if np.linalg.norm(np.clip(step, -6., 6.)) < 1e-4:
                break

        pose = {'Param94': float(q[0]), 'Param95': float(q[1]),
                'Param97': float(q[2]), 'Param98': float(q[3])}
        p99 = 0.
        if target.get('palmRelativeDeg') is not None:
            # Seed from the measured palm map, then refine on the real mesh: the palm
            # response shifts with the arm pose, so the map alone leaves a coupling
            # error that is measured here rather than left in the residual.
            grid = np.asarray(self.cal['controlMaps']['palmRelativeHeading']['grid'], dtype=float)
            values = np.asarray(self.cal['controlMaps']['palmRelativeHeading']['values'], dtype=float)
            p99, _, _ = interp_inverse(grid[::-1], values[::-1], target['palmRelativeDeg'])
            best, best_error = float(p99), None
            lo99, hi99 = self.limits.get('Param99', [-30., 30.])
            for step in (6., 2., .5, .1):
                improved = True
                while improved:
                    improved = False
                    for candidate in (best - step, best + step):
                        if not lo99 <= candidate <= hi99:
                            continue
                        probe = dict(pose)
                        probe['Param99'] = float(candidate)
                        error = abs(signed_degrees(
                            self.measure(probe)['palmRelativeDeg'] - target['palmRelativeDeg']))
                        if best_error is None or error < best_error - 1e-9:
                            best, best_error = float(candidate), error
                            improved = True
            p99 = best
        pose['Param99'] = float(p99)
        pose['Param100'] = 0.

        m = self.measure(pose)
        reasons, clamped = [], []
        up_err = abs(signed_degrees(m['upperHeadingDeg'] - target['upperHeadingDeg']))
        fore_err = abs(signed_degrees(m['forearmHeadingDeg'] - target['forearmHeadingDeg']))
        up_ratio_err = abs(m['upperRatio'] - target['upperProjectedRatio'])
        fore_ratio_err = abs(m['forearmRatio'] - target['forearmProjectedRatio'])
        palm_err = (abs(signed_degrees(m['palmRelativeDeg'] - target['palmRelativeDeg']))
                    if target.get('palmRelativeDeg') is not None else None)
        if up_err > self.tol_deg:
            reasons.append('upper-direction-residual')
        if fore_err > self.tol_deg:
            reasons.append('forearm-direction-residual')
        if up_ratio_err > self.tol_ratio:
            reasons.append('upper-foreshortening-residual')
        if fore_ratio_err > self.tol_ratio:
            reasons.append('forearm-foreshortening-residual')
        if palm_err is not None and palm_err > self.tol_deg * 1.5:
            reasons.append('palm-heading-residual')
        if self.in_unreachable_arc(target['bendUnwrappedDeg']):
            reasons.append('bend-unreachable-arc')
        if abs(q[0] - lo[0]) < 1e-6 or abs(q[0] - hi[0]) < 1e-6:
            clamped.append('Param94')
        if abs(q[1] - lo[1]) < 1e-6 or abs(q[1] - hi[1]) < 1e-6:
            clamped.append('Param95')
        if abs(q[2] - lo[2]) < 1e-6 or abs(q[2] - hi[2]) < 1e-6:
            clamped.append('Param97')
        if abs(q[3] - lo[3]) < 1e-6 or abs(q[3] - hi[3]) < 1e-6:
            clamped.append('Param98')
        if clamped:
            reasons = ['channel-at-limit:' + ','.join(clamped)] + reasons
        # A pose sitting on the fold is not a pose we can draw smoothly, so it is
        # named even when every numeric residual looks small.
        if abs(q[2]) > self.limits['Param97'][1] - 1e-6 or abs(q[2]) < self.limits['Param97'][0] + 1e-6:
            if 'bend-fold-margin' not in reasons:
                reasons.append('bend-fold-margin')
        wrist_target = self.expected_wrist(target)
        residuals = dict(upperDirectionDeg=float(up_err), forearmDirectionDeg=float(fore_err),
                         bendDeg=float(abs(signed_degrees(m['bendDeg'] -
                                                           signed_degrees(target['bendUnwrappedDeg'])))),
                         palmRelativeDeg=(None if palm_err is None else float(palm_err)),
                         upperLengthRatio=float(up_ratio_err),
                         forearmLengthRatio=float(fore_ratio_err),
                         drawnVsSourceLength=float(
                             abs(m['upperRatio'] - target['upperProjectedRatio']) * self.lengths['upper'] +
                             abs(m['forearmRatio'] - target['forearmProjectedRatio']) * self.lengths['forearm']),
                         wristProxyPosition=float(np.linalg.norm(m['points']['wrist'] - wrist_target)))
        worst = max(up_err / self.tol_deg, fore_err / self.tol_deg,
                    up_ratio_err / self.tol_ratio, fore_ratio_err / self.tol_ratio)
        status = 'Unsupported' if worst > 3. else ('Degraded' if reasons else 'Ok')
        achieved = dict(shoulder=m['points']['shoulder'].tolist(),
                        elbow=m['points']['elbow'].tolist(),
                        wrist=m['points']['wrist'].tolist(),
                        palm=m['points']['palm'].tolist(),
                        upperHeadingDeg=float(m['upperHeadingDeg']),
                        forearmHeadingDeg=float(m['forearmHeadingDeg']),
                        bendDeg=float(m['bendDeg']), palmRelativeDeg=float(m['palmRelativeDeg']),
                        palmHeadingDeg=float(math.degrees(math.atan2(m['palm'][1], m['palm'][0]))),
                        drawnUpperLength=float(np.linalg.norm(m['upper'])),
                        drawnForearmLength=float(np.linalg.norm(m['forearm'])),
                        upperLengthRatio=float(m['upperRatio']),
                        forearmLengthRatio=float(m['forearmRatio']))
        return dict(backend=self.name, side=self.side, pose=pose, achieved=achieved,
                    residuals=residuals, status=status, reasons=reasons,
                    achievedFrom='actual Cubism Core mesh readback',
                    step=None if previous is None else
                    {k: float(pose[k] - previous.get(k, 0.)) for k in pose})
