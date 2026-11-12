"""Restricted 2-D material-node controller. No learned inverse or clip scheduler.

Positions: model-local x right/y up; native=anatomical left, patch=right.
Targets are timestamped and never retimed. Infeasibility and SDK disturbance
are observable results, not silently repaired input trajectories.
"""
from dataclasses import dataclass
from itertools import combinations
from itertools import product
import numpy as np

DT = 1 / 30

# Explicit execution budgets, not a claim about model limits or expression force.
# Raising speed by k also raises acceleration by k^2, preserving the nominal
# stopping distance. Geometry, branch state and exact-node checks remain active.
SPEED_PROFILES = {
    'conservative': dict(speed=45., acceleration=180., node_speed=.24, node_acceleration=1.8, feedback_gain=.2),
    'responsive': dict(speed=90., acceleration=720., node_speed=.48, node_acceleration=7.2, feedback_gain=1-.8**2),
    'fast': dict(speed=135., acceleration=1620., node_speed=.72, node_acceleration=16.2, feedback_gain=1-.8**3),
}
DEFAULT_PROFILE = 'fast'


def speed_limits(profile):
    if profile not in SPEED_PROFILES:
        raise ValueError('unknown speed profile: '+str(profile))
    return SPEED_PROFILES[profile].copy()


def wrap(a):
    return (np.asarray(a) + 180) % 360 - 180


def bend(nodes):
    u, v = np.diff(nodes[:3], axis=0)
    return float(wrap(np.degrees(np.arctan2(v[1], v[0]) - np.arctan2(u[1], u[0]))))


def patch_forward(home, q):
    out = np.asarray(home, dtype=float).copy()
    for j in range(1, 4):
        a = np.radians(sum(q[:j])); c, s = np.cos(a), np.sin(a)
        out[j] = out[j-1] + np.array([[c, -s], [s, c]]) @ (home[j]-home[j-1])
    return out


@dataclass
class Target:
    wrist: np.ndarray
    time: float
    elbow: np.ndarray = None
    palm_heading: float = None
    shoulder: np.ndarray = None
    branch: int = 0
    frame: str = 'model-local-xy'
    mode: str = 'hold'
    velocity: np.ndarray = None
    angular_velocity: float = 0.

    def __post_init__(self):
        if self.frame != 'model-local-xy':
            raise ValueError('unsupported coordinate frame')
        for key in ('wrist', 'elbow', 'shoulder', 'velocity'):
            value = getattr(self, key)
            if value is not None:
                value = np.asarray(value, dtype=float)
                if value.shape != (2,) or not np.isfinite(value).all():
                    raise ValueError(key)
                setattr(self, key, value)
        if not np.isfinite(self.time) or (self.palm_heading is not None and not np.isfinite(self.palm_heading)):
            raise ValueError('target time/heading')
        if self.branch not in (-1, 0, 1):
            raise ValueError('branch')
        if self.mode not in ('hold', 'pass', 'track'):
            raise ValueError('execution mode')
        if self.mode == 'pass' and self.velocity is None:
            raise ValueError('pass requires terminal wrist velocity')
        if self.mode == 'hold' and ((self.velocity is not None and np.linalg.norm(self.velocity) > 1e-12) or self.angular_velocity != 0):
            raise ValueError('hold requires zero terminal velocity')
        if not np.isfinite(self.angular_velocity):
            raise ValueError('angular velocity')


def constrained_step(h, g, a, b):
    """Exact active-face enumeration for <=3 actuators, no extra dependency.

    Enumerating faces avoids mistaking slow convergence of nearly parallel
    material-node constraints for physical infeasibility. No accumulated steps.
    """
    inv = np.linalg.inv(h)
    free = inv @ g
    norm = np.linalg.norm(a, axis=1)
    if np.any((norm < 1e-12) & (b < -1e-8)):
        return np.zeros(len(g)), False
    keep = norm > 1e-12
    a, b = a[keep]/norm[keep, None], b[keep]/norm[keep]
    if np.all(a @ free <= b+1e-8):
        return free, True
    if len(g) > 3:
        # Small dual working set for the five-axis native backend. Return only
        # with primal feasibility and nonnegative KKT multipliers; otherwise use
        # the exact face enumeration below, never an unchecked approximate step.
        active = []; x = free.copy()
        for _ in range(80):
            violation = a@x-b
            if np.max(violation) <= 1e-7:
                return x, True
            index = int(np.argmax(violation))
            if index in active:
                break
            active.append(index)
            for _ in range(len(g)+5):
                face = a[active]; gram = face@inv@face.T
                multipliers = np.linalg.lstsq(gram, face@free-b[active], rcond=1e-10)[0]
                if np.min(multipliers) >= -1e-8:
                    x = free-inv@face.T@multipliers
                    break
                active.pop(int(np.argmin(multipliers)))
                if not active:
                    x = free.copy(); break
    candidates = []
    for size in range(1, len(g)+1):
        indices = np.array(list(combinations(range(len(b)), size)))
        faces = a[indices]
        gram = faces @ inv @ faces.transpose(0, 2, 1)
        scale = np.prod(np.diagonal(gram, axis1=1, axis2=2), axis=1)
        good = np.linalg.det(gram) > 1e-10*scale
        faces, gram, indices = faces[good], gram[good], indices[good]
        if not len(gram):
            continue
        multipliers = np.linalg.solve(gram, (faces @ free-b[indices])[..., None])[..., 0]
        x = free - (inv @ faces.transpose(0, 2, 1) @ multipliers[..., None])[..., 0]
        feasible = np.all(x @ a.T <= b+1e-7, axis=1)
        candidates.extend(x[feasible])
    if not candidates:
        return np.zeros(len(g)), False
    candidates = np.array(candidates)
    cost = .5*np.sum((candidates @ h)*candidates, axis=1)-candidates @ g
    return candidates[np.argmin(cost)], True


class ArmController:
    """A local, branch-retaining servo; does not certify global reachability.

    forward(q) must consume the latest physical context. It predicts next material
    nodes, not SDK physics evolution. Feedback correction is local and expires on
    the next observation. Projection is locked to identity in this version.
    """
    def __init__(self, side, forward, lo, hi, q, nodes, *,
                 speed=45., acceleration=180., node_speed=.24, node_acceleration=1.8, feedback_gain=.2):
        if side not in ('native', 'patch'):
            raise ValueError('side identity')
        self.side, self.forward = side, forward
        self.lo, self.hi = np.asarray(lo), np.asarray(hi)
        self.q, self.nodes = np.array(q, float), np.array(nodes, float)
        self.dof = len(self.q)
        self.q_velocity = np.zeros(self.dof); self.velocity = np.zeros((4, 2))
        self.speed, self.acceleration = speed, acceleration
        self.node_speed, self.node_acceleration = node_speed, node_acceleration
        if not np.isfinite(feedback_gain) or not 0 < feedback_gain <= 1:
            raise ValueError('feedback gain')
        self.feedback_gain = feedback_gain
        self.branch = int(np.sign(bend(self.nodes)))
        self.pending = None; self.last = None; self.limited_frames = 0
        self.geometry = None

    def _objective(self, target, now, q, actual, predict, jac, desired):
        """Legacy objective, kept explicit for controlled same-request comparisons."""
        weights = np.zeros((4, 2)); weights[2] = 1
        if target.elbow is not None:
            weights[1] = 1
        if target.palm_heading is not None:
            weights[3] = 1
        w = weights.ravel()
        task_jac = jac.copy(); error = (desired-actual).ravel()
        reference_step = np.zeros(8)
        if self.last is not None:
            reference_step = (desired-np.array(self.last['requested'])).ravel()
        if target.palm_heading is not None:
            task_jac[6:8] -= jac[4:6]
            error[6:8] -= error[4:6]
            reference_step[6:8] -= reference_step[4:6]
        h = task_jac.T @ (w[:, None]*task_jac) + 1e-8*np.eye(self.dof)
        g = task_jac.T @ (w*(error*self.feedback_gain+reference_step*(1-self.feedback_gain)))
        return h, g

    def command(self, target, now):
        self.pending = target
        q, actual = self.q.copy(), self.nodes.copy()
        base = self.forward(q)
        # The last actual material readback is the linearization origin.
        def predict(z):
            return actual + self.forward(z) - base
        jac = []
        for j in range(self.dof):
            step = .02 if q[j]+.02 <= self.hi[j] else -.02
            jac.append((predict(q+np.eye(self.dof)[j]*step)-actual).ravel()/step)
        jac = np.stack(jac, axis=1)
        desired = actual.copy(); desired[2] = target.wrist
        if target.shoulder is not None:
            desired[0] = target.shoulder
        if target.elbow is not None:
            desired[1] = target.elbow
        if target.palm_heading is not None:
            r = np.radians(target.palm_heading)
            desired[3] = target.wrist + np.linalg.norm(actual[3]-actual[2])*np.array([np.cos(r), np.sin(r)])
        self._policy_info = {}
        h, g = self._objective(target, now, q, actual, predict, jac, desired)
        # Optional command-side reserve for float32 SDK actuator readback. The
        # declared actual-state acceleration limit and observer stay unchanged.
        execution_acceleration = max(0., self.acceleration-getattr(self, 'actuator_acceleration_reserve', 0.))
        lower = np.maximum.reduce([self.lo-q, np.full(self.dof, -self.speed*DT),
                                   self.q_velocity*DT-execution_acceleration*DT**2])
        upper = np.minimum.reduce([self.hi-q, np.full(self.dof, self.speed*DT),
                                   self.q_velocity*DT+execution_acceleration*DT**2])
        # Reserve stopping distance at actuator bounds (discrete braking).
        # Braking also has to fit the Cartesian acceleration budget. A parameter
        # boundary must not be approached assuming actuator-only deceleration.
        gain = np.max(np.linalg.norm(jac.reshape(4, 2, self.dof), axis=1), axis=0)
        braking = np.minimum(self.acceleration, self.node_acceleration/(3*np.maximum(gain, 1e-8)))
        if getattr(self, 'exact_native_stop', False):
            # Software deformation axes have no independent mechanical inertia.
            # Use their declared acceleration for the candidate stop; the actual
            # nonlinear node trajectory is still checked below at every step.
            braking = np.full(self.dof, execution_acceleration)
        ad = braking*DT**2
        upper = np.minimum(upper, -.5*ad+np.sqrt(.25*ad**2+2*ad*np.maximum(0, self.hi-q)))
        lower = np.maximum(lower, .5*ad-np.sqrt(.25*ad**2+2*ad*np.maximum(0, q-self.lo)))
        ns = self.node_speed*DT*.95 / np.sqrt(2)
        na = self.node_acceleration*DT**2*.85 / np.sqrt(2)
        vlo = np.maximum(-ns, self.velocity.ravel()*DT-na)
        vhi = np.minimum(ns, self.velocity.ravel()*DT+na)
        a = np.vstack([np.eye(self.dof), -np.eye(self.dof), jac, -jac])
        b = np.r_[upper, -lower, vhi, -vlo]
        def mesh_constraint(a, b):
            if self.geometry is not None:
                a = np.vstack([a, -mesh_gradients])
                b = np.r_[b, mesh_areas-.55]
            return a, b
        if self.geometry is not None:
            mesh_areas = np.asarray(self.geometry.get('meshAreaRatios', [self.geometry['minTriangleAreaRatio']]))
            mesh_gradients = np.asarray(self.geometry.get('meshAreaGradients', self.geometry['areaGradient'])).reshape(-1, self.dof)
        a, b = mesh_constraint(a, b)
        delta, feasible = constrained_step(h, g, a, b)
        # Sequential constraint linearization always uses the SAME frame origin
        # and budgets. These are not successive actuator steps. Resolve Core
        # curvature before executing once, rather than smoothing after a jump.
        for _ in range(5):
            if not feasible:
                break
            point = q+delta; predicted = predict(point)
            movement = (predicted-actual).ravel()
            if np.all(movement <= vhi+1e-7) and np.all(movement >= vlo-1e-7):
                break
            local_jac = []
            for j in range(self.dof):
                step = .02 if point[j]+.02 <= self.hi[j] else -.02
                local_jac.append((predict(point+np.eye(self.dof)[j]*step)-predicted).ravel()/step)
            local_jac = np.stack(local_jac, axis=1)
            offset = movement-local_jac @ delta
            a = np.vstack([np.eye(self.dof), -np.eye(self.dof), local_jac, -local_jac])
            b = np.r_[upper, -lower, vhi-offset, -vlo+offset]
            a, b = mesh_constraint(a, b)
            delta, feasible = constrained_step(h, g, a, b)
        reasons = list(self._policy_info.get('reasons', []))
        def safe(d):
            if self.geometry is not None and np.min(mesh_areas+mesh_gradients @ d) < .55-1e-6:
                return False
            movement = predict(q+d)-actual
            immediate = (np.max(np.linalg.norm(movement, axis=1)) <= self.node_speed*DT+1e-7 and
                    np.max(np.linalg.norm(movement/DT-self.velocity, axis=1)) <= self.node_acceleration*DT+1e-5 and
                    np.all(d >= lower-1e-7) and np.all(d <= upper+1e-7))
            if not immediate:
                return False
            # Keep a feasible stop in the frozen local context. This detects
            # derivative changes ahead (e.g. native shoulder crossing zero)
            # before momentum makes the next frame's constraints inconsistent.
            steps = max(1, int(np.ceil(np.max(abs(d)/np.maximum(braking*DT**2*.8, 1e-8)))))
            prior_nodes, prior_step, point = actual+movement, movement, q+d
            for k in range(1, steps+1):
                next_step = d*(1-k/steps)
                point = point+next_step
                if self.geometry is not None and np.min(mesh_areas+mesh_gradients @ (point-q)) < .55-1e-6:
                    return False
                if np.any(point < self.lo-1e-7) or np.any(point > self.hi+1e-7):
                    return False
                next_nodes = predict(point); moved = next_nodes-prior_nodes
                if (np.max(np.linalg.norm(moved, axis=1)) > self.node_speed*DT+1e-7 or
                    np.max(np.linalg.norm(moved-prior_step, axis=1)) > self.node_acceleration*DT**2+1e-6):
                    return False
                prior_nodes, prior_step = next_nodes, moved
            return True
        finite = getattr(self, 'finite_target_step', None)
        if finite is not None and safe(finite):
            # A calibrated finite IK candidate can be safe even when a local
            # Jacobian across a Core piece boundary predicts the wrong motion.
            delta = finite.copy(); feasible = True
            self._policy_info['finiteTargetAccepted'] = True
        if not feasible or not safe(delta):
            # At a Core piece boundary a local derivative can have no feasible
            # solution although a nearby finite step does. Test a small LOCAL
            # actuator lattice inside this frame's box, not a parameter scan.
            # The exact forward nodes decide safety; no temporal post-filter.
            candidates = [np.clip(self.q_velocity*DT, lower, upper)]
            if self.dof <= 3:
                candidates += [np.array(d) for d in product(*(np.linspace(l, u, 7) for l, u in zip(lower, upper)))]
            else:
                candidates += [np.clip(delta*scale, lower, upper) for scale in (0., .25, .5, .75, 1.)]
                candidates += [np.clip(self.q_velocity*DT + axis*step, lower, upper)
                    for axis in np.eye(self.dof) for step in (-1., -.25, .25, 1.)]
            candidates = [d for d in candidates if safe(d)] if np.all(lower <= upper) else []
            if candidates:
                delta = min(candidates, key=lambda d: .5*d @ h @ d-g @ d)
                feasible = True; reasons.append('nonlinear-feasibility-recovery')
            else:
                delta = np.clip(np.zeros(self.dof), self.lo-q, self.hi-q)
                feasible = False; reasons.append('infeasible-continuity-envelope')
        command = q+delta; ideal = predict(command)
        limiting = []
        if np.max(abs(delta)) >= self.speed*DT-1e-4:
            limiting.append('parameter-speed')
        if np.max(abs(delta-self.q_velocity*DT)) >= self.acceleration*DT**2-1e-4:
            limiting.append('parameter-acceleration')
        movement = (ideal-actual).ravel()
        if np.any(abs(movement-vhi) < 1e-6) or np.any(abs(movement-vlo) < 1e-6):
            limiting.append('cartesian-envelope')
        if self.geometry is not None and np.min(mesh_areas+mesh_gradients @ delta) < .551:
            limiting.append('mesh-area')
        upper_length, fore_length = np.linalg.norm(np.diff(actual[:3], axis=0), axis=1)
        # Conservative local envelope for this identity-projection, frozen-body
        # adapter. Not a statement about the model's full 244-parameter capacity.
        if np.linalg.norm(target.wrist-actual[0]) > 1.1*(upper_length+fore_length)+.02:
            reasons.append('outside-local-fixed-length-reach-envelope')
        if target.elbow is not None and np.linalg.norm(target.elbow-actual[0]) > 1.1*upper_length+.02:
            reasons.append('elbow-outside-local-length-envelope')
        if self.geometry is not None and self.geometry['minTriangleAreaRatio'] < .57:
            reasons.append('mesh-area-floor')
        node_step = ideal-actual
        if np.max(np.linalg.norm(node_step, axis=1)) > self.node_speed*DT+1e-5:
            reasons.append('predicted-node-speed')
        if np.max(np.linalg.norm(node_step/DT-self.velocity, axis=1)) > self.node_acceleration*DT+1e-4:
            reasons.append('predicted-node-acceleration')
        if target.shoulder is not None and np.linalg.norm(target.shoulder-actual[0]) > .003:
            reasons.append('shoulder-translation-unsupported')
        branch = int(np.sign(bend(ideal)))
        if target.branch and branch != target.branch:
            reasons.append('requested-branch-conflict')
        if self.branch and branch and branch != self.branch and abs(bend(actual)) > 3:
            reasons.append('branch-change')
        if (np.any(command < self.lo+1e-4) or np.any(command > self.hi-1e-4)) and np.max(np.linalg.norm(desired[1:]-actual[1:], axis=1)) > .005:
            reasons.append('actuator-bound')
        self.last = dict(command=command.tolist(), requested=desired.tolist(), ideal=ideal.tolist(),
                         reasons=reasons, deadline=target.time, branch=branch, now=now,
                         requestedBranch=target.branch, qpFeasible=feasible, limitingConstraints=limiting,
                         mode=target.mode, policy=self._policy_info)
        return command.copy(), self.last

    def observe(self, applied_q, nodes, geometry=None):
        """Call once per executed SDK frame, with final mesh material nodes."""
        q, nodes = np.asarray(applied_q), np.asarray(nodes)
        if not np.isfinite(q).all() or not np.isfinite(nodes).all():
            raise ValueError('nonfinite SDK feedback')
        v = (nodes-self.nodes)/DT; qv = (q-self.q)/DT
        reasons = self.last['reasons'].copy()
        if geometry is not None:
            gradient = np.asarray(geometry['areaGradient'])
            if gradient.shape != (3,) or not np.isfinite(gradient).all() or not np.isfinite(geometry['minTriangleAreaRatio']):
                raise ValueError('invalid mesh geometry feedback')
            self.geometry = geometry
            if geometry['minTriangleAreaRatio'] < .5 or geometry['flippedTriangles']:
                reasons.append('actual-mesh-geometry-violation')
        metrics = dict(maxNodeStep=float(np.max(np.linalg.norm(nodes-self.nodes, axis=1))),
                       maxNodeSpeed=float(np.max(np.linalg.norm(v, axis=1))),
                       maxNodeAcceleration=float(np.max(np.linalg.norm(v-self.velocity, axis=1))/DT),
                       maxParameterStep=float(np.max(abs(q-self.q))),
                       maxParameterAcceleration=float(np.max(abs(qv-self.q_velocity))/DT),
                       modelResidual=float(np.max(np.linalg.norm(nodes-np.array(self.last['ideal']), axis=1))))
        for key, limit in [('maxNodeSpeed', self.node_speed), ('maxNodeAcceleration', self.node_acceleration),
                           ('maxParameterStep', self.speed*DT), ('maxParameterAcceleration', self.acceleration)]:
            if metrics[key] > limit+1e-3:
                reasons.append('actual-'+key+'-violation')
        wrist_error = float(np.linalg.norm(nodes[2]-self.pending.wrist))
        elbow_error = None if self.pending.elbow is None else float(np.linalg.norm(nodes[1]-self.pending.elbow))
        heading = np.degrees(np.arctan2(*(nodes[3]-nodes[2])[::-1]))
        palm_error = None if self.pending.palm_heading is None else float(abs(wrap(heading-self.pending.palm_heading)))
        reached = wrist_error <= .005 and (elbow_error is None or elbow_error <= .005) and (palm_error is None or palm_error <= 4)
        if self.last['now'] >= self.pending.time and not reached:
            reasons.append('deadline-missed')
        if not reached and np.max(abs(q-self.q)) < .002:
            reasons.append('local-stall-or-constraint-conflict')
        if not self.last['qpFeasible'] or any(r.startswith('actual-') for r in reasons):
            status = 'failed'
        elif reasons:
            status = 'limited'
        else:
            status = 'reached' if reached else 'tracking'
        self.limited_frames = self.limited_frames+1 if status in ('limited', 'failed') else 0
        self.q, self.nodes = q.copy(), nodes.copy()
        self.q_velocity, self.velocity = qv, v
        current_branch = int(np.sign(bend(nodes)))
        if abs(bend(nodes)) > 3:
            self.branch = current_branch
        return self.last | dict(actual=nodes.tolist(), applied=q.tolist(), status=status,
                                reasons=sorted(set(reasons)), wristError=wrist_error, elbowError=elbow_error,
                                palmErrorDeg=palm_error, metrics=metrics, limitedFrames=self.limited_frames,
                                actualBranch=current_branch)
