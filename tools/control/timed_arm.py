"""Timed endpoint execution + normalized route tracking on the existing safety layer.

The endpoint connector is a restricted lower-level fallback, not an upper gesture
generator or a universal human joint synergy. Explicit routes bypass it entirely.
"""
import numpy as np
from tools.control.stateful_arm import ArmController, DT, wrap


POSITION_TOL = .005
ANGLE_TOL = 4.
STOP_SPEED = .02
STOP_ANGULAR_SPEED = 8.


def heading(nodes):
    v = np.asarray(nodes)[3]-np.asarray(nodes)[2]
    return float(np.degrees(np.arctan2(v[1], v[0])))


def task_error(nodes, target):
    parts = []
    if target.elbow is not None:
        parts.extend((target.elbow-nodes[1])/POSITION_TOL)
    parts.extend((target.wrist-nodes[2])/POSITION_TOL)
    if target.palm_heading is not None:
        parts.append(float(wrap(target.palm_heading-heading(nodes)))/ANGLE_TOL)
    return np.array(parts)


def task_jacobian(nodes, jac, target):
    rows = []
    if target.elbow is not None:
        rows.extend(jac[2:4]/POSITION_TOL)
    rows.extend(jac[4:6]/POSITION_TOL)
    if target.palm_heading is not None:
        v = nodes[3]-nodes[2]; dv = jac[6:8]-jac[4:6]
        rows.append((v[0]*dv[1]-v[1]*dv[0])/max(float(v@v), 1e-12)*180/np.pi/ANGLE_TOL)
    return np.array(rows)


def response_jacobian(forward, q, lo, hi):
    base = forward(q); columns = []
    for j in range(len(q)):
        step = .05 if q[j]+.05 <= hi[j] else -.05
        columns.append((forward(q+np.eye(len(q))[j]*step)-base).ravel()/step)
    return base, np.stack(columns, axis=1)


class TimedArmController(ArmController):
    def __init__(self, *args, policy='timed', **kwargs):
        super().__init__(*args, **kwargs)
        if policy not in ('timed', 'normalized'):
            raise ValueError('timed policy')
        self.policy = policy
        self.plan = None
        self.q_acceleration = np.zeros(self.dof)
        self.pass_completed = False
        self._active_signature = None
        self.actuator_acceleration_reserve = .02

    @staticmethod
    def _signature(target):
        # A body-following pose keeps one actuator endpoint/clock while its
        # material-node coordinates follow the changing body context. Only the
        # lower realization layer supplies this key; scene-space routes do not.
        if target.connection_key is not None:
            return ('body-following-hold',target.connection_key,round(float(target.time),8))
        vector = lambda x: None if x is None else tuple(np.round(x, 8))
        # Ignore clock reconstruction roundoff, not actual goal changes. Otherwise
        # repeated identical packets would restart the connection every frame.
        return (target.mode, round(float(target.time), 8), vector(target.wrist), vector(target.elbow),
                None if target.palm_heading is None else round(target.palm_heading, 7),
                target.branch, vector(target.velocity), round(target.angular_velocity, 7),vector(target.projection))

    def _fit_endpoint(self, target, predict, q):
        # Two local seeds only, no parameter inventory/grid scan. The elbow seed
        # exploits the existing proximal actuator, then all task dimensions are
        # refined jointly. This seed is not sent to the executor.
        seeds = [q.copy()]
        if target.elbow is not None:
            seed = q.copy()
            for _ in range(16):
                base, j = response_jacobian(predict, seed, self.lo, self.hi)
                column = j[2:4, 0]; error = target.elbow-base[1]
                step = float(column@error)/(float(column@column)+1e-10)
                seed[0] = np.clip(seed[0]+np.clip(step, -8, 8), self.lo[0], self.hi[0])
                if np.linalg.norm(error) < 1e-6:
                    break
            seeds.insert(0, seed)
        # A bounded neutral initialization can escape a saturated local basin.
        # It is a solver candidate, never a direct reset of actual actuators.
        # Existing successful local fits exit before evaluating this fallback.
        seeds.append(np.clip(np.zeros(self.dof), self.lo, self.hi))
        best = None
        for seed in seeds:
            z = seed.copy()
            for _ in range(32):
                nodes, raw = response_jacobian(predict, z, self.lo, self.hi)
                error = task_error(nodes, target); jac = task_jacobian(nodes, raw, target)
                cost = float(error@error)
                if best is None or cost < best[0]:
                    best = (cost, z.copy(), nodes.copy())
                if np.max(abs(error)) < .001:
                    break
                step = np.linalg.solve(jac.T@jac+1e-5*np.eye(self.dof), jac.T@error)
                step *= min(1., 10/max(float(np.max(abs(step))), 1e-12))
                accepted = False
                for scale in (1., .5, .25, .1):
                    proposal = np.clip(z+step*scale, self.lo, self.hi)
                    e = task_error(predict(proposal), target)
                    if float(e@e) < cost-1e-10:
                        z = proposal; accepted = True; break
                if not accepted:
                    break
            if best[0] < 1e-6:
                break
        return best[1], best[2], float(np.sqrt(best[0]))

    def _terminal_actuator_velocity(self, target, goal, predict):
        if target.mode != 'pass':
            return np.zeros(self.dof), 0., 0.
        nodes, raw = response_jacobian(predict, goal, self.lo, self.hi)
        a = list(raw[4:6]/STOP_SPEED); b = list(target.velocity/STOP_SPEED)
        if target.palm_heading is not None:
            v = nodes[3]-nodes[2]; dv = raw[6:8]-raw[4:6]
            rate = (v[0]*dv[1]-v[1]*dv[0])/max(float(v@v), 1e-12)*180/np.pi
            a.append(rate/STOP_ANGULAR_SPEED); b.append(target.angular_velocity/STOP_ANGULAR_SPEED)
        a, b = np.array(a), np.array(b)
        velocity = np.linalg.solve(a.T@a+1e-6*np.eye(self.dof), a.T@b)
        node_rates = (raw@velocity).reshape(4, 2)
        scale = min(1., self.speed/max(float(np.max(abs(velocity))), 1e-12),
                    self.node_speed/max(float(np.max(np.linalg.norm(node_rates, axis=1))), 1e-12))
        velocity *= scale
        angular_error = 0. if target.palm_heading is None else float(abs(rate@velocity-target.angular_velocity))
        return velocity, float(np.linalg.norm(raw[4:6]@velocity-target.velocity)), angular_error

    def _new_plan(self, target, now, q, predict):
        goal, goal_nodes, residual = self._fit_endpoint(target, predict, q)
        final_velocity, velocity_residual, angular_residual = self._terminal_actuator_velocity(target, goal, predict)
        start = now-DT
        duration = max(target.time-start, DT)
        v0 = self.q_velocity.copy(); a0 = self.q_acceleration.copy()
        rhs = np.array([goal-q-v0*duration-.5*a0*duration**2,
                        final_velocity*duration-v0*duration-a0*duration**2,
                        -a0*duration**2])
        coefficients = np.linalg.solve(np.array([[1., 1., 1.], [3., 4., 5.], [6., 12., 20.]]), rhs)
        self.plan = dict(signature=self._signature(target), start=start, duration=duration,
            deadline=target.time, initial=q.copy(), velocity=v0, acceleration=a0,
            goal=goal, goalNodes=goal_nodes, finalVelocity=final_velocity, coefficients=coefficients,
            endpointResidual=residual, terminalVelocityResidual=velocity_residual, terminalAngularVelocityResidual=angular_residual)
        self.pass_completed = False

    def _normalized_objective(self, target, actual, jac):
        error = task_error(actual, target); j = task_jacobian(actual, jac, target)
        step = np.zeros_like(error)
        if self.last is not None:
            old = np.array(self.last['requested']); start = 0
            if target.elbow is not None:
                step[:2] = (target.elbow-old[1])/POSITION_TOL; start = 2
            step[start:start+2] = ((target.wrist-old[2]) if target.velocity is None else target.velocity*DT)/POSITION_TOL
            if target.palm_heading is not None:
                step[-1] = (target.angular_velocity*DT if target.velocity is not None or target.angular_velocity != 0 else float(wrap(target.palm_heading-heading(old))))/ANGLE_TOL
        return j.T@j+1e-5*np.eye(self.dof), j.T@(self.feedback_gain*error+(1-self.feedback_gain)*step)

    def _objective(self, target, now, q, actual, predict, jac, desired):
        signature = self._signature(target)
        if signature != self._active_signature:
            self.pass_completed = False
            self._active_signature = signature
        if self.policy == 'normalized' or target.mode == 'track':
            self._policy_info = dict(kind='normalized-task-tracking', mode=target.mode, reasons=[],
                                     referenceNodes=desired.tolist(), deadlineControlsPlan=False)
            self.plan = None
            return self._normalized_objective(target, actual, jac)
        if self.plan is None or self.plan['signature'] != self._signature(target):
            self._new_plan(target, now, q, predict)
        p = self.plan
        t = (now-p['start'])/p['duration']
        if t >= 1:
            reference = p['goal']+p['finalVelocity']*max(0., now-target.time)
            reference_velocity = p['finalVelocity']
        else:
            c3, c4, c5 = p['coefficients']
            reference = (p['initial']+p['velocity']*p['duration']*t+.5*p['acceleration']*p['duration']**2*t*t+
                         c3*t**3+c4*t**4+c5*t**5)
            reference_velocity = (p['velocity']*p['duration']+p['acceleration']*p['duration']**2*t+
                                  3*c3*t*t+4*c4*t**3+5*c5*t**4)/p['duration']
        reasons = []
        if p['endpointResidual'] > 1:
            reasons.append('endpoint-solution-limited')
        if p['terminalVelocityResidual'] > STOP_SPEED or p['terminalAngularVelocityResidual'] > STOP_ANGULAR_SPEED:
            reasons.append('terminal-velocity-limited')
        if np.any(reference < self.lo) or np.any(reference > self.hi):
            reasons.append('reference-actuator-bound')
        bounded = np.clip(reference, self.lo, self.hi)
        # Relative endpoint displacement sets a pose metric, not a force model.
        # Axes not needed by this endpoint are discouraged from compensating for
        # other axes, but remain free to satisfy actual safety constraints.
        weights = 1/np.maximum(abs(p['goal']-p['initial']), 1.)**2
        h = np.diag(weights)+1e-8*np.eye(self.dof)
        self._policy_info = dict(kind='timed-endpoint-connector', mode=target.mode,
            start=p['start'], deadline=target.time, remaining=target.time-now,
            endpointActuators=p['goal'].tolist(), referenceActuators=bounded.tolist(),
            referenceActuatorVelocity=reference_velocity.tolist(),
            referenceNodes=predict(bounded).tolist(), endpointResidual=p['endpointResidual'],
            terminalVelocityResidual=p['terminalVelocityResidual'],
            terminalAngularVelocityResidual=p['terminalAngularVelocityResidual'],
            terminalActuatorVelocity=p['finalVelocity'].tolist(), reasons=reasons, deadlineControlsPlan=True)
        # Position AND velocity tracking. A clipped reference may run ahead of
        # the executor; chasing only its position would discard planned braking.
        return h, h@(.5*(bounded-q)+.5*reference_velocity*DT)

    def observe(self, applied_q, nodes, geometry=None):
        previous_limited = self.limited_frames
        old_velocity = self.q_velocity.copy()
        old_nodes = self.nodes.copy()
        row = super().observe(applied_q, nodes, geometry)
        self.q_acceleration = (self.q_velocity-old_velocity)/DT
        rate = float(wrap(heading(nodes)-heading(old_nodes)))/DT
        target = self.pending
        positional = row['wristError'] <= POSITION_TOL and (row['elbowError'] is None or row['elbowError'] <= POSITION_TOL) and (row['palmErrorDeg'] is None or row['palmErrorDeg'] <= ANGLE_TOL)
        wrist_velocity_error = float(np.linalg.norm(self.velocity[2]-(np.zeros(2) if target.velocity is None else target.velocity)))
        if target.mode == 'hold':
            terminal = row['metrics']['maxNodeSpeed'] <= STOP_SPEED and (target.palm_heading is None or abs(rate) <= STOP_ANGULAR_SPEED)
        elif target.mode == 'pass':
            terminal = wrist_velocity_error <= STOP_SPEED and (target.palm_heading is None or abs(rate-target.angular_velocity) <= STOP_ANGULAR_SPEED)
        else:
            terminal = target.velocity is None or wrist_velocity_error <= STOP_SPEED
        reasons = set(row['reasons'])
        if target.mode == 'hold' and positional and not terminal:
            reasons.add('terminal-not-settled')
        if target.mode == 'pass' and positional and terminal and row['now'] >= target.time-DT-1e-9:
            self.pass_completed = True
        reached = positional and terminal
        if row['now'] >= target.time and not reached and not self.pass_completed:
            reasons.add('deadline-missed')
        if self.pass_completed:
            reasons.discard('deadline-missed'); reasons.discard('local-stall-or-constraint-conflict')
        if not row['qpFeasible'] or any(r.startswith('actual-') for r in reasons):
            status = 'failed'
        elif reasons:
            status = 'limited'
        else:
            status = 'passed' if self.pass_completed else 'reached' if reached else 'tracking'
        self.limited_frames = previous_limited+1 if status in ('limited', 'failed') else 0
        row.update(status=status, reasons=sorted(reasons), positionReached=positional, terminalStateMet=terminal,
            goalCompleted=self.pass_completed if target.mode == 'pass' else reached,
            wristVelocity=self.velocity[2].tolist(), wristVelocityError=wrist_velocity_error,
            palmAngularVelocity=rate, limitedFrames=self.limited_frames)
        return row
