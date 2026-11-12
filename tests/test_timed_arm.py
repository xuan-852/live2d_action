import unittest
import numpy as np
from tools.control.stateful_arm import Target, patch_forward, speed_limits, DT
from tools.control.timed_arm import TimedArmController, task_error


def linear(q):
    wrist = np.array([q[1]*.01, .2])
    r = np.radians(q[2])
    return np.array([[0, 0], [q[0]*.01, .1], wrist, wrist+.03*np.array([np.cos(r), np.sin(r)])])


def controller(q=None):
    q = np.zeros(3) if q is None else np.array(q, float)
    return TimedArmController('native', linear, [-30, -30, -180], [30, 30, 180], q, linear(q), **speed_limits('fast'))


class TimedArmTests(unittest.TestCase):
    def test_float32_readback_stays_inside_declared_actuator_acceleration(self):
        c=TimedArmController('native',linear,[-150,-150,-150],[150,150,150],
            np.array([120.,110.,100.]),linear([120,110,100]),
            speed=135,acceleration=1620,node_speed=100,node_acceleration=100000)
        for i in range(100):
            sign=1 if i<40 else -1
            target=Target([sign*1.4,.2],(i+1)*DT,elbow=[sign*1.4,.1],palm_heading=sign*140,mode='track')
            q,_=c.command(target,(i+1)*DT)
            actual=q.astype(np.float32).astype(float)
            row=c.observe(actual,linear(actual))
            self.assertLessEqual(row['metrics']['maxParameterAcceleration'],1620.001)

    def test_neutral_solver_seed_escapes_saturated_local_basin_without_reset(self):
        def nonlinear(q):
            p=linear(q).copy()
            p[2:,0]=.2 if q[1]<-20 else q[1]*.01
            return p
        q=np.array([0.,-25.,0.])
        c=TimedArmController('native',nonlinear,[-30,-30,-30],[30,30,30],q,nonlinear(q),**speed_limits('fast'))
        target=Target(nonlinear(np.zeros(3))[2],.6,elbow=nonlinear(np.zeros(3))[1],palm_heading=0)
        goal,nodes,residual=c._fit_endpoint(target,nonlinear,q)
        self.assertLess(residual,.01)
        command,_=c.command(target,DT)
        self.assertLess(np.max(abs(command-q)),.1)
        self.assertGreater(np.max(abs(command-goal)),1.)

    def test_default_point_target_uses_deadline_and_hold_semantics(self):
        commands = []
        for deadline in (.3, 1.2):
            c = controller()
            q, info = c.command(Target([.1, .2], deadline), DT)
            self.assertEqual(info['mode'], 'hold')
            self.assertTrue(info['policy']['deadlineControlsPlan'])
            commands.append(q)
        self.assertGreater(np.max(abs(commands[0]-commands[1])), 1e-3)

    def test_deadline_changes_executed_motion_without_moving_deadline(self):
        commands = []; arrivals = []
        for deadline in (.4, 1.2):
            c = controller(); path = []; arrived = None
            for i in range(50):
                now = (i+1)*DT
                q, info = c.command(Target([.1, .2], deadline, elbow=[.05, .1], palm_heading=0, mode='hold'), now)
                row = c.observe(q, linear(q)); path.append(q.copy())
                self.assertEqual(info['policy']['deadline'], deadline)
                self.assertNotEqual(row['status'], 'failed')
                if row['goalCompleted'] and arrived is None:
                    arrived = now
            commands.append(np.array(path)); arrivals.append(arrived)
        self.assertGreater(np.max(abs(commands[0]-commands[1])), 1)
        self.assertLess(arrivals[0], arrivals[1])

    def test_hold_position_hit_is_not_settled_while_moving(self):
        c = controller([0, -.25, 0])
        c.command(Target([0, .2], 1, palm_heading=0, mode='hold'), DT)
        row = c.observe(np.zeros(3), linear(np.zeros(3)))
        self.assertTrue(row['positionReached'])
        self.assertFalse(row['terminalStateMet'])
        self.assertFalse(row['goalCompleted'])
        self.assertIn('terminal-not-settled', row['reasons'])
        c.command(Target([0, .2], 1, palm_heading=0, mode='hold'), 2*DT)
        row = c.observe([0, .25, 0], linear([0, .25, 0]))
        self.assertEqual(row['limitedFrames'], 2)

    def test_repeated_packet_roundoff_does_not_restart_connection(self):
        c = controller()
        q, _ = c.command(Target([.1, .2], 1.+2e-16, mode='hold'), DT)
        start = c.plan['start']; initial = c.plan['initial'].copy()
        c.observe(q, linear(q))
        c.command(Target([.1, .2], 1., mode='hold'), 2*DT)
        self.assertEqual(c.plan['start'], start)
        np.testing.assert_array_equal(c.plan['initial'], initial)

    def test_unachievable_pass_velocity_is_reported(self):
        c = controller()
        _, info = c.command(Target([.1, .2], .8, mode='pass', velocity=[2, 0]), DT)
        self.assertIn('terminal-velocity-limited', info['reasons'])
        self.assertGreater(info['policy']['terminalVelocityResidual'], 1)

    def test_pass_reaches_with_velocity_and_continues_through_point(self):
        c = controller(); rows = []
        for i in range(30):
            q, _ = c.command(Target([.1, .2], .8, palm_heading=0, mode='pass', velocity=[.08, 0]), (i+1)*DT)
            rows.append(c.observe(q, linear(q)))
        self.assertTrue(any(r['goalCompleted'] for r in rows))
        self.assertGreater(rows[-1]['actual'][2][0], .11)
        self.assertGreater(rows[-1]['wristVelocity'][0], .06)
        self.assertEqual(rows[-1]['status'], 'passed')
        self.assertFalse(any(r['status'] == 'failed' for r in rows))

    def test_explicit_route_bypasses_endpoint_connector_and_does_not_stop(self):
        c = controller(); rows = []
        for i in range(30):
            now = (i+1)*DT; goal = np.array([.05*now, .2])
            q, info = c.command(Target(goal, now, mode='track', velocity=[.05, 0]), now)
            rows.append(c.observe(q, linear(q)))
            np.testing.assert_allclose(info['requested'][2], goal)
            self.assertIsNone(c.plan)
            self.assertFalse(info['policy']['deadlineControlsPlan'])
        self.assertLess(rows[-1]['wristError'], .005)
        self.assertGreater(rows[-1]['wristVelocity'][0], .04)

    def test_target_switch_uses_actual_velocity_and_keeps_frame_limits(self):
        c = controller(); prior = None
        for i in range(45):
            target = Target([.15 if i < 9 else -.05, .2], .6 if i < 9 else 1.2, mode='hold')
            old_velocity = c.q_velocity.copy()
            q, _ = c.command(target, (i+1)*DT)
            if i == 9:
                np.testing.assert_allclose(c.plan['velocity'], old_velocity)
                self.assertGreater(np.linalg.norm(old_velocity), 0)
            row = c.observe(q, linear(q))
            self.assertNotEqual(row['status'], 'failed')
            if prior is not None:
                self.assertLessEqual(np.max(abs(q-prior)), 4.50001)
            prior = q.copy()

    def test_unreachable_reports_residual_then_recovers(self):
        c = controller(); rows = []
        for i in range(90):
            target = Target([2 if i < 30 else 0, .2], .2 if i < 30 else 2, mode='hold')
            q, _ = c.command(target, (i+1)*DT); rows.append(c.observe(q, linear(q)))
        self.assertIn('endpoint-solution-limited', rows[20]['reasons'])
        self.assertGreater(rows[20]['wristError'], 1)
        self.assertTrue(rows[-1]['goalCompleted'])
        self.assertFalse(any(r['status'] == 'failed' for r in rows))

    def test_endpoint_connection_does_not_invent_large_patch_elbow_wrist_bends(self):
        home = np.array([[0., 0.], [-.02, -.12], [-.1, -.3], [-.11, -.33]])
        f = lambda q: patch_forward(home, q)
        goal = f([-35, 0, 0]); v = goal[3]-goal[2]
        c = TimedArmController('patch', f, [-56, 0, -100], [56, 145, 30], np.zeros(3), home, **speed_limits('fast'))
        applied = []
        for i in range(45):
            q, _ = c.command(Target(goal[2], .6, elbow=goal[1], palm_heading=np.degrees(np.arctan2(v[1], v[0])), mode='hold'), (i+1)*DT)
            row = c.observe(q, f(q)); applied.append(q)
            self.assertNotEqual(row['status'], 'failed')
        self.assertLess(np.max(abs(np.array(applied)[:, 1:])), 1)
        self.assertTrue(row['goalCompleted'])

    def test_angular_task_error_wraps_and_conflicting_modes_are_rejected(self):
        p = linear([0, 0, -179])
        e = task_error(p, Target(p[2], 1, palm_heading=179, mode='hold'))
        self.assertAlmostEqual(e[-1], -.5)
        with self.assertRaises(ValueError):
            Target([0, 0], 1, mode='pass')
        with self.assertRaises(ValueError):
            Target([0, 0], 1, mode='hold', velocity=[.1, 0])


if __name__ == '__main__':
    unittest.main()
