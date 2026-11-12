import unittest
import numpy as np
from tools.control.stateful_arm import ArmController, Target, constrained_step, patch_forward, DT, speed_limits


class StatefulArmTests(unittest.TestCase):
    def test_faster_budgets_reduce_arrival_time_without_disabling_feedback_limits(self):
        def f(q):
            return np.array([[0, 0], [q[0]*.01, .1], [q[0]*.01, .2], [q[0]*.01, .23]])
        arrivals = []
        for profile in ('conservative', 'responsive', 'fast'):
            limits = speed_limits(profile)
            c = ArmController('native', f, [-30]*3, [30]*3, np.zeros(3), f(np.zeros(3)), **limits)
            first = None
            for i in range(90):
                q, _ = c.command(Target([.18, .2], .6), i*DT)
                row = c.observe(q, f(q))
                self.assertNotEqual(row['status'], 'failed')
                self.assertLessEqual(row['metrics']['maxNodeSpeed'], limits['node_speed']+.001)
                self.assertLessEqual(row['metrics']['maxParameterAcceleration'], limits['acceleration']+.001)
                if first is None and row['wristError'] <= .005:
                    first = i
            self.assertIsNotNone(first)
            arrivals.append(first)
        self.assertLess(arrivals[1], arrivals[0])
        self.assertLess(arrivals[2], arrivals[1])

    def test_deadline_is_checked_against_executed_pose(self):
        home = np.array([[0, 0], [0, -.1], [0, -.3], [0, -.33]])
        f = lambda q: patch_forward(home, q)
        c = ArmController('patch', f, [-56, 0, -100], [56, 145, 30], np.zeros(3), home)
        q, _ = c.command(Target([1, 1], 0), 0)
        row = c.observe(q, f(q))
        self.assertIn('deadline-missed', row['reasons'])
        self.assertIn('outside-local-fixed-length-reach-envelope', row['reasons'])

    def test_mesh_floor_constrains_execution_and_reports_actual_failure(self):
        def f(q):
            return np.array([[0, 0], [q[0]*.01, .1], [q[0]*.01, .2], [q[0]*.01, .23]])
        c = ArmController('patch', f, [-10]*3, [10]*3, np.zeros(3), f(np.zeros(3)))
        for i in range(60):
            c.geometry = dict(minTriangleAreaRatio=.6-.1*c.q[0], areaGradient=[-.1, 0, 0], flippedTriangles=0)
            q, _ = c.command(Target([.08, .2], 1), i*DT)
            row = c.observe(q, f(q), dict(minTriangleAreaRatio=.6-.1*q[0], areaGradient=[-.1, 0, 0], flippedTriangles=0))
            self.assertGreaterEqual(c.geometry['minTriangleAreaRatio'], .54999)
        self.assertIn('mesh-area-floor', row['reasons'])
        q, _ = c.command(Target([.08, .2], 1), 2)
        row = c.observe(q, f(q), dict(minTriangleAreaRatio=.4, areaGradient=[-.1, 0, 0], flippedTriangles=0))
        self.assertEqual(row['status'], 'failed')
        self.assertIn('actual-mesh-geometry-violation', row['reasons'])

    def test_inactive_mesh_limit_is_kept_through_minimum_switch_and_return(self):
        def f(q):
            return np.array([[0, 0], [q[0]*.01, .1], [q[0]*.01, .2], [q[0]*.01, .23]])
        def geometry(q):
            areas = [1-.02*q[0], 1.2-.1*q[0]]
            return dict(minTriangleAreaRatio=min(areas), areaGradient=[-.02 if areas[0] < areas[1] else -.1, 0, 0],
                        meshAreaRatios=areas, meshAreaGradients=[-.02, 0, 0, -.1, 0, 0], flippedTriangles=0)
        c = ArmController('patch', f, [-10]*3, [10]*3, np.zeros(3), f(np.zeros(3)))
        c.geometry = geometry(c.q)
        rows = []
        for i in range(160):
            q, _ = c.command(Target([.08 if i < 80 else 0, .2], 2), i*DT)
            rows.append(c.observe(q, f(q), geometry(q)))
            self.assertGreaterEqual(c.geometry['minTriangleAreaRatio'], .54999)
        self.assertFalse(any(r['status'] == 'failed' for r in rows))
        self.assertLess(rows[-1]['wristError'], .001)

    def test_piecewise_response_crossing_keeps_a_feasible_stop(self):
        # Regression for a gain change ahead: current-step limits alone allow
        # momentum into a point where the next step has no feasible acceleration.
        def f(q):
            x = q[0]*(.012 if q[0] >= 0 else .003)
            return np.array([[0, 0], [x, .1], [x, .2+q[1]*.002], [x+q[2]*.001, .23]])
        initial = np.array([3., 0, 0])
        c = ArmController('native', f, [-10]*3, [10]*3, initial, f(initial))
        rows = []
        for i in range(120):
            q, _ = c.command(Target([-.012, .2], 2, elbow=[-.012, .1]), i*DT)
            rows.append(c.observe(q, f(q)))
        self.assertFalse(any(r['status'] == 'failed' for r in rows))
        self.assertLess(rows[-1]['wristError'], .001)
        self.assertLess(max(r['metrics']['maxNodeAcceleration'] for r in rows), 1.801)

    def test_nearly_parallel_constraints_are_not_called_infeasible(self):
        a = np.array([[1., 0., 0.], [1., 1e-6, 0.], [-1., 0., 0.], [0., 1., 0.], [0., -1., 0.]])
        x, feasible = constrained_step(np.diag([1e-4, 1e-5, 1.]), np.array([2., 1., 0.]), a, np.array([.2, .2, .2, .1, .1]))
        self.assertTrue(feasible)
        self.assertTrue(np.all(a @ x <= np.array([.2, .2, .2, .1, .1])+1e-7))

    def test_inconsistent_constraints_are_not_success(self):
        _, feasible = constrained_step(np.eye(1), np.ones(1), np.array([[1.], [-1.]]), np.array([-1., -1.]))
        self.assertFalse(feasible)

    def test_switch_unreachable_return_without_frame_jump(self):
        def forward(q):
            return np.array([[0, 0], [q[0]*.01, .1], [q[1]*.01, .2], [q[2]*.01, .23]])
        c = ArmController('native', forward, [-10]*3, [10]*3, np.zeros(3), forward(np.zeros(3)))
        rows = []
        for i in range(240):
            goal = [.06, .2] if i < 45 else [-.06, .2] if i < 90 else [2, .2] if i < 150 else [0, .2]
            q, _ = c.command(Target(goal, (i//60+1)*2), i*DT)
            rows.append(c.observe(q, forward(q)))
        self.assertLess(rows[-1]['wristError'], .005)
        self.assertGreater(rows[149]['wristError'], 1)
        self.assertTrue(any('actuator-bound' in r['reasons'] for r in rows))
        self.assertLessEqual(max(r['metrics']['maxParameterStep'] for r in rows), 1.50001)
        self.assertLessEqual(max(r['metrics']['maxNodeSpeed'] for r in rows), .2401)
        self.assertLessEqual(max(r['metrics']['maxParameterAcceleration'] for r in rows), 180.01)

    def test_actual_disturbance_is_reported_and_state_retained(self):
        home = np.array([[0, 0], [0, -.1], [0, -.3], [0, -.33]])
        f = lambda q: patch_forward(home, q)
        c = ArmController('patch', f, [-56, 0, -100], [56, 145, 30], np.zeros(3), home)
        q, _ = c.command(Target(home[2], 1), 0)
        observed = f(q)+[.05, 0]
        row = c.observe(q, observed)
        self.assertEqual(row['status'], 'failed')
        self.assertIn('actual-maxNodeSpeed-violation', row['reasons'])
        np.testing.assert_allclose(c.nodes, observed)

    def test_coordinate_identity_and_conflict(self):
        with self.assertRaises(ValueError):
            Target([0, 0], 1, frame='world-xyz')
        with self.assertRaises(ValueError):
            Target([np.nan, 0], 1)
        home = np.array([[0, 0], [0, -.1], [0, -.3], [0, -.33]])
        c = ArmController('patch', lambda q: patch_forward(home, q), [-56, 0, -100], [56, 145, 30], np.zeros(3), home)
        q, row = c.command(Target(home[2], 0, shoulder=[1, 1], branch=-1), 1)
        self.assertIn('shoulder-translation-unsupported', row['reasons'])
        self.assertIn('requested-branch-conflict', row['reasons'])
        self.assertIsNotNone(c.pending)


if __name__ == '__main__':
    unittest.main()
