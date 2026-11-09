"""Mechanism tests for the expression batch: metrics, mapping and honesty checks."""
import math
import unittest

from tools.expression import adapter, evaluate, roles, source


def toy_joints(wrist_left=(0.0, 0.7, 0.0), wrist_right=(0.0, 0.7, 0.0), head=(0.0, 1.6, 0.0)):
    joints = {name: [0.0, 0.0, 0.0] for name in source.JOINTS}
    joints['pelvis'] = [0.0, 1.0, 0.0]
    joints['rightShoulder'] = [0.2, 1.4, 0.0]
    joints['leftShoulder'] = [-0.2, 1.4, 0.0]
    joints['head'] = list(head)
    joints['rightElbow'] = [0.25, 1.1, 0.0]
    joints['leftElbow'] = [-0.25, 1.1, 0.0]
    joints['rightWrist'] = list(wrist_right)
    joints['leftWrist'] = list(wrist_left)
    return joints


class SourceFeatureTests(unittest.TestCase):
    def test_scale_is_torso_length_and_mirrored_lateral(self):
        feature = source.features(toy_joints(wrist_left=(-0.5, 0.9, 0.0)))
        # head at y=1.6, pelvis at 1.0 -> torso length 0.6
        self.assertAlmostEqual(feature['right'][0], -0.0, places=6)
        self.assertAlmostEqual(feature['left'][0], 0.5 / 0.6, places=6)
        self.assertAlmostEqual(feature['left'][1], (1.4 - 0.9) / 0.6 * -1, places=6)

    def test_hand_further_from_body_has_larger_distance(self):
        close = source.features(toy_joints(wrist_left=(0.0, 1.4, 0.0)))
        far = source.features(toy_joints(wrist_left=(0.4, 0.8, 0.0)))
        self.assertLess(close['leftDistance'], far['leftDistance'])


class ArmSolverTests(unittest.TestCase):
    def calibration(self):
        return {'neutral': {'rightHand': [0.2, 0.0], 'chest': [0.0, 0.05], 'head': [0.0, 0.32],
                            'hip': [0.0, -0.5]},
                'grid': [{'shoulder': 0, 'swing': 0, 'hand': [0.2, 0.0]},
                         {'shoulder': 30, 'swing': 0, 'hand': [0.4, 0.3]},
                         {'shoulder': 60, 'swing': 0, 'hand': [0.35, 0.65]}]}

    def test_solver_picks_nearest_measured_pose(self):
        shoulder, swing, residual = adapter.solve_arm([0.39, 0.29], self.calibration(), 1.0)
        self.assertEqual((shoulder, swing), (30, 0))
        self.assertLess(residual, 0.02)

    def test_unreachable_target_reports_residual_instead_of_pretending(self):
        _, _, residual = adapter.solve_arm([-2.0, -2.0], self.calibration(), 1.0)
        self.assertGreater(residual, 1.0)

    def test_curves_stay_inside_parameter_limits(self):
        samples = [{'left': (2.0, 3.0), 'yaw': 1.2, 'headLateral': 0.9, 'wristSlope': 2.0},
                   {'left': (-2.0, -3.0), 'yaw': -1.2, 'headLateral': -0.9, 'wristSlope': -2.0}]
        curves, info = adapter.candidate_curves(samples, self.calibration(), 30, (0.0, 0.0), 'left')
        self.assertLessEqual(max(curves[adapter.SHOULDER]), 60)
        self.assertGreaterEqual(min(curves[adapter.SHOULDER]), -30)
        self.assertLessEqual(max(curves[adapter.BODY_TURN]), 10)
        self.assertGreaterEqual(min(curves[adapter.BODY_TURN]), -10)
        self.assertGreater(info['maxResidual'], 0)


class EvaluationTests(unittest.TestCase):
    def frames(self, lateral, height=None):
        rendered = []
        for index, value in enumerate(lateral):
            rendered.append({'right': (0.2 + value, 0.0 if height is None else height[index])})
        return rendered

    def source(self, lateral, height=None):
        return [{'left': (0.2 + value, 0.0 if height is None else height[index])}
                for index, value in enumerate(lateral)]

    def test_matching_motion_passes(self):
        values = [0.0, 0.05, 0.1, 0.15, 0.2]
        result = evaluate.compare(self.source(values), self.frames(values), 'left')
        self.assertEqual(result['status'], 'pass')
        self.assertAlmostEqual(result['peakRatio'], 1.0, places=4)

    def test_mirrored_motion_is_rejected(self):
        values = [0.0, 0.05, 0.1, 0.15, 0.2]
        result = evaluate.compare(self.source(values), self.frames([-v for v in values]), 'left')
        self.assertEqual(result['status'], 'fail')
        self.assertLess(result['lateralCorrelation'], 0)

    def test_amplitude_collapse_is_rejected(self):
        values = [0.0, 0.2, 0.4, 0.6]
        small = [v * 0.1 for v in values]
        result = evaluate.compare(self.source(values), self.frames(small), 'left')
        self.assertEqual(result['status'], 'fail')
        self.assertIn('lateral opening ratio outside accepted band', result['reasons'])

    def test_tiny_collapse_is_no_motion_not_pass(self):
        values = [0.0, 0.1, 0.2, 0.3]
        result = evaluate.compare(self.source(values), self.frames([v * 0.1 for v in values]), 'left')
        self.assertEqual(result['status'], 'no-motion')

    def test_static_hand_is_no_motion_not_pass(self):
        values = [0.0, 0.0, 0.0, 0.0]
        result = evaluate.compare(self.source(values), self.frames(values), 'left')
        self.assertEqual(result['status'], 'no-motion')

    def test_hidden_geometry_is_not_measured(self):
        frame = {'drawables': []}
        self.assertIsNone(evaluate.fx_features(frame, {'rightHand': [], 'leftHand': [], 'chest': [],
                                                       'head': [], 'hip': []}))
        self.assertIsNone(evaluate.series([frame], {'rightHand': ['a'], 'leftHand': ['b'], 'chest': ['c'],
                                                    'head': ['d'], 'hip': ['e']}))


class HonestyTests(unittest.TestCase):
    def test_physics_override_of_owned_channel_is_reported(self):
        frames = [{'frame': 0, 'requested': [{'id': 'Param94', 'value': 10.0}],
                   'applied': [{'id': 'Param94', 'value': 4.0}]}]
        report = evaluate.write_check(frames, {'Param94'})
        self.assertEqual(report['count'], 1)

    def test_channel_handed_to_physics_is_not_faked_as_written(self):
        frames = [{'frame': 0, 'requested': [{'id': 'ParamBodyAngleX2', 'value': 7.0}],
                   'applied': [{'id': 'ParamBodyAngleX2', 'value': 6.9}]}]
        self.assertEqual(evaluate.write_check(frames, {'Param94'})['count'], 0)

    def test_pairing_delta_is_checked(self):
        self.assertFalse(evaluate.pairing([{'pairingMaxDelta': 0.001}])['ok'])
        self.assertTrue(evaluate.pairing([{'pairingMaxDelta': 0.0}])['ok'])
        self.assertFalse(evaluate.pairing([])['ok'])


class RoleTests(unittest.TestCase):
    def test_missing_drawables_return_none(self):
        anchor = roles.anchors({'rightHand': ['x'], 'chest': ['y'], 'head': ['z'], 'leftHand': ['w'], 'hip': ['v']},
                               {'x': {'cx': 1.0, 'cy': 2.0}})
        self.assertEqual(anchor['rightHand'], [1.0, 2.0])
        self.assertIsNone(anchor['chest'])


if __name__ == '__main__':
    unittest.main()
