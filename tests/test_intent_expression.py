import unittest
import copy
import numpy as np
from tools.control.intent_expression_trial import plan


class IntentExpressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.stream,cls.phases,cls.markers,cls.fixed=plan()

    def test_hand_ablation_preserves_task_route_gaze_and_clocks(self):
        neutral,phases,markers,fixed=plan('neutral-fingers')
        self.assertEqual(phases,self.phases);self.assertEqual(fixed,self.fixed);self.assertEqual(markers,self.markers)
        for a,b in zip(self.stream['frames'],neutral['frames']):
            x,y=copy.deepcopy(a),copy.deepcopy(b)
            self.assertEqual(y.pop('nativeFingerGoal')['values'],[0]*5)
            x.pop('nativeFingerGoal');x['intent'].pop('variant');y['intent'].pop('variant')
            self.assertEqual(x,y)

    def test_track_velocity_is_same_interval_requested_wrist_difference(self):
        packets=self.stream['frames']
        for previous,current in zip(packets,packets[1:]):
            target=current['targets']['native']
            if target['mode']=='track':
                expected=(np.array(target['wrist'])-previous['targets']['native']['wrist'])*30
                np.testing.assert_allclose(target['velocity'],expected,atol=1e-10)

    def test_three_tasks_have_actual_marker_and_noncontact_contract(self):
        self.assertEqual({p['task'] for p in self.phases},{'idle','point','explain','near-head'})
        self.assertEqual({m['id'] for m in self.markers},{'indexBase','indexTip','headSide','irisLeft','irisRight'})
        self.assertTrue(all('palm_basis' not in p['targets']['patch'] for p in self.stream['frames']))
        self.assertTrue(all(p['targets']['patch']['mode']=='hold' for p in self.stream['frames']))
        self.assertTrue(all(abs(v)<=20 for p in self.stream['frames'] for v in p['nativeFingerGoal']['values']))
        self.assertGreater(np.linalg.norm(self.fixed['nearHeadPalmOffset']),.02)
