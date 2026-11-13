import unittest
import numpy as np
from tools.control.finger_control import FingerController,FINGER_IDS


class FingerControllerTests(unittest.TestCase):
    def test_visible_ids_exclude_anger_and_alternate_hand(self):
        self.assertEqual(FINGER_IDS,('Param102','Param103','Param105','Param106','Param107'))

    def test_pose_and_reversal_keep_frame_budgets_and_eventually_settle(self):
        c=FingerController(np.zeros(5))
        for i in range(120):
            goal=np.array([-15,20,-20,-20,-20]) if i<12 else np.array([10,-10,8,5,3])
            q=c.command(goal,3.,(i+1)/30);r=c.observe(q)
            self.assertLessEqual(r['metrics']['maxSpeed'],90.001)
            self.assertLessEqual(r['metrics']['maxAcceleration'],540.001)
            self.assertNotEqual(r['status'],'failed')
        self.assertTrue(r['settled'])
        np.testing.assert_allclose(c.q,goal,atol=.001)

    def test_command_does_not_substitute_for_applied_state(self):
        c=FingerController(np.zeros(5));q=c.command([10]*5,1.,1/30)
        np.testing.assert_array_equal(c.q,np.zeros(5))
        r=c.observe(q*.5)
        np.testing.assert_allclose(c.q,q*.5)
        self.assertIn('actual-finger-command-residual',r['reasons'])

    def test_unavailable_pose_retains_request_and_returns_to_range(self):
        c=FingerController(np.zeros(5))
        for i in range(60):r=c.observe(c.command([40]*5,.5,(i+1)/30))
        self.assertIn('finger-pose-range-limited',r['reasons']);self.assertFalse(r['settled'])
        self.assertEqual(r['requested'],[40]*5);self.assertEqual(r['feasibleReference'],[20]*5)
        for i in range(60):r=c.observe(c.command([0]*5,3.,2+(i+1)/30))
        self.assertTrue(r['settled'])

    def test_invalid_values_rejected(self):
        c=FingerController(np.zeros(5))
        for value in ([1]*4,[1,2,float('nan'),4,5]):
            with self.assertRaises(ValueError):c.command(value,1.,0.)

    def test_float32_sdk_readback_at_stop_faces_and_midmotion_switches(self):
        c=FingerController(np.zeros(5))
        for i in range(200):
            goal=[-20]*5 if i<28 else [20,-18,-18,-18,-18] if i<42 else [-20,20,-20,20,-20] if i<51 else [0]*5
            q=c.command(goal,5.,(i+1)/30).astype(np.float32)
            row=c.observe(q)
            self.assertNotEqual(row['status'],'failed',row)
            self.assertLessEqual(row['metrics']['maxAcceleration'],540.001)
        self.assertTrue(row['settled'])
