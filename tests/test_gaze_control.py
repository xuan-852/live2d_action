import unittest
import numpy as np
from tools.control.gaze_control import GazeController


class GazeControlTests(unittest.TestCase):
    def test_applied_state_reversal_and_float32_limits(self):
        c=GazeController([0,0])
        for i in range(100):
            target=[.7,-.7] if i<9 else [-.6,.5] if i<21 else [0,0]
            prior=c.q.copy()
            command=c.command(target,2.,(i+1)/30)
            self.assertTrue(np.array_equal(c.q,prior))
            row=c.observe(command.astype(np.float32))
            self.assertNotEqual(row['status'],'failed',row)
            self.assertLessEqual(row['metrics']['maxAcceleration'],15.001)
        self.assertTrue(row['settled'])

    def test_range_residual_and_recovery(self):
        c=GazeController([0,0])
        for i in range(60):row=c.observe(c.command([2,0],.5,(i+1)/30).astype(np.float32))
        self.assertEqual(row['requested'],[2,0]);self.assertIn('gaze-range-limited',row['reasons'])
        self.assertFalse(row['settled'])
        for i in range(60):row=c.observe(c.command([0,0],3.,2+(i+1)/30).astype(np.float32))
        self.assertTrue(row['settled'])

    def test_nonfinite_and_wrong_dimensions_rejected(self):
        c=GazeController([0,0])
        for value in [[0],[float('nan'),0]]:
            with self.assertRaises(ValueError):c.command(value,1.,0.)
