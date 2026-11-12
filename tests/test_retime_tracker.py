import unittest
import numpy as np
from tools.control.retime_tracker import time_map

class RetimeTrackerTests(unittest.TestCase):
    def test_monotonic_endpoints_and_original_prefix(self):
        for mode in ('slow-return','ease-stop'):
            t=np.linspace(0,3,901);s=time_map(t,1.75,mode)
            self.assertTrue(np.all(np.diff(s)>=-1e-10))
            np.testing.assert_allclose(s[t<=.5],t[t<=.5])
            self.assertAlmostEqual(s[-1],1.75)
    def test_ease_derivative(self):
        h=1e-5
        self.assertAlmostEqual((time_map(.5+h,1.75,'ease-stop')-.5)/h,1,places=4)
        self.assertAlmostEqual((1.75-time_map(3-h,1.75,'ease-stop'))/h,0,places=4)
