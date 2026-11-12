import unittest
import numpy as np
from tools.control.measure_pivots import fit_rotation

class PivotMeasurementTests(unittest.TestCase):
    def test_known_pivot_both_directions(self):
        rng=np.random.default_rng(3);x=rng.normal(size=(30,2));c=np.array([.3,-.5])
        for a in (-.1,.1):
            r=np.array([[np.cos(a),np.sin(a)],[-np.sin(a),np.cos(a)]])
            result=fit_rotation(x,(x-c)@r+c)
            np.testing.assert_allclose(result['pivot'],c,atol=1e-10)
            self.assertLess(result['rigidRMS'],1e-10)
    def test_translation_does_not_invent_pivot(self):
        x=np.array([[0.,0],[1,0],[0,1]])
        self.assertIsNone(fit_rotation(x,x+[1,2]))
