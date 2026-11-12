import unittest
import numpy as np
from tools.control.wrist_revision import area_axis

class WristAxisTests(unittest.TestCase):
    def test_rectangle_axis(self):
        v=np.array([[-1,-3],[1,-3],[1,3],[-1,3]],float)
        centre,axis,ratio=area_axis(v,np.array([[0,1,2],[0,2,3]]))
        np.testing.assert_allclose(centre,0,atol=1e-10)
        np.testing.assert_allclose(axis,[0,-1],atol=1e-10)
        self.assertAlmostEqual(ratio,9)
