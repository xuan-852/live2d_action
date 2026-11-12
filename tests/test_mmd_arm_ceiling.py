import unittest
import numpy as np
from tools.control.mmd_arm_ceiling import target_chain

class ArmCeilingTests(unittest.TestCase):
    def test_absolute_directions_and_projection(self):
        h=np.array([[1,2],[1,1],[1,0],[1,-1]],float)
        t={'upperHeadingDeg':0,'forearmHeadingDeg':90,'palmHeadingDeg':0,'upperProjectedRatio':.5,'forearmProjectedRatio':1}
        np.testing.assert_allclose(target_chain(h,t),[[1,2],[1.5,2],[1.5,3],[2.5,3]],atol=1e-10)
