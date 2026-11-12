import unittest
import numpy as np
from tools.control.upper_node_trial import chain,inverse_patch

class UpperNodeTrialTests(unittest.TestCase):
    def test_node_adapter_roundtrip(self):
        home=np.array([[0,0],[0,-1],[0,-2],[0,-3]],float);angles=np.array([-35,55,-25])
        target=chain(home,angles)
        np.testing.assert_allclose(inverse_patch(home,target),angles,atol=1e-8)
        np.testing.assert_allclose(np.linalg.norm(np.diff(target,axis=0),axis=1),1)

    def test_relative_wrap_crossing(self):
        home=np.array([[0,0],[0,-1],[0,-2],[0,-3]],float)
        angles=np.array([-8,145,33])
        np.testing.assert_allclose(inverse_patch(home,chain(home,angles)),angles,atol=1e-8)
