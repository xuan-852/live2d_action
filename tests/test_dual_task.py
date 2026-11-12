import unittest
import numpy as np
from tools.control.dual_task import patch_predict,wrist_step


class DualTaskTests(unittest.TestCase):
    def test_patch_identity_and_cumulative_rotation(self):
        chain=[0,0,1,0,2,0,3,0]
        np.testing.assert_allclose(patch_predict(chain,[0,0],[0,0]),[2,0])
        np.testing.assert_allclose(patch_predict(chain,[0,0],[90,-90]),[1,1],atol=1e-8)

    def test_step_bounds(self):
        q=np.array([.99,0.])
        nxt=wrist_step(lambda z:z,q,np.array([5.,-5]),np.full(2,-1),np.ones(2),.1)
        self.assertTrue(np.all(nxt<=1))
        self.assertTrue(np.all(abs(nxt-q)<=.100001))
