import unittest
import numpy as np
from tools.control.closed_loop import increment
from tools.control.pilot import HI,LO,STEP


class ClosedLoopTests(unittest.TestCase):
    def test_limits_and_feedback_direction(self):
        jac=np.array([[1.,0],[0,1],[0,0],[0,0]])
        q=HI-.01
        d=increment(jac,np.array([10.,-10,0,0]),q)
        self.assertTrue(np.all(abs(d)<=STEP))
        self.assertTrue(np.all(q+d<=HI))
        self.assertTrue(np.all(q+d>=LO))
        self.assertGreater(d[0],0)
        self.assertLess(d[1],0)
