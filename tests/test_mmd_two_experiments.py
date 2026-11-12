import unittest
import numpy as np
from tools.control.mmd_two_experiments import bridge,join

class MmdTwoExperimentTests(unittest.TestCase):
    def test_bridge_and_common_clock(self):
        b=bridge(np.zeros(6),np.ones(6),np.zeros(6),np.zeros(6))
        self.assertTrue(np.all(np.diff(b,axis=0)>0))
        clips=[('a',np.zeros((10,6))),('b',np.ones((10,6)))]
        a,la=join(clips,False);b,lb=join(clips,True)
        self.assertEqual(a.shape,b.shape);self.assertEqual(len(la),len(a));self.assertEqual(len(lb),len(b))
        np.testing.assert_array_equal(a[-60:],0)
