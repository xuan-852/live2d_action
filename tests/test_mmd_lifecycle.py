import unittest
import numpy as np
from tools.control.mmd_lifecycle import timeline

class LifecycleTests(unittest.TestCase):
    def test_complete_sources_preserved_and_settle(self):
        v=np.arange(36).reshape(6,6)*.01
        x,labels=timeline([('one',v),('two',v)])
        self.assertEqual(len(x),2*(30+24+6+24+60))
        np.testing.assert_array_equal(x[54:60],v)
        np.testing.assert_array_equal(x[-60:],0)
        self.assertEqual(labels[54]['phase'],'source')
