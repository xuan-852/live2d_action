import unittest
import numpy as np
from tools.control.node_overlay_trial import stimuli

class NodeOverlayTests(unittest.TestCase):
    def test_isolation_and_reproducibility(self):
        x=stimuli();np.testing.assert_array_equal(x,stimuli())
        self.assertEqual(x.shape,(361,6))
        for section in range(3):
            for j in range(6):
                if j%3!=section:self.assertTrue(np.all(x[section*90:(section+1)*90,j]==0))
        np.testing.assert_array_equal(x[-1],np.zeros(6))
