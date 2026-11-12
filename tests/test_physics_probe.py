import unittest
import numpy as np
from tools.control.physics_probe import transitions,ridge

class PhysicsProbeTests(unittest.TestCase):
    def test_alignment_and_no_future_observation(self):
        u=np.arange(484).reshape(121,4)
        s=u+1000
        rows=transitions(u,s)
        self.assertEqual(len(rows),118)
        for i,row in enumerate(rows,2):
            for key,expected in [('command',u[i+1]),('state',s[i]),('previous',s[i-1]),('priorInput',u[i]),('target',s[i+1])]:
                np.testing.assert_array_equal(row[key],expected)

    def test_invalid_samples_rejected(self):
        with self.assertRaises(ValueError):transitions(np.zeros((120,4)),np.zeros((120,4)))

    def test_ridge_heldout_affine_and_constant_feature(self):
        rng=np.random.default_rng(14)
        x=np.c_[rng.normal(size=(100,3)),np.ones(100)]
        y=x[:,:2]*2+3
        pred,_=ridge(x[:80],y[:80])
        np.testing.assert_allclose(pred(x[80:]),y[80:],atol=.001)
