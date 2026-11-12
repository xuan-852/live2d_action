import unittest
import numpy as np
from tools.control.node_policy import demonstration,observation,decode,Network

class NodePolicyTests(unittest.TestCase):
    def test_node_interface_and_chain_teacher(self):
        home=np.array([[[-1,0],[-1,-1],[-1,-2]],[[1,0],[1,-1],[1,-2]]],float)
        angles=np.array([[20,10],[-10,20]])
        np.testing.assert_allclose(demonstration(home,angles,0),home)
        end=demonstration(home,angles,1)
        np.testing.assert_allclose(np.linalg.norm(np.diff(end,axis=1),axis=-1),1)
        self.assertEqual(observation(home,home,end[:,2],0,2).shape,(17,))
        np.testing.assert_array_equal(decode(np.zeros(8),home,2),home)
        self.assertEqual(Network().predict(np.zeros(17)).shape,(8,))
