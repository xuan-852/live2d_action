import unittest
import numpy as np
from tools.control.sequence_training import feature,model_nodes

class SequenceTrainingTests(unittest.TestCase):
    def test_history_and_model_lengths(self):
        self.assertEqual(feature(np.zeros((5,4))).shape,(20,))
        home=np.array([[[0,0],[0,-1],[0,-2],[0,-3]],[[1,0],[1,-1],[1,-2],[1,-3]]],float)
        nodes=model_nodes(home,np.array([10,30,-20,15]))
        np.testing.assert_allclose(np.linalg.norm(np.diff(nodes,axis=1),axis=-1),1,atol=1e-10)
        rotated=model_nodes(home,np.array([np.pi/2,np.pi/2,0,0]))
        np.testing.assert_allclose(rotated[0,1],[1,0],atol=1e-10)
