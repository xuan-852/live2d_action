import unittest
import numpy as np
from tools.control.patch_node_revision import section_midpoint

class PatchNodeRevisionTests(unittest.TestCase):
    def test_section_uses_mesh_intersection(self):
        vertices=np.array([[0.,0],[2,0],[2,2],[0,2]])
        point,span=section_midpoint(vertices,np.array([[0,1,2],[0,2,3]]),1)
        np.testing.assert_allclose(point,[1,1]);np.testing.assert_allclose(span,[0,2])
    def test_missing_section_rejected(self):
        with self.assertRaises(ValueError):section_midpoint(np.array([[0,0],[1,0],[0,1]]),[[0,1,2]],2)
