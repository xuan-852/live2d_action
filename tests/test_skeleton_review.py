import unittest
import numpy as np
from tools.control.skeleton_review import normalise

class SkeletonReviewTests(unittest.TestCase):
    def test_clip_transform_preserves_root_motion(self):
        a={'pelvis':[0,0,0],'head':[0,2,0]};b={'pelvis':[2,0,0],'head':[2,2,0]}
        frames,_=normalise([a,b])
        np.testing.assert_allclose(frames[1]['pelvis'],[1,0,0])
    def test_invalid_scale_rejected(self):
        with self.assertRaises(ValueError):normalise([{'pelvis':[0,0,0],'head':[0,0,0]}])
