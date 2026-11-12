import unittest
import numpy as np
from tools.control.coordination import heading,angle_error

class CoordinationTests(unittest.TestCase):
    def test_heading_and_wrap(self):
        np.testing.assert_allclose(heading([0,0,1,0,1,1,2,1]),[0,90])
        np.testing.assert_allclose(angle_error([179,-179],np.array([-179,179])),[-2,2])
