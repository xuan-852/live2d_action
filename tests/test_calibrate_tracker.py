import unittest
import numpy as np
from tools.control.calibrate_virtual_tracker import attenuate,calibrate

class CalibrateTrackerTests(unittest.TestCase):
    def test_separate_weights_and_preserve_source(self):
        src={'PhysicsSettings':[{'Output':[{'Destination':{'Id':'ParamBodyAngleX2'},'Weight':80},
              {'Destination':{'Id':'Param12'},'Weight':60}]}]}
        out=attenuate(src,.5,.25)
        self.assertEqual([p['Weight'] for p in out['PhysicsSettings'][0]['Output']],[40,15])
        self.assertEqual(src['PhysicsSettings'][0]['Output'][0]['Weight'],80)
    def test_mapping_keeps_temporal_values(self):
        x=np.ones((3,6));x[1]*=2;y=calibrate(x)
        np.testing.assert_allclose(y[1],2*y[0])
