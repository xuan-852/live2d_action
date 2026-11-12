import unittest
import numpy as np
from tools.control.mmd_virtual_tracker import signals

class MmdVirtualTrackerTests(unittest.TestCase):
    def test_relative_local_not_world_sum(self):
        d={'coordinated':[{'headLocal':dict(yaw=i,pitch=0,roll=0),'body':dict(yaw=3*i,pitch=0,roll=0)} for i in (10,11)],
           'sides':{'anatomicalLeft':{'frames':[{'seconds':0},{'seconds':1/30}]}}}
        _,_,v,_=signals(d)
        np.testing.assert_allclose(v[0],0)
        np.testing.assert_allclose(v[1,[0,3]],[1,1])
