import unittest
import numpy as np
from tools.control.pilot import Forward,step_control,LO,HI,STEP

class ControlPilotTests(unittest.TestCase):
    def test_forward_learns_coupled_quadratic(self):
        r=np.random.default_rng(5);z=r.uniform(-1,1,(80,3));x=z*np.array([30.,30.,3.]);y=np.c_[z[:,0]+z[:,1]**2,z[:,2]*z[:,0]]
        m=Forward();m.fit(x,y)
        np.testing.assert_allclose(m.predict(x),y,atol=.002)
    def test_action_limits(self):
        q=HI-.01
        a=step_control(q,0,np.array([100.,100.]),lambda z,c:z,q)
        self.assertTrue(np.all(abs(a)<=STEP));self.assertTrue(np.all(q+a<=HI))
