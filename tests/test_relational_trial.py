import unittest
import numpy as np
from tools.whole.expression_trial import target,bounded_solve

class RelationalTrialTests(unittest.TestCase):
    def test_body_relative_target_retains_absolute_pose_and_scale(self):
        p={'左腕':[1,2,0],'右腕':[-1,2,0],'左ひじ':[2,1,0],'左手首':[3,1,0],
           '右ひじ':[-2,1,0],'右手首':[-3,1,0]}
        a=target(p)
        shifted={k:(np.array(v)*2+np.array([7,8,0])).tolist() for k,v in p.items()}
        np.testing.assert_allclose(a,target(shifted))
        self.assertEqual(a[1,0],1.5)
        self.assertEqual(a[1,1],-.5)

    def test_solver_respects_bound_when_target_is_unreachable(self):
        q,cost,calls=bounded_solve(lambda x:x-np.array([3.,-2.]),np.zeros(2),np.array([-1.,-1.]),np.ones(2))
        np.testing.assert_allclose(q,[1.,-1.],atol=.001)
        self.assertGreater(cost,0)

    def test_coupled_residual_recovers_two_coordinates(self):
        q,cost,calls=bounded_solve(lambda x:np.array([x[0]+x[1]-3,x[0]-x[1]-1]),np.zeros(2),np.full(2,-5),np.full(2,5))
        np.testing.assert_allclose(q,[2,1],atol=.001)
