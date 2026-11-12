import unittest
import numpy as np
from tools.control.task_objective import evaluate


class TaskObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.goals=np.array([[-1.,0],[1,0]])
        self.actual=np.repeat(self.goals[None],30,axis=0)

    def test_correct_but_unreviewed_is_not_visible_success(self):
        r=evaluate(self.actual,self.goals,.1,15)
        self.assertTrue(r['materialTaskPassed'])
        self.assertFalse(r['visibleTaskPassed'])

    def test_verified_hold(self):
        self.assertTrue(evaluate(self.actual,self.goals,.1,15,[True]*30)['visibleTaskPassed'])

    def test_one_hand_and_identity_swap_fail(self):
        self.actual[:,0]=0
        self.assertFalse(evaluate(self.actual,self.goals,.1,15)['materialTaskPassed'])
        self.assertFalse(evaluate(np.repeat(self.goals[::-1][None],30,axis=0),self.goals,.1,15)['materialTaskPassed'])

    def test_intermittent_hits_and_occlusion_fail(self):
        self.actual[::2]+=1
        self.assertFalse(evaluate(self.actual,self.goals,.1,15)['materialTaskPassed'])
        exact=np.repeat(self.goals[None],30,axis=0)
        self.assertFalse(evaluate(exact,self.goals,.1,15,[False]*30)['visibleTaskPassed'])

    def test_bad_evidence_rejected(self):
        self.actual[0,0,0]=np.nan
        with self.assertRaises(ValueError):evaluate(self.actual,self.goals,.1,15)
