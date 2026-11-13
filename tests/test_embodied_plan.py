import copy
import unittest
import numpy as np
from tools.control.embodied_plan import validate_model_plan,EmbodiedExecution,POSTURE_KEYS
from tools.control.model_body_loop import validate_reply
from tests.test_dialogue_body import SCENE,HOME


def plan():
    return dict(id='model-1',kind='reference',text='看这里。',target='A',affect='warm',utteranceSeconds=3.,
                posture=dict(zip(POSTURE_KEYS,[8.,-2.,2.,4.,1.,2.])),reachSeconds=2.5,holdSeconds=.5,
                finish=False,explanation='以头身配合右侧指向。')


def executor():
    e=EmbodiedExecution(SCENE)
    e.initialize(dict(time=0.,headBody=[0.]*6,arms={s:dict(actual=HOME,applied=[0.]*5) for s in ('native','patch')}))
    return e


class EmbodiedPlanTests(unittest.TestCase):
    def test_model_cannot_inject_cubism_actuators(self):
        p=plan();p['q']=[1,2]
        with self.assertRaises(ValueError):validate_model_plan(p,SCENE)

    def test_invalid_posture_does_not_replace_current_plan(self):
        e=executor();e.submit(plan(),1/30)
        p=plan();p['id']='bad';p['posture']['bodyTurn']=7
        with self.assertRaises(ValueError):e.submit(p,1/30)
        self.assertEqual(e.plan['id'],'model-1')
        self.assertEqual(e.body.session.active['id'],'model-1')

    def test_nonfinite_boolean_and_unknown_posture_axes_rejected(self):
        for v in (float('nan'),True):
            p=plan();p['posture']['bodyTurn']=v
            with self.assertRaises(ValueError):validate_model_plan(p,SCENE)
        p=plan();p['posture']['force']=1
        with self.assertRaises(ValueError):validate_model_plan(p,SCENE)

    def test_posture_is_forwarded_without_changing_arm_goal(self):
        e=executor();p=plan();e.submit(p,1/30)
        np.testing.assert_allclose(e.body.session.active['headBody'],[8,-2,2,4,1,2])
        self.assertEqual(e.body.session.active['arms']['native']['wrist'],HOME[2])
        self.assertAlmostEqual(e.body.deadline,2.5+1/30)

    def test_finish_requires_a_safe_intent_and_cancel_neutral_posture(self):
        p=plan();p['finish']=True
        with self.assertRaises(ValueError):validate_model_plan(p,SCENE)
        p['kind']='cancel';p['target']='user'
        with self.assertRaises(ValueError):validate_model_plan(p,SCENE)
        p['posture']={k:0 for k in POSTURE_KEYS}
        self.assertTrue(validate_model_plan(p,SCENE)['finish'])

    def test_stale_model_response_cannot_apply_to_new_observation(self):
        req=dict(requestId=2,observationTime=.5)
        reply=dict(requestId=1,observationTime=.5,plan=plan())
        with self.assertRaises(ValueError):validate_reply(reply,req,SCENE)
        reply['requestId']=2;reply['observationTime']=.4
        with self.assertRaises(ValueError):validate_reply(reply,req,SCENE)
        reply['observationTime']=.5
        self.assertEqual(validate_reply(reply,req,SCENE)['id'],'model-1')

    def test_bad_deadline_does_not_cancel_previous_request(self):
        e=executor();e.submit(plan(),1/30)
        p=plan();p['id']='bad';p['reachSeconds']=0
        with self.assertRaises(ValueError):e.submit(p,1/30)
        self.assertEqual(e.plan['id'],'model-1')

    def test_actual_numpy_context_exports_without_mutating_session(self):
        import json
        from tools.control.model_body_loop import export_feedback
        live=dict(time=1.,headBody=np.array([1.,2.,3.,4.,0.,0.]),arms={})
        snapshot=export_feedback(live)
        json.dumps(snapshot)
        self.assertIsInstance(live['headBody'],np.ndarray)
        snapshot['headBody'][0]=99
        self.assertEqual(live['headBody'][0],1)

    def test_short_utterance_does_not_cancel_unfinished_motor_goal(self):
        e=executor();p=plan();p['utteranceSeconds']=.5;e.submit(p,1/30)
        for i in range(1,100):packet=e.next(i/30)
        self.assertFalse(e.body.idle_released)
        self.assertEqual(packet['targets']['native']['wrist'],HOME[2])
        self.assertEqual(e.body.session.active['id'],'model-1')

    def test_new_user_turn_reopens_a_finished_expression(self):
        import tempfile
        from pathlib import Path
        from tools.control.model_body_loop import Provider
        with tempfile.TemporaryDirectory() as root:
            provider=Provider(Path(root),SCENE);provider.finished=True
            provider.receive_user('新的任务')
            self.assertFalse(provider.finished)
            self.assertEqual(provider.query_reason,'new-user-turn')
            self.assertEqual(provider.messages[-1]['text'],'新的任务')

    def test_exhausted_budget_stops_safely_without_fabricating_model_response(self):
        import tempfile
        from pathlib import Path
        from tools.control.model_body_loop import Provider
        with tempfile.TemporaryDirectory() as root:
            p=Provider(Path(root),SCENE)
            p.initialize(dict(time=0.,headBody=[0.]*6,arms={s:dict(actual=HOME,applied=[0.]*5) for s in ('native','patch')}))
            p.history=[{} for _ in range(10)]
            packet=p.next(1/30)
            self.assertTrue(p.budget_stop);self.assertEqual(len(p.history),10)
            self.assertEqual(packet['intent']['kind'],'cancel')
            self.assertFalse(p.guards[0]['modelGenerated'])
