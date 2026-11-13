import copy
import unittest
import numpy as np
from tools.control.dialogue_body import DialogueBody,validate_plan
from tools.control.dialogue_body_trial import event
from tools.control.body_session import nodes_goal
from tools.control.face_control import FaceController,NEUTRAL

HOME=[[0,.3],[0,.2],[0,.1],[0,.06]]
SCENE={'A':dict(gaze=[.7,-.25],pointing=dict(nodes=nodes_goal(HOME),headBody=[14,-3,0,0,0,0])),
       'B':dict(gaze=[-.6,.1],pointing=None)}


def body():
    b=DialogueBody(SCENE)
    b.initialize(dict(time=0.,headBody=[0.]*6,arms={s:dict(actual=HOME,applied=[0.]*5) for s in ('native','patch')}))
    return b


def tick(b,i,reached=True):
    p=b.next(i/30)
    r=dict(time=i/30)
    for s in ('native','patch'):
        r[s]=dict(actual=HOME,applied=[0.]*5,status='reached' if reached else 'limited',reasons=[] if reached else ['deadline-missed'],
                  wristError=0. if reached else .1,elbowError=0.,palmErrorDeg=0.,terminalStateMet=reached,positionReached=reached)
    report=b.observe(r,p['headBody'])
    return p,report


class DialogueBodyTests(unittest.TestCase):
    def test_rejects_unknown_capability_and_actuator_injection(self):
        p=event('x','say','hello');p['q']=[1,2]
        with self.assertRaises(ValueError):validate_plan(p,SCENE)
        with self.assertRaises(ValueError):validate_plan(event('x','fist','hello'),SCENE)

    def test_invalid_plan_does_not_cancel_active_plan(self):
        b=body();b.submit(event('a','reference','A','A'),1/30)
        old=copy.deepcopy(b.session.active)
        with self.assertRaises(ValueError):b.submit(event('b','reference','?','missing'),1/30)
        self.assertEqual(b.plan['id'],'a');self.assertEqual(b.session.active['id'],old['id'])

    def test_unknown_referent_is_rejected_not_substituted(self):
        with self.assertRaises(ValueError):validate_plan(event('x','reference','where','unknown'),SCENE)
        with self.assertRaises(ValueError):validate_plan(event('x','reference','where',[]),SCENE)

    def test_decision_does_not_depend_on_fixture_global_frame(self):
        a=body();b=body()
        a.submit(event('listen','listen','',seconds=0),1/30)
        b.submit(event('listen','listen','',seconds=0),1/30)
        for i in range(1,61):tick(a,i);tick(b,i)
        a.submit(event('different-order','reference','B','B'),61/30)
        b.submit(event('different-order','reference','A','A'),61/30)
        pa,_=tick(a,61);pb,_=tick(b,61)
        self.assertEqual(pa['intent']['gesture'],'neutral')
        self.assertEqual(pb['intent']['gesture'],'index-emphasis')
        self.assertNotEqual(pa['gazeGoal'],pb['gazeGoal'])

    def test_registered_unavailable_target_uses_gaze_without_point(self):
        b=body();b.submit(event('b','reference','B','B'),1/30);p,_=tick(b,1)
        self.assertEqual(p['intent']['gesture'],'neutral')
        self.assertEqual(p['gazeGoal']['values'],[-.6,.1])
        self.assertEqual(p['targets']['native']['wrist'],HOME[2])
        self.assertIsNotNone(p['intent']['fallback'])

    def test_cancel_changes_ownership_immediately_and_never_resumes(self):
        b=body();b.submit(event('a','reference','A','A'),1/30);tick(b,1)
        b.submit(event('stop','cancel','stop',seconds=0),2/30)
        for i in range(2,70):
            p,_=tick(b,i);self.assertEqual(p['intent']['planId'],'stop')
            self.assertEqual(p['intent']['gesture'],'neutral')

    def test_utterance_end_releases_once_without_repeated_restarts(self):
        b=body();b.submit(event('a','say','hello',seconds=.2),1/30)
        for i in range(1,100):tick(b,i)
        self.assertEqual(sum(e['event']=='utterance-finished' for e in b.events),1)
        self.assertTrue(b.idle_released)

    def test_actual_failure_triggers_pointing_fallback(self):
        b=body();b.submit(event('a','reference','A','A',seconds=10),1/30)
        for i in range(1,70):tick(b,i,reached=False)
        self.assertEqual(b.fallback,'actual-pointing-target-not-reached')
        self.assertEqual(b.gesture,'neutral')

    def test_duplicate_ids_are_rejected(self):
        b=body();p=event('same','listen','');b.submit(p,1/30)
        with self.assertRaises(ValueError):b.submit(p,1/30)

    def test_listening_does_not_add_hand_gestures_or_expire(self):
        b=body();b.submit(event('a','listen','',seconds=0),1/30)
        for i in range(1,100):p,_=tick(b,i)
        self.assertFalse(p['intent']['released']);self.assertEqual(p['intent']['gesture'],'neutral')

    def test_face_switches_from_actual_without_resets_and_recovers(self):
        c=FaceController(NEUTRAL)
        for i in range(150):
            goal=[.8,.8,.2,.35,.18,.18] if i<55 else NEUTRAL
            command=c.command(goal,i/30);r=c.observe(command.astype(np.float32))
            self.assertFalse(r['reasons'])
        np.testing.assert_allclose(c.q,NEUTRAL,atol=1e-5)
        with self.assertRaises(ValueError):c.command([2]*6,6)
