import copy,unittest
import numpy as np
from tools.control.body_session import BodyExecutionSession,nodes_goal
from tools.control.stateful_arm import DT

HOME=np.array([[0.,0.],[0.,.1],[0.,.2],[.03,.2]])


def session():
    s=BodyExecutionSession();s.initialize(dict(time=0.,headBody=[0.]*6,
        arms={x:dict(actual=HOME.tolist(),applied=[0.]*3) for x in ('native','patch')}));return s


def feedback(s,now,offset=0.,settled=True):
    row=dict(time=now)
    for side in ('native','patch'):
        p=HOME.copy();p[1:,0]+=offset
        row[side]=dict(actual=p.tolist(),applied=[1.,2.,3.],status='reached' if settled else 'limited',
            reasons=[] if settled else ['elbow-target-limited'],wristError=0 if settled else .1,
            elbowError=0 if settled else .1,palmErrorDeg=0,terminalStateMet=settled,positionReached=settled)
    return s.observe(row,s.last_packet['headBody'])


class BodySessionTests(unittest.TestCase):
    def test_projection_limit_prevents_false_settle_and_freeze_keeps_measured_length(self):
        s=session();arms={x:nodes_goal(HOME) for x in ('native','patch')}
        s.submit_pose('p',arms,[0.]*6,.1,DT);s.packet(DT)
        row=dict(time=DT)
        for side in ('native','patch'):
            row[side]=dict(actual=HOME.tolist(),applied=[0.]*3,status='reached',reasons=[],
                wristError=0.,elbowError=0.,palmErrorDeg=0.,terminalStateMet=True,positionReached=True)
        row['patch'].update(projectionMeasured=[.85,.8,.65],projectionApplied=[.84,.8,.65],projectionResidual=[0.,0.,.1],
            projectionStateMet=False,status='limited',reasons=['projection-target-limited'])
        report=s.observe(row,[0.]*6)
        self.assertFalse(report['settled'])
        s.freeze('f',2*DT,.3);packet=s.packet(2*DT)
        self.assertEqual(packet['targets']['patch']['projection'],[.85,.8,.65])

    def test_route_samples_and_timing_are_not_changed(self):
        s=session();arms={x:nodes_goal(HOME) for x in ('native','patch')}
        samples=[dict(time=i*DT,arms=copy.deepcopy(arms),headBody=[float(i),0,0,0,0,0]) for i in range(3)]
        original=copy.deepcopy(samples);s.submit_route('route',samples,2*DT,DT)
        seen=[]
        for i in range(1,7):
            p=s.packet(i*DT)
            if p['sourceTime'] is not None:
                seen.append(p['sourceTime']);self.assertEqual(p['headBody'],samples[len(seen)-1]['headBody'])
                self.assertEqual(p['targets']['native']['wrist'],arms['native']['wrist'])
            feedback(s,i*DT)
        self.assertEqual(seen,[0.,DT,2*DT]);self.assertEqual(samples,original)
        self.assertEqual(s.history[0]['sourceSamplesDelivered'],3)

    def test_interrupt_uses_actual_pose_and_never_resumes_old_route(self):
        s=session();arms={x:nodes_goal(HOME) for x in ('native','patch')}
        samples=[dict(time=i*DT,arms=arms,headBody=[0.]*6) for i in range(8)]
        s.submit_route('r',samples,DT,DT);s.packet(DT);feedback(s,DT,offset=.027)
        s.freeze('stop',2*DT,.5);p=s.packet(2*DT)
        self.assertAlmostEqual(p['targets']['native']['wrist'][0],.027)
        self.assertEqual(s.history[0]['endedReason'],'interrupted')
        for i in range(2,7):
            if i>2:p=s.packet(i*DT)
            self.assertEqual(p['session']['requestId'],'stop');self.assertIsNone(p['sourceTime'])
            feedback(s,i*DT,offset=.027)

    def test_release_does_not_claim_idle_from_old_settled_feedback(self):
        s=session();arms={x:nodes_goal(HOME) for x in ('native','patch')}
        s.submit_pose('p',arms,[0.]*6,.1,DT);s.packet(DT);feedback(s,DT)
        s.release('release',2*DT,3*DT);p=s.packet(2*DT)
        self.assertNotEqual(p['session']['phase'],'idle');feedback(s,2*DT,settled=False)
        p=s.packet(3*DT);self.assertNotEqual(p['session']['phase'],'idle');feedback(s,3*DT)
        p=s.packet(4*DT);self.assertEqual(p['session']['phase'],'idle')

    def test_limits_are_feedback_not_success_and_clock_pairing_is_strict(self):
        s=session();arms={x:nodes_goal(HOME) for x in ('native','patch')}
        s.submit_pose('p',arms,[0.]*6,.1,DT);s.packet(DT);r=feedback(s,DT,settled=False)
        self.assertEqual(r['executionStatus'],'limited');self.assertFalse(r['settled'])
        self.assertEqual(r['arms']['native']['applied'],[1,2,3])
        with self.assertRaises(ValueError):s.packet(3*DT)

    def test_invalid_request_does_not_supersede_active_goal(self):
        s=session();arms={x:nodes_goal(HOME) for x in ('native','patch')}
        s.submit_pose('p',arms,[0.]*6,.1,DT)
        with self.assertRaises(ValueError):s.submit_pose('bad',arms,[100.]*6,.2,DT)
        self.assertEqual(s.active['id'],'p');self.assertNotIn('endedReason',s.history[0])
        with self.assertRaises(ValueError):s.submit_route('bad',[],DT,DT)

    def test_feedback_is_an_owned_actual_snapshot(self):
        s=session();arms={x:nodes_goal(HOME) for x in ('native','patch')}
        s.submit_pose('p',arms,[0.]*6,.1,DT);s.packet(DT);r=feedback(s,DT,offset=.012)
        r['arms']['native']['actual'][2][0]=99.
        s.freeze('f',2*DT,.3);p=s.packet(2*DT)
        self.assertAlmostEqual(p['targets']['native']['wrist'][0],.012)

    def test_body_connection_inherits_measured_input_velocity(self):
        s=session();arms={x:nodes_goal(HOME) for x in ('native','patch')}
        s.submit_pose('p',arms,[5.,0,0,0,0,0],1.,DT);s.packet(DT);feedback(s,DT)
        s.packet(2*DT);feedback(s,2*DT)
        measured=(s.feedback['headBody']-s.previous_feedback['headBody'])/DT
        s.submit_pose('new',arms,[-5.,0,0,0,0,0],1.,3*DT)
        np.testing.assert_allclose(s.active['contextVelocity'],measured)
        self.assertGreater(abs(measured[0]),0.)
