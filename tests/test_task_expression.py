import copy
import unittest
from tools.control.task_expression import TaskExpressionDirector,validate_task
from tools.control.task_expression_trial import task
from tools.control.body_session import nodes_goal

HOME=[[0,.3],[0,.2],[0,.1],[0,.06]]
SCENE={k:dict(label=k,xy=[.3,.1],gaze=[0,0],criterion='bilateral-node-hold',
              pointing=dict(nodes=nodes_goal(HOME),headBody=[0.]*6,reachSeconds=.5)) for k in ('A','C')}


def director():
    d=TaskExpressionDirector(SCENE)
    d.initialize(dict(time=0.,headBody=[0.]*6,arms={s:dict(actual=HOME,applied=[0.]*5) for s in ('native','patch')}))
    return d


def tick(d,i,reached=True,markers=None,geometry=None):
    p=d.next(i/30);r=dict(time=i/30,geometry=geometry or dict(flippedTriangles=0))
    for s in ('native','patch'):
        r[s]=dict(actual=HOME,applied=[0.]*5,status='reached' if reached else 'limited',reasons=[],
                  wristError=0. if reached else .1,elbowError=0.,palmErrorDeg=0.,terminalStateMet=reached,positionReached=reached)
    r.update(nativeFingers=dict(settled=True,reasons=[]),gaze=dict(settled=True,reasons=[]),face=dict(reasons=[]))
    if markers is not None:r['taskMarkers']=markers
    return p,d.observe(r,p['headBody'])


class TaskExpressionTests(unittest.TestCase):
    def test_goals_do_not_advance_on_elapsed_time_without_actual_arrival(self):
        d=director();d.submit(task('t','compare',['A','C'],'compare'),1/30)
        for i in range(1,20):tick(d,i,False)
        self.assertEqual(d.index,0);self.assertEqual(d.stable,0)

    def test_goal_requires_consecutive_actual_hold(self):
        d=director();d.submit(task('t','compare',['A','C'],'compare'),1/30)
        for i in range(1,13):tick(d,i)
        tick(d,13,False)
        for i in range(14,28):tick(d,i)
        self.assertEqual(d.index,0)
        tick(d,28);tick(d,29)
        self.assertEqual(d.index,1);self.assertEqual(d.outcomes[0]['result'],'execution-goal-met')

    def test_feedback_delay_changes_transition_time(self):
        a=director();b=director()
        for d in (a,b):d.submit(task('t','compare',['A','C'],'compare'),1/30)
        for i in range(1,30):tick(a,i);tick(b,i,i>=10)
        ta=next(e['time'] for e in a.events if e['event']=='actual-goal-held')
        tb=next(e['time'] for e in b.events if e['event']=='actual-goal-held')
        self.assertAlmostEqual(tb-ta,9/30)

    def test_actual_failure_lowers_confidence_and_marks_limited(self):
        d=director();d.submit(task('t','inspect',['A'],'inspect'),1/30)
        for i in range(1,50):tick(d,i,False)
        self.assertEqual(d.phase,'limited');self.assertLess(d.confidence,.85)
        self.assertEqual(d.outcomes[0]['result'],'limited')
        self.assertTrue(any(e['event']=='actual-feedback-fallback' for e in d.events))

    def test_cancel_removes_pending_goals_without_resume(self):
        d=director();d.submit(task('t','compare',['A','C'],'compare'),1/30)
        for i in range(1,10):tick(d,i)
        d.submit(task('stop','cancel',[],'stop'),10/30)
        for i in range(10,60):p,_=tick(d,i)
        self.assertEqual(d.phase,'cancelled');self.assertEqual(p['intent']['gesture'],'neutral')
        self.assertFalse(any(e.get('target')=='C' for e in d.events))

    def test_bad_task_does_not_replace_current_task(self):
        d=director();d.submit(task('t','inspect',['A'],'inspect'),1/30)
        with self.assertRaises(ValueError):d.submit(task('bad','inspect',['unknown'],'?'),1/30)
        self.assertEqual(d.task['id'],'t')
        with self.assertRaises(ValueError):d.submit(task('t','inspect',['C'],'?'),1/30)

    def test_input_order_drives_goal_order_not_fixture_clock(self):
        d=director();d.submit(task('t','compare',['C','A'],'compare'),1/30)
        p,_=tick(d,1);self.assertEqual(p['intent']['target'],'C')
        for i in range(2,17):p,_=tick(d,i)
        self.assertEqual(p['intent']['target'],'A')

    def test_task_cannot_inject_actuator_or_motor_schedule(self):
        t=task('t','inspect',['A'],'inspect');t['actuators']=[1,2]
        with self.assertRaises(ValueError):validate_task(t,SCENE)
        with self.assertRaises(ValueError):validate_task(task('x','compare',['A'],'?'),SCENE)

    def test_pointing_axis_cannot_be_substituted_by_nodes_arrival(self):
        scene=copy.deepcopy(SCENE);scene['A']['criterion']='native-index-material-axis'
        d=TaskExpressionDirector(scene)
        d.initialize(dict(time=0.,headBody=[0.]*6,arms={s:dict(actual=HOME,applied=[0.]*5) for s in ('native','patch')}))
        # The geometry helper rejects a ray pointing away, even with zero node error.
        from tools.control.task_expression import axis_error
        self.assertGreater(axis_error(dict(indexBase=[0.,0.],indexTip=[-1.,0.]),[1.,0.]),179.)
        d.submit(task('t','inspect',['A'],'inspect'),1/30)
        for i in range(1,50):tick(d,i,markers=dict(indexBase=[0.,0.],indexTip=[-1.,0.]))
        self.assertEqual(d.phase,'limited')
        self.assertEqual(d.outcomes[0]['reason'],'actual-task-constraint-not-met')

    def test_failure_selects_next_goal_and_revises_expression_without_external_event(self):
        d=director();d.submit(task('t','compare',['A','C'],'compare',.6),1/30)
        for i in range(1,35):p,_=tick(d,i,False)
        self.assertEqual(p['intent']['target'],'C')
        self.assertEqual(p['intent']['affect'],'uncertain')
        self.assertEqual(p['intent']['task']['failedTargets'],['A'])
        self.assertEqual(sum(e['event']=='task-accepted' for e in d.events),1)

    def test_failed_geometry_cannot_earn_goal_completion(self):
        # A ready node snapshot alone must not certify a failed execution backend.
        d=director();d.submit(task('t','inspect',['A'],'inspect'),1/30)
        for i in range(1,50):tick(d,i,geometry=dict(flippedTriangles=1,minTriangleAreaRatio=.4))
        self.assertEqual(d.phase,'limited')
        self.assertFalse(any(o['result']=='execution-goal-met' for o in d.outcomes))
