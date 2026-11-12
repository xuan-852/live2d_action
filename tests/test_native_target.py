import unittest
import numpy as np
from tools.control.stateful_arm import Target,constrained_step,speed_limits,DT
from tools.control.timed_arm import heading
from tools.control.native_target import NativeTargetController,NATIVE_KEYS
from tools.control.pilot import Plant


class NativeTargetTests(unittest.TestCase):
    def test_bounded_fit_preserves_reachable_wrist_when_elbow_conflicts(self):
        plant=Plant();model=plant.model
        def forward(q):
            model.set(dict(zip(NATIVE_KEYS,q)));a=plant.anchors.read()
            return np.array([a[k] for k in ('shoulder','elbow','wrist','palm')])
        initial=np.zeros(5);nodes=forward(initial)
        c=NativeTargetController(forward,model,plant.anchors,initial,nodes,**speed_limits('fast'))
        # Fixed node request reproducing the boundary case; no actuator solution
        # supplied to the solver, and no requirement is removed or reweighted.
        wrist=np.array([-.0590522736,.3604157423]);elbow=np.array([.0366799312,.2127760301])
        palm=np.array([-.0776060429,.3862231293])
        target=Target(wrist,1.,elbow=elbow,palm_heading=np.degrees(np.arctan2(*(palm-wrist)[::-1])))
        goal,actual,residual=c._fit_endpoint(target,forward,initial)
        components,reasons=c.endpoint_diagnostics(target,actual)
        self.assertLess(components['wrist'],.001)
        self.assertGreater(components['elbow'],.01)
        self.assertLess(components['palmDegrees'],.2)
        self.assertIn('elbow-target-limited',reasons)
        self.assertNotIn('wrist-target-limited',reasons)
        self.assertGreater(residual,1.)
        self.assertTrue(np.all(goal>=c.lo) and np.all(goal<=c.hi))

    def test_large_finite_goal_does_not_bypass_node_speed_constraint(self):
        plant=Plant();model=plant.model
        def forward(q):
            model.set(dict(zip(NATIVE_KEYS,q)));a=plant.anchors.read()
            return np.array([a[k] for k in ('shoulder','elbow','wrist','palm')])
        initial=np.zeros(5);nodes=forward(initial);target_nodes=forward([37.,7.,20.,-2.,-.9])
        c=NativeTargetController(forward,model,plant.anchors,initial,nodes,
            speed=2700,acceleration=162000,node_speed=3,node_acceleration=180)
        q,info=c.command(Target(target_nodes[2],DT,elbow=target_nodes[1],palm_heading=heading(target_nodes),mode='track'),DT)
        self.assertFalse(info['policy'].get('finiteTargetAccepted',False))
        row=c.observe(q,forward(q))
        self.assertLessEqual(row['metrics']['maxNodeSpeed'],3.001)
        self.assertGreater(row['wristError'],.1)

    def test_finite_target_uses_actual_nodes_and_preserves_actual_frame_limits(self):
        plant=Plant();model=plant.model
        def forward(q):
            model.set(dict(zip(NATIVE_KEYS,q)));a=plant.anchors.read()
            return np.array([a[k] for k in ('shoulder','elbow','wrist','palm')])
        initial=np.array([36.,7.,19.,-2.,-.9]);nodes=forward(initial)
        target_nodes=forward([37.,7.,20.,-2.,-.9])
        c=NativeTargetController(forward,model,plant.anchors,initial,nodes,
            speed=2700,acceleration=162000,node_speed=3,node_acceleration=180)
        target=Target(target_nodes[2],DT,elbow=target_nodes[1],palm_heading=heading(target_nodes),mode='track')
        q,info=c.command(target,DT)
        self.assertTrue(info['policy']['finiteTargetAccepted'])
        row=c.observe(q,forward(q))
        self.assertLess(row['wristError'],.0001)
        self.assertLess(row['elbowError'],.0001)
        self.assertLessEqual(row['metrics']['maxNodeSpeed'],3.)
        self.assertNotEqual(row['status'],'failed')

    def test_hold_refits_current_context_without_restarting_deadline(self):
        plant=Plant();model=plant.model;context={}
        def forward(q):
            model.set(context|dict(zip(NATIVE_KEYS,q)));a=plant.anchors.read()
            return np.array([a[k] for k in ('shoulder','elbow','wrist','palm')])
        initial=np.array([37.,7.,20.,-2.,-.9]);nodes=forward(initial)
        c=NativeTargetController(forward,model,plant.anchors,initial,nodes,**speed_limits('fast'))
        target=Target(nodes[2],.1,elbow=nodes[1],palm_heading=heading(nodes))
        q,_=c.command(target,DT);c.observe(q,forward(q))
        context['ParamBodyAngleX2']=1.
        c.nodes=forward(c.q)
        q,info=c.command(target,.5)
        self.assertTrue(info['policy']['currentContextCorrection'])
        self.assertEqual(info['deadline'],.1)
        self.assertLess(np.linalg.norm(np.array(info['policy']['referenceNodes'])[2]-target.wrist),.0001)

    def test_five_axis_box_qp(self):
        h=np.diag([1.,2.,3.,4.,5.]);free=np.array([3.,-2.,.3,-.2,4.])
        a=np.vstack([np.eye(5),-np.eye(5)]);b=np.ones(10)
        q,ok=constrained_step(h,h@free,a,b)
        self.assertTrue(ok)
        np.testing.assert_allclose(q,np.clip(free,-1,1),atol=1e-7)

    def test_known_sdk_model_overhead_pose_can_be_solved_without_fixture_actuators(self):
        plant=Plant();model=plant.model
        def forward(q):
            model.set(dict(zip(NATIVE_KEYS,q)));a=plant.anchors.read()
            return np.array([a[k] for k in ('shoulder','elbow','wrist','palm')])
        home=forward(np.zeros(5));requested=forward([37.,7.,20.,-2.,-.9])
        self.assertGreater(requested[2,1],home[0,1])
        c=NativeTargetController(forward,model,plant.anchors,np.zeros(5),home,**speed_limits('fast'))
        target=Target(requested[2],2.,elbow=requested[1],palm_heading=heading(requested))
        q,n,res=c._fit_endpoint(target,forward,np.zeros(5))
        self.assertLess(np.linalg.norm(n[2]-requested[2]),.0001)
        self.assertLess(np.linalg.norm(n[1]-requested[1]),.0001)
        command,_=c.command(target,DT)
        self.assertEqual(len(command),5)
        self.assertLessEqual(np.max(abs(command)),c.acceleration*DT**2)
        row=c.observe(command,forward(command))
        self.assertNotEqual(row['status'],'failed')
