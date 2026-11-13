import unittest,json
from pathlib import Path
import numpy as np
from tools.control.patch_projection import PatchProjectionPlant,PatchProjectionController,projection_guard,projected_shape_guard
from tools.control.pilot import Plant
from tools.control.stateful_arm import Target,speed_limits
from tools.control.timed_arm import heading


class PatchProjectionTests(unittest.TestCase):
    def test_rooted_connection_preserves_identity_and_distal_axis_chain(self):
        p,approved=self.plant();r=PatchProjectionPlant(p.model,approved,rooted_upper=True)
        np.testing.assert_allclose(r.forward(np.zeros(6)),approved,atol=2e-7)
        q=np.array([-48.,110.,-30.,-15.,-15.,-45.])
        # Only shoulder attachment mesh changes, not the approved distal axes.
        np.testing.assert_allclose(r.forward(q)[1:],p.forward(q)[1:],atol=2e-7)
    def test_projection_local_guard_reduces_to_original_area_at_identity(self):
        np.testing.assert_allclose(projected_shape_guard([.6,.8],[1,1],[.6,.8],[1,1]),[.6,.8])

    def test_projection_local_guard_checks_real_projected_distortion(self):
        self.assertGreater(float(projected_shape_guard(.48,.8,.6,.8)),.55)
        self.assertLess(float(projected_shape_guard(.2,.8,.25,.8)),.5)
        self.assertLess(float(projected_shape_guard(.5,-.1,0,.8)),.5)
    def test_palm_envelope_is_used_when_feasible(self):
        p,_=self.plant();q=np.array([-15.,35.,0.,-10.,-10.,-10.]);n=p.forward(q)
        c=PatchProjectionController(p.forward,lambda:p.raw_nodes,q,n,**speed_limits('fast'))
        target=Target(n[2],1/30,elbow=n[1],palm_heading=heading(n)+2.,projection=c.ratios(n),mode='track')
        command,info=c.command(target,1/30)
        self.assertEqual(info['policy']['palmDirectionConstraint'],'enforced')
        row=c.observe(command,p.forward(command))
        self.assertLess(row['palmErrorDeg'],4.)

    def test_impossible_one_frame_palm_turn_reports_relaxation_not_safety_bypass(self):
        p,_=self.plant();q=np.zeros(6);n=p.forward(q)
        c=PatchProjectionController(p.forward,lambda:p.raw_nodes,q,n,**speed_limits('fast'))
        target=Target(n[2],1/30,elbow=n[1],palm_heading=heading(n)+90.,projection=[1,1,1],mode='track')
        command,info=c.command(target,1/30)
        self.assertIn('palm-direction-envelope-not-enforced',info['reasons'])
        row=c.observe(command,p.forward(command))
        self.assertLessEqual(row['metrics']['maxNodeSpeed'],.721)
        self.assertLessEqual(row['metrics']['maxParameterStep'],4.501)
        self.assertIn('palm-direction-limited',row['reasons'])
    def test_projected_length_jacobian_matches_finite_node_response(self):
        p,_=self.plant();q=np.array([-15.,35.,-10.,-15.,-20.,-25.]);n=p.forward(q)
        c=PatchProjectionController(p.forward,lambda:p.raw_nodes,q,n,**speed_limits('fast'))
        step=1e-3
        raw=np.stack([((p.forward(q+np.eye(6)[i]*step)-n)/step).ravel() for i in range(6)],axis=1)
        expected=np.stack([(c.ratios(p.forward(q+np.eye(6)[i]*step))-c.ratios(n))/step/.03 for i in range(6)],axis=1)
        np.testing.assert_allclose(c.ratio_jacobian(n,raw),expected,atol=1e-5)
    def test_guard_separates_intended_shortening_from_extra_collapse(self):
        self.assertAlmostEqual(float(projection_guard(.6,1.,.6)),1.)
        self.assertLess(float(projection_guard(.2,1.,.6)),.5)
        self.assertLess(float(projection_guard(.4,.4,.6)),.5)
        self.assertLess(float(projection_guard(.25,1.,.55)),.5)
    def plant(self):
        p=Plant();nodes=np.array(json.loads(Path('E:/FuXuanLab/runs/20261109-wrist-axis-revision-01/protocol.json').read_text())['newNodes'])
        return PatchProjectionPlant(p.model,nodes),nodes

    def test_identity_keeps_approved_nodes_and_shorten_changes_correct_segments(self):
        p,approved=self.plant();np.testing.assert_allclose(p.forward(np.zeros(6)),approved,atol=2e-7)
        base=np.linalg.norm(np.diff(p.raw_nodes,axis=0),axis=1)
        for axis,minimum in [(3,.8),(4,.75),(5,.45)]:
            q=np.array([-15.,35.,0,0,0,0]);q[axis]=(minimum-1)*100
            measured=np.linalg.norm(np.diff(p.forward(q),axis=0),axis=1)/base
            self.assertAlmostEqual(measured[axis-3],minimum,delta=.001)
            for other in range(3):
                if other!=axis-3:self.assertAlmostEqual(measured[other],1,delta=.001)

    def test_full_projection_inverse_from_nodes_and_ratios(self):
        p,_=self.plant();home=p.raw_nodes.copy();q=np.array([-15.,35.,0,-15,-20,-25])
        requested=p.forward(q);target=Target(requested[2],1.,elbow=requested[1],palm_heading=heading(requested),projection=[.85,.8,.75])
        c=PatchProjectionController(p.forward,lambda:p.raw_nodes,np.zeros(6),home,**speed_limits('fast'))
        goal,nodes,residual=c._fit_endpoint(target,p.forward,np.zeros(6))
        self.assertLess(np.linalg.norm(nodes[2]-requested[2]),.0002)
        np.testing.assert_allclose(c.ratios(nodes),target.projection,atol=.002)
        self.assertTrue(np.all(goal>=c.lo) and np.all(goal<=c.hi))

    def test_outside_projection_range_is_retained_as_residual(self):
        p,_=self.plant();q=np.array([0.,0.,0.,0.,0.,-45.]);nodes=p.forward(q)
        c=PatchProjectionController(p.forward,lambda:p.raw_nodes,np.zeros(6),p.raw_nodes,**speed_limits('fast'))
        target=Target(nodes[2],1.,elbow=nodes[1],palm_heading=heading(nodes),projection=[1,1,.3])
        goal,actual,residual=c._fit_endpoint(target,p.forward,np.zeros(6))
        self.assertGreaterEqual(c.ratios(actual)[2],.4-1e-6)
        self.assertGreater(residual,1.)
        self.assertAlmostEqual(target.projection[2],.3)

    def test_invalid_ratio_shape_and_nonfinite_rejected(self):
        for value in ([1,1],[1,-1,1],[1,float('nan'),1]):
            with self.assertRaises(ValueError):Target([0,0],1,projection=value)

    def test_projection_feedback_is_json_serializable(self):
        p,_=self.plant();c=PatchProjectionController(p.forward,lambda:p.raw_nodes,np.zeros(6),p.raw_nodes,**speed_limits('fast'))
        n=p.raw_nodes;c.command(Target(n[2],1.,elbow=n[1],palm_heading=heading(n),projection=[1,1,1]),1/30)
        row=c.observe(np.zeros(6),n)
        json.dumps(row)
        self.assertIs(type(row['projectionStateMet']),bool)
