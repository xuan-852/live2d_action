import unittest
import numpy as np
from tools.control.palm_plane import palm_basis
from tools.control.patch_projection import PatchProjectionPlant,PatchProjectionController
from tools.control.pilot import Plant
from tools.control.validate_stateful_arm import APPROVED,pose
from tools.control.stateful_arm import Target,speed_limits
from tools.control.timed_arm import heading
import json


class PalmPlaneTests(unittest.TestCase):
    def test_pose_acceptance_cannot_ignore_unmet_transverse_shape(self):
        from tools.control.assess_stateful_arm import pose_task_met
        gates=dict(wristError=.005,elbowError=.005,palmErrorDeg=4)
        pose=dict(wristError=0.,elbowError=0.,palmErrorDeg=0.,palmBasisStateMet=False)
        self.assertFalse(pose_task_met(pose,gates))
        pose['palmBasisStateMet']=True
        self.assertTrue(pose_task_met(pose,gates))
        pose['projectionStateMet']=False
        self.assertFalse(pose_task_met(pose,gates))

    def plant(self):
        p=Plant();nodes=np.array(json.loads((APPROVED/'protocol.json').read_text())['newNodes'])
        return PatchProjectionPlant(p.model,nodes,rooted_upper=True)

    def test_basis_has_orthographic_singular_values_and_explicit_two_branches(self):
        for r,n in [(1.,1.),(.485,.409),(.8,.01),(.6,-.3)]:
            a=palm_basis(r,n,1);b=palm_basis(r,n,-1)
            self.assertEqual(a[0],b[0]);self.assertEqual(a[1],-b[1])
            matrix=np.array([[r,a[1]],[0,a[0]]])
            np.testing.assert_allclose(np.linalg.svd(matrix,compute_uv=False),[1,abs(n)],atol=1e-7)
            self.assertAlmostEqual(np.linalg.det(matrix),n)

    def test_inconsistent_or_ambiguous_inputs_rejected(self):
        for args in [(.5,.8,1),(0,0,1),(.5,.2,0),(.5,float('nan'),1)]:
            with self.assertRaises(ValueError):palm_basis(*args)
        for basis in [[1], [float('nan'),0]]:
            with self.assertRaises(ValueError):Target([0,0],1,palm_basis=basis)

    def test_limited_plane_cannot_stretch_to_chase_missing_side_view(self):
        p=self.plant();q=np.zeros(8);n=p.forward(q)
        c=PatchProjectionController(p.forward,lambda:p.raw_nodes,q,n,**speed_limits('fast'))
        z=np.array([0.,0.,0.,0.,0.,-40.,-15.,-80.])
        self.assertFalse(c.nonlinear_actuator_valid(z))
        bounded=c.bound_shape(z)
        self.assertTrue(c.nonlinear_actuator_valid(bounded))
        np.testing.assert_array_equal(bounded[:7],z[:7])
        r,w=1+bounded[5]/100,1+bounded[6]/100
        self.assertLessEqual(np.linalg.norm([[r,bounded[7]/100],[0,w]],2),1+1e-8)

    def test_unavailable_edge_view_retreats_transverse_basis_without_changing_route(self):
        p=self.plant();q=np.zeros(8);n=p.forward(q)
        c=PatchProjectionController(p.forward,lambda:p.raw_nodes,q,n,**speed_limits('fast'))
        t=Target(n[2],1.,elbow=n[1],projection=[.9,.9,.8],palm_basis=palm_basis(.8,.01,-1))
        original=t.palm_basis.copy();basis=c.basis_reference(t)
        self.assertGreaterEqual(.8*basis[0],.38-1e-8)
        self.assertLessEqual(np.linalg.norm([[.8,basis[1]],[0,basis[0]]],2),1+1e-8)
        np.testing.assert_array_equal(t.projection,[.9,.9,.8])
        np.testing.assert_array_equal(t.palm_basis,original)
        np.testing.assert_array_equal(t.wrist,n[2])

    def test_final_mesh_basis_readback_removes_pose_rotation(self):
        from tools.control.review_palm_plane import mesh_basis
        a=np.radians(57);rot=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
        matrix=rot@np.array([[.6,-.48],[0,.8]])
        frame=dict(wholeArms=dict(palmAffineReadback=matrix.ravel().tolist(),sourcePatchChain=[0,0,0,0,0,0,1,0]))
        np.testing.assert_allclose(mesh_basis(frame),[.6,.8,-.48],atol=1e-8)

    def test_coordinated_return_recovers_without_width_closing_before_shear(self):
        p=self.plant();q=np.array([-15.,35.,0.,-10.,-10.,-40.,-20.,48.]);n=p.forward(q)
        c=PatchProjectionController(p.forward,lambda:p.raw_nodes,q,n,**speed_limits('fast'))
        home=p.forward(np.zeros(8));t=Target(home[2],1.,elbow=home[1],palm_heading=heading(home),projection=[1,1,1],palm_basis=[1,0])
        for i in range(60):
            command,_=c.command(t,(i+1)/30)
            self.assertTrue(c.nonlinear_actuator_valid(command))
            row=c.observe(command,p.forward(command))
            self.assertLessEqual(row['metrics']['maxNodeSpeed'],.721)
            self.assertLessEqual(row['metrics']['maxParameterAcceleration'],1620.001)
        self.assertLess(row['wristError'],.005)
        self.assertTrue(row['palmBasisStateMet'])
        self.assertTrue(row['terminalStateMet'])

    def test_transverse_shape_does_not_move_approved_axis_nodes(self):
        p=self.plant();q=np.array([-15.,35.,0.,-10.,-10.,-45.])
        for shear in [-50,50]:
            np.testing.assert_allclose(p.forward(np.r_[q,-25,shear]),p.forward(q),atol=2e-7)
        self.assertAlmostEqual(pose(np.r_[q,-25,50])['projection']['palmWidth'],.75)
        self.assertAlmostEqual(pose(np.r_[q,-25,50])['projection']['palmShear'],.5)

    def test_eight_axis_fit_and_unreachable_basis_reported(self):
        p=self.plant();q=np.array([-15.,35.,0.,-10.,-10.,-40.,-20.,40.]);n=p.forward(q)
        c=PatchProjectionController(p.forward,lambda:p.raw_nodes,q,n,**speed_limits('fast'))
        t=Target(n[2],1.,elbow=n[1],palm_heading=heading(n),projection=[.9,.9,.6],palm_basis=[.8,.4],mode='track')
        goal,_,residual=c._fit_endpoint(t,p.forward,q)
        self.assertLess(residual,.01)
        c.command(t,1/30);row=c.observe(goal,p.forward(goal))
        self.assertTrue(row['palmBasisStateMet'])
        t.palm_basis=np.array([-.2,.4]);c.command(t,2/30)
        row=c.observe(goal,p.forward(goal))
        self.assertIn('palm-back-face-unsupported',row['reasons'])
        self.assertIn('palm-plane-limited',row['reasons'])

    def test_source_basis_added_without_changing_route_clock_or_body(self):
        from tools.control.mmd_timed_controller import make_stream
        from tests.test_mmd_timed_controller import MmdTimedInterfaceTests
        doc=MmdTimedInterfaceTests().document()
        for f in doc['sides']['anatomicalRight']['frames']:
            f['target']=f['target']|dict(palmProjectedRatio=.6,palmNormalZ=.3)
        home=np.array([[0.,0.],[0.,-1.],[0.,-2.],[0.,-2.2]])
        a=make_stream(doc,dict(native=home,patch=home),True)
        b=make_stream(doc,dict(native=home,patch=home),True,palm_branch=-1)
        for old,new in zip(a['frames'],b['frames']):
            basis=new['targets']['patch'].pop('palm_basis')
            self.assertEqual(old,new)
            if new['stage']=='mmd-source':np.testing.assert_allclose(basis,palm_basis(.6,.3,-1))

    def test_existing_fk_matches_all_source_samples_and_recovers_signed_shear(self):
        from pathlib import Path
        from tools.control.palm_plane import bases_from_reference
        source=json.loads(Path('E:/FuXuanLab/runs/20261112-mmd-timed-controller-01/source-map.json').read_text())
        reference=json.loads(Path('E:/FuXuanLab/runs/20261030-whole-performance-adaptation-01/references/mmd-hair.json').read_text(encoding='utf-8-sig'))
        bases=np.array(bases_from_reference(reference,source))
        self.assertEqual(bases.shape,(54,2))
        np.testing.assert_allclose(bases[0],[.8427833570064136,-.4707303136576863],atol=1e-10)
        self.assertLess(bases[:,1].min(),0);self.assertGreater(bases[:,1].max(),0)

    def test_mismatched_fk_reference_is_rejected_instead_of_silently_retimed(self):
        from pathlib import Path
        from tools.control.palm_plane import bases_from_reference
        source=json.loads(Path('E:/FuXuanLab/runs/20261112-mmd-timed-controller-01/source-map.json').read_text())
        reference=json.loads(Path('E:/FuXuanLab/runs/20261030-whole-performance-adaptation-01/references/mmd-hair.json').read_text(encoding='utf-8-sig'))
        reference['frames'][1]['\u53f3\u624b\u9996'][0]+=.01
        with self.assertRaisesRegex(ValueError,'does not match source sample 1'):bases_from_reference(reference,source)
