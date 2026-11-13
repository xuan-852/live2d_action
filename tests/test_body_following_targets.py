import copy,unittest
import numpy as np
from tools.control.body_following_targets import BodyFollowingTargets
from tools.control.body_session import nodes_goal
from tools.control.stateful_arm import Target,DT,speed_limits
from tools.control.timed_arm import TimedArmController
from tests.test_timed_arm import linear


class PatchSource:
    def set_context(self,context):
        self.raw_nodes=np.array([[0,.3],[0,.2],[0,.1],[0,.06]])+[context['ParamBodyAngleX']*.01,0]


class BodyFollowingTests(unittest.TestCase):
    def test_fixed_scene_native_goal_is_not_rewritten_when_patch_rest_follows_body(self):
        adapter=BodyFollowingTargets.__new__(BodyFollowingTargets)
        adapter.applied={};adapter.patch=PatchSource()
        packet=dict(headBody=[0,0,0,4,0,0],targets=dict(native=dict(wrist=[.12,.5],elbow=[.2,.36],palm_heading=120.,time=2.,mode='hold'),
                    patch=dict(time=2.)),intent={})
        before=copy.deepcopy(packet['targets']['native'])
        result=adapter.realize(packet,dict(id='p',kind='reference',target='C'))
        self.assertEqual(result['targets']['native'],before)
        self.assertEqual(result['targets']['patch']['wrist'],[.04,.1])
        self.assertIn('body-following',result['intent']['armReferenceFrames']['patch'])

    def test_moving_material_frame_preserves_one_timed_connection(self):
        offset=np.zeros(2)
        def forward(q):return linear(q)+offset
        c=TimedArmController('native',forward,[-30,-30,-180],[30,30,180],np.zeros(3),forward(np.zeros(3)),**speed_limits('fast'))
        starts=[]
        for i in range(1,25):
            offset[:]=[i*.0001,0]
            q,_=c.command(Target(np.array([.1,.2])+offset,2.,elbow=np.array([.05,.1])+offset,
                                 palm_heading=0,connection_key='pose-1'),i*DT)
            c.observe(q,forward(q));starts.append(c.plan['start'])
        self.assertEqual(len(set(starts)),1)
        q,_=c.command(Target(np.array([.05,.2])+offset,3.,connection_key='pose-2'),25*DT)
        self.assertGreater(c.plan['start'],starts[0])

    def test_connection_identity_cannot_be_attached_to_a_world_route(self):
        with self.assertRaises(ValueError):Target([0,.2],1.,mode='track',connection_key='bad')
