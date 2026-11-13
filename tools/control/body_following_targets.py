"""Body-relative rest/expansion from measured Core material nodes, not 3D bones.

Scene pointing keeps its original world/model-space target. Only neutral arms
and the registered bilateral expansion are body-following. No mesh is warped.
"""
import numpy as np
from tools.control.pilot import Plant
from tools.control.patch_projection import PatchProjectionPlant
from tools.control.native_target import NATIVE_KEYS
from tools.control.intent_expression_trial import POINT
from tools.control.body_session import HEAD_BODY_IDS,nodes_goal
from tools.control.stateful_arm import patch_forward


class BodyFollowingTargets:
    def __init__(self,approved_nodes):
        self.plant=Plant();self.patch=PatchProjectionPlant(self.plant.model,approved_nodes)
        self.applied=None

    def set_context(self,applied):self.applied=dict(applied)

    def realize(self,packet,plan,released=False):
        if self.applied is None:raise ValueError('actual SDK context required')
        context=self.applied|dict(zip(HEAD_BODY_IDS,packet['headBody']))
        expansion=plan['kind']=='reference' and plan['target']=='E' and not released
        rest=plan['kind']!='reference' or released
        frames={s:'model-local scene target' for s in ('native','patch')}
        for side in ('native','patch'):
            follow=expansion or rest or (side=='patch' and plan['kind']=='reference')
            if not follow:continue
            if side=='native':
                q=POINT if expansion else np.zeros(5)
                self.plant.model.set(context|dict(zip(NATIVE_KEYS,q)))
                a=self.plant.anchors.read();nodes=np.array([a[k] for k in ('shoulder','elbow','wrist','palm')])
            else:
                self.patch.set_context(context)
                nodes=patch_forward(self.patch.raw_nodes,[-38.,18.,-4.] if expansion else [0.,0.,0.])
            old=packet['targets'][side]
            packet['targets'][side]=nodes_goal(nodes)|dict(mode='hold',time=old['time'],angular_velocity=0.,
                connection_key=plan['id']+'/'+side+('/body-rest' if rest or not expansion else '/body-expansion'))
            frames[side]='body-following measured material chain'
        packet['intent']['armReferenceFrames']=frames
        return packet
