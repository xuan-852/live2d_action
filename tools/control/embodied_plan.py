"""Strict model-output boundary for language, scene goals and head/torso axes."""
import copy
import math
import numpy as np
from tools.control.dialogue_body import DialogueBody,validate_plan
from tools.control.task_expression import axis_error

POSTURE_KEYS=('headTurn','headNod','headTilt','bodyTurn','bodyLean','bodyTilt')
POSTURE_LIMITS=np.array([20.,15.,12.,6.,4.,5.])
BASE=('id','kind','text','target','affect','utteranceSeconds')


def validate_model_plan(plan,scene):
    if not isinstance(plan,dict) or set(plan)!=set(BASE)|{'posture','reachSeconds','holdSeconds','finish','explanation'}:raise ValueError('model plan fields')
    validate_plan({k:plan[k] for k in BASE},scene)
    if not isinstance(plan['posture'],dict) or set(plan['posture'])!=set(POSTURE_KEYS):raise ValueError('posture axes')
    values=list(plan['posture'].values())
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) for v in values):raise ValueError('posture numbers')
    posture=np.array([plan['posture'][k] for k in POSTURE_KEYS],float)
    if not np.isfinite(posture).all() or np.any(abs(posture)>POSTURE_LIMITS):raise ValueError('posture range')
    for key,lo,hi in [('reachSeconds',.3,5.),('holdSeconds',.3,2.)]:
        v=plan[key]
        if isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) or not lo<=v<=hi:raise ValueError(key)
    if type(plan['finish']) is not bool:raise ValueError('finish flag')
    if not isinstance(plan['explanation'],str) or len(plan['explanation'])>500:raise ValueError('explanation')
    if plan['kind']=='cancel' and max(abs(posture))>1e-8:raise ValueError('cancel posture must release to neutral')
    if plan['finish'] and plan['kind'] not in ('cancel','listen'):raise ValueError('finish requires safe idle intent')
    return copy.deepcopy(plan)


class EmbodiedExecution:
    def __init__(self,scene):
        self.body=DialogueBody(scene,auto_release=False);self.scene=scene;self.plan=None;self.stable=0;self.last_report=None;self.previous_context=None

    def initialize(self,feedback):self.body.initialize(feedback)

    def submit(self,plan,now):
        plan=validate_model_plan(plan,self.scene)
        posture=[plan['posture'][k] for k in POSTURE_KEYS]
        self.body.submit({k:plan[k] for k in BASE},now,head_body=posture,reach_seconds=plan['reachSeconds'])
        self.plan=plan;self.stable=0;self.accepted=now

    def next(self,now):
        packet=self.body.next(now)
        packet['intent'].update(origin='current conversation model, tool-mediated online decisions; not a prerecorded rule output',
                                upperPlan=copy.deepcopy(self.plan))
        return packet

    def observe(self,row,context):
        report=self.body.observe(row,context);plan=self.plan;angle=None
        if plan['kind']=='reference':
            obj=self.scene[plan['target']]
            ready=all(row[s]['positionReached'] and row[s]['terminalStateMet'] for s in self.body.active_sides)
            if obj['criterion']=='native-index-material-axis':
                angle=axis_error(row['taskMarkers'],obj['xy']);ready=ready and angle<=6.
        else:ready=report['settled']
        velocity=0. if self.previous_context is None else max(abs(np.array(context)-self.previous_context))*30
        self.previous_context=np.array(context)
        ready=ready and report['headBodyInputResidual']<=.05 and velocity<.5
        ready=ready and row['nativeFingers']['settled'] and row['gaze']['settled']
        ready=ready and all(row[s]['status']!='failed' for s in ('native','patch'))
        ready=ready and row['geometry']['minTriangleAreaRatio']>=.5 and row['geometry']['flippedTriangles']==0
        self.stable=self.stable+1 if ready else 0
        reason=report['expressionFallback']
        if not reason and row['time']>self.body.deadline+.7 and not ready:reason='model-plan-constraints-not-met'
        held=self.stable>=round(plan['holdSeconds']*30)
        report['upperExecution']=dict(planId=plan['id'],held=held,stableFrames=self.stable,axisErrorDegrees=angle,
            postureActual=list(context),postureMaxSpeed=float(velocity),constraintFailure=reason,
            readyForNext=bool(reason or (held and row['time']>=self.accepted+plan['utteranceSeconds'])),
            elapsed=row['time']-self.accepted)
        self.last_report=copy.deepcopy(report)
        return report
