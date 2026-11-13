"""Offline expression-plan consumer. Rule baseline, not a learned dialogue model.

Only semantic events enter this interface. Scene registrations supply tested
motor affordances; unavailable pointing becomes gaze/language, never fake reach.
"""
import copy
import math
import numpy as np
from tools.control.body_session import BodyExecutionSession
from tools.control.gesture_exploration import POSES
from tools.control.face_control import NEUTRAL

KINDS=('listen','say','reference','think','decline','cancel')
AFFECTS=('neutral','warm','uncertain','firm')


def validate_plan(value,scene):
    required={'id','kind','text','target','affect','utteranceSeconds'}
    if not isinstance(value,dict) or set(value)!=required:raise ValueError('expression plan fields')
    if not isinstance(value['id'],str) or not value['id'] or len(value['id'])>80:raise ValueError('plan id')
    if value['kind'] not in KINDS or value['affect'] not in AFFECTS:raise ValueError('unknown expression capability')
    if not isinstance(value['text'],str) or len(value['text'])>500:raise ValueError('text length')
    if not isinstance(value['target'],str) or (value['target']!='user' and value['target'] not in scene):raise ValueError('unknown scene referent')
    seconds=value['utteranceSeconds']
    if isinstance(seconds,bool) or not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or not 0<=seconds<=20:raise ValueError('utterance clock')
    if value['kind']=='reference' and value['target']=='user':raise ValueError('reference requires scene object')
    return copy.deepcopy(value)


class DialogueBody:
    def __init__(self,scene):
        self.scene=copy.deepcopy(scene);self.session=BodyExecutionSession()
        self.plan=None;self.events=[];self.ids=set();self.last_tick=None
        self.fallback=None;self.idle_released=False

    def initialize(self,feedback):self.session.initialize(feedback)

    def submit(self,value,now):
        p=validate_plan(value,self.scene)
        if p['id'] in self.ids:raise ValueError('duplicate plan id')
        if self.session.feedback is None:raise ValueError('SDK actual feedback required')
        if not math.isfinite(now) or now<self.session.feedback['time']:raise ValueError('stale plan clock')
        # Fully validated before cancelling an existing plan.
        previous=self.plan['id'] if self.plan else None
        self.plan=p;self.ids.add(p['id']);self.accepted=now;self.end=now+p['utteranceSeconds']
        self.fallback=None;self.idle_released=False;self.gesture='neutral'
        self.active_sides=['native']
        obj=self.scene.get(p['target']);hb=np.zeros(6);self.gaze=[0.,0.]
        if obj is not None:self.gaze=list(obj['gaze'])
        if p['kind']=='think':hb=np.array([-6.,3.,-2.,0.,0.,0.]);self.gaze=[-.3,.25]
        elif p['kind']=='decline':hb=np.array([-5.,0.,-2.,0.,0.,0.])
        elif p['kind']=='listen':hb=np.array([0.,0.,2.,0.,0.,0.])
        elif p['affect']=='warm':hb=np.array([0.,-2.,1.,0.,0.,0.])
        arms=copy.deepcopy(self.session.homes)
        if p['kind']=='reference':
            if obj.get('pointing') is not None:
                registration=obj['pointing']
                selected=registration.get('arms',{'native':registration.get('nodes')})
                self.active_sides=list(selected)
                for side,goal in selected.items():arms[side]=copy.deepcopy(goal)
                hb=np.array(registration['headBody'])
                self.gesture=registration.get('gesture','index-emphasis')
            else:self.fallback='referent-outside-tested-pointing-capability'
        self.deadline=now+(1.4 if self.gesture!='neutral' else .9)
        if p['kind']=='reference' and obj.get('pointing') is not None:
            self.deadline=now+obj['pointing'].get('reachSeconds',1.4)
        if p['kind']=='cancel':
            self.session.release(p['id'],now,self.deadline);self.gaze=[0.,0.]
        else:self.session.submit_pose(p['id'],arms,hb,self.deadline,now)
        self.events.append(dict(time=now,event='plan-accepted',id=p['id'],supersedes=previous,
                                kind=p['kind'],target=p['target'],fallback=self.fallback))

    def _release(self,now,reason):
        self.session.release(self.plan['id']+'/'+reason,now,now+.9)
        self.deadline=now+.9
        self.gesture='neutral';self.gaze=[0.,0.];self.idle_released=True
        self.events.append(dict(time=now,event=reason,id=self.plan['id']))

    def next(self,now):
        if self.plan is None:raise ValueError('expression plan required')
        if not self.idle_released and now>=self.end+.45 and self.plan['kind'] not in ('listen','cancel'):
            self._release(now,'utterance-finished')
        packet=self.session.packet(now);p=self.plan
        speaking=bool(now<self.end and p['kind'] in ('say','reference','decline'))
        face=NEUTRAL.copy()
        if not self.idle_released:
            if p['affect']=='warm':face=[1.,1.,0.,.35,.18,.18]
            elif p['affect']=='uncertain':face=[.85,.85,0.,-.15,0.,0.]
            elif p['affect']=='firm':face=[.9,.9,0.,-.2,0.,0.]
        # Text-clock mouth cue only: explicitly not audio-aligned lip sync.
        if speaking:face[2]=.10+.12*(.5+.5*np.sin((now-self.accepted)*2*np.pi*2.3))
        packet.update(nativeFingerGoal=dict(values=POSES[self.gesture].copy(),deadline=self.deadline),
                      gazeGoal=dict(values=self.gaze.copy(),deadline=self.deadline),faceGoal=face,
                      intent=dict(planId=p['id'],kind=p['kind'],text=p['text'],target=p['target'],
                                  affect=p['affect'],speaking=speaking,gesture=self.gesture,
                                  fallback=self.fallback,released=self.idle_released,
                                  origin='offline semantic rule baseline; not model-generated'))
        return packet

    def observe(self,row,context):
        report=self.session.observe(row,context)
        reasons=[]
        for name in ('nativeFingers','gaze','face'):reasons.extend(row.get(name,{}).get('reasons',[]))
        report['expressionReasons']=reasons
        if (self.plan['kind']=='reference' and not self.idle_released and self.fallback is None and
                row['time']>self.deadline+.5 and
                (any(not row[s]['positionReached'] or not row[s]['terminalStateMet'] for s in self.active_sides) or
                 not row.get('nativeFingers',{'settled':True})['settled'] or
                 not row.get('gaze',{'settled':True})['settled'] or reasons)):
            self.fallback='actual-pointing-target-not-reached'
            self._release(row['time'],'execution-fallback')
        report['expressionFallback']=self.fallback
        return report
