"""Feedback-driven task director above dialogue plans. Explicit rule baseline.

Tasks contain purposes and scene identities, never actuator values or a timed
motion schedule. Progress is earned by actual SDK execution, not elapsed playback.
"""
import copy
import math
import numpy as np
from tools.control.dialogue_body import DialogueBody


def axis_error(markers,target):
    base=np.asarray(markers['indexBase']);axis=np.asarray(markers['indexTip'])-base
    toward=np.asarray(target)-base
    denominator=np.linalg.norm(axis)*np.linalg.norm(toward)
    if denominator<1e-10:return 180.
    return float(np.degrees(np.arccos(np.clip(axis@toward/denominator,-1,1))))


def validate_task(task,scene):
    if not isinstance(task,dict) or set(task)!={'id','purpose','targets','request','confidence'}:raise ValueError('task fields')
    if not isinstance(task['id'],str) or not task['id'] or len(task['id'])>60:raise ValueError('task id')
    if task['purpose'] not in ('inspect','compare','emphasize','wait','cancel'):raise ValueError('task purpose')
    targets=task['targets']
    if not isinstance(targets,list) or len(targets)>4 or any(not isinstance(x,str) or x not in scene for x in targets):raise ValueError('task scene identities')
    minimum=2 if task['purpose']=='compare' else 1 if task['purpose'] in ('inspect','emphasize') else 0
    if len(targets)<minimum or (task['purpose'] in ('cancel','wait') and targets):raise ValueError('task target cardinality')
    if not isinstance(task['request'],str) or len(task['request'])>500:raise ValueError('task text')
    c=task['confidence']
    if isinstance(c,bool) or not isinstance(c,(int,float)) or not math.isfinite(c) or not 0<=c<=1:raise ValueError('task confidence')
    return copy.deepcopy(task)


class TaskExpressionDirector:
    def __init__(self,scene):
        self.scene=copy.deepcopy(scene);self.body=DialogueBody(scene)
        self.task=None;self.seen=set();self.events=[];self.outcomes=[]
        self.index=0;self.stable=0;self.phase='waiting';self.pending=None
        self.confidence=1.;self.failed=[];self.current_outcome=None

    def initialize(self,feedback):self.body.initialize(feedback)

    def submit(self,task,now):
        t=validate_task(task,self.scene)
        if t['id'] in self.seen:raise ValueError('duplicate task id')
        if self.body.session.feedback is None:raise ValueError('actual feedback required')
        if not math.isfinite(now) or now<self.body.session.feedback['time']:raise ValueError('stale task clock')
        if self.task is not None and self.phase not in ('complete','limited','cancelled','waiting'):
            if self.current_outcome is not None and self.current_outcome['result']=='executing':self.current_outcome.update(endedAt=now,result='interrupted')
            self.events.append(dict(time=now,event='task-interrupted',taskId=self.task['id'],by=t['id']))
        self.task=t;self.seen.add(t['id']);self.index=0;self.stable=0;self.failed=[];self.pending=None
        self.confidence=t['confidence'];self.current_outcome=None
        self.events.append(dict(time=now,event='task-accepted',task=copy.deepcopy(t)))
        if t['purpose'] in ('wait','cancel'):
            self.phase='waiting' if t['purpose']=='wait' else 'cancelled'
            self._neutral(now,t['purpose'])
        else:self._start_goal(now)

    def _neutral(self,now,reason):
        text='当前目标受限，先收回；保留失败，不宣称完成。' if self.phase=='limited' else self.task['request']
        self.body.submit(dict(id=self.task['id']+'/'+reason,kind='cancel',text=text,
                              target='user',affect='neutral',utteranceSeconds=0.),now)

    def _start_goal(self,now):
        target=self.task['targets'][self.index];self.phase='seeking';self.stable=0
        self.current_outcome=dict(taskId=self.task['id'],target=target,index=self.index,requestedAt=now,
                                  result='executing',initialObservationTime=self.body.session.feedback['time'])
        self.outcomes.append(self.current_outcome)
        affect='uncertain' if self.confidence<.4 else 'warm' if self.confidence>.7 else 'neutral'
        prefix='对照这一侧' if self.task['purpose']=='compare' else '查看这个位置'
        if self.failed:prefix='前一目标受限，改看这里'
        self.body.submit(dict(id=f"{self.task['id']}/goal-{self.index}",kind='reference',
                              text=f"{prefix}：{self.scene[target]['label']}。",target=target,
                              affect=affect,utteranceSeconds=8.),now)
        self.events.append(dict(time=now,event='goal-selected',taskId=self.task['id'],target=target,index=self.index))

    def next(self,now):
        if self.task is None:raise ValueError('task required')
        if self.pending is not None:
            self.pending=None;self.index+=1
            if self.index<len(self.task['targets']):self._start_goal(now)
            else:
                self.phase='limited' if self.failed else 'complete'
                self.events.append(dict(time=now,event='task-finished',taskId=self.task['id'],result=self.phase,failedTargets=self.failed.copy()))
                self._neutral(now,'finished')
        packet=self.body.next(now)
        packet['intent']['task']=dict(id=self.task['id'],purpose=self.task['purpose'],request=self.task['request'],
                                     targetIndex=self.index,phase=self.phase,stableFrames=self.stable,
                                     confidence=self.confidence,failedTargets=self.failed.copy())
        return packet

    def observe(self,row,context):
        report=self.body.observe(row,context)
        measured=None
        if self.phase in ('seeking','holding'):
            target=self.task['targets'][self.index];obj=self.scene[target]
            fallback=report['expressionFallback']
            pointing=obj.get('pointing');sides=list(pointing.get('arms',{'native':pointing.get('nodes')})) if pointing else []
            ready=bool(sides) and all(row[s]['positionReached'] and row[s]['terminalStateMet'] for s in sides)
            ready=ready and row['nativeFingers']['settled'] and row['gaze']['settled']
            ready=ready and not any(row[s]['status']=='failed' for s in sides)
            ready=ready and row['geometry'].get('minTriangleAreaRatio',1.)>=.5 and row['geometry'].get('restoredMeshError',0.)<=1e-6
            ready=ready and row['geometry'].get('flippedTriangles',0)==0
            if obj['criterion']=='native-index-material-axis':
                measured=axis_error(row['taskMarkers'],obj['xy']);ready=ready and measured<=6.
            if not fallback and row['time']>self.body.deadline+.6 and not ready:
                fallback='actual-task-constraint-not-met'
                self.body.fallback=fallback;self.body._release(row['time'],'task-constraint-fallback')
            if fallback:
                self.failed.append(target);self.confidence=max(0.,self.confidence-.35)
                self.current_outcome.update(endedAt=row['time'],result='limited',reason=fallback,axisErrorDegrees=measured)
                self.events.append(dict(time=row['time'],event='actual-feedback-fallback',taskId=self.task['id'],target=target,reason=fallback))
                self.pending='advance';self.phase='recovering';self.stable=0
            else:
                self.stable=self.stable+1 if ready else 0
                self.phase='holding' if ready else 'seeking'
                if self.stable>=15:
                    self.current_outcome.update(endedAt=row['time'],result='execution-goal-met',axisErrorDegrees=measured,stableFrames=self.stable)
                    self.events.append(dict(time=row['time'],event='actual-goal-held',taskId=self.task['id'],target=target,frames=self.stable))
                    self.pending='advance'
        report['task']=dict(id=self.task['id'],phase=self.phase,targetIndex=self.index,stableFrames=self.stable,
                            confidence=self.confidence,failedTargets=self.failed.copy(),axisErrorDegrees=measured)
        return report
