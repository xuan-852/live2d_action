"""Upper node requests -> persistent body execution lifecycle -> actual feedback.

No IK or gesture generation here. Actuator state remains in the two controllers.
All times use the fixed 30Hz action clock, never wall time.
"""
import copy
import numpy as np
from tools.control.stateful_arm import Target, DT
from tools.control.timed_arm import heading

SIDES=('native','patch')
HEAD_BODY_IDS=('ParamAngleX','ParamAngleY','ParamAngleZ','ParamBodyAngleX','ParamBodyAngleY','ParamBodyAngleZ')
CONTEXT_LIMITS=np.array([25.,25.,20.,8.,8.,8.])


def nodes_goal(nodes):
    p=np.array(nodes,float)
    return dict(wrist=p[2].tolist(),elbow=p[1].tolist(),palm_heading=heading(p))


def hold_goal(goal,deadline):
    value=copy.deepcopy(goal)
    value.update(time=float(deadline),mode='hold',angular_velocity=0.)
    value.pop('velocity',None)
    Target(**value)
    return value


class BodyExecutionSession:
    def __init__(self):
        self.feedback=None;self.previous_feedback=None;self.active=None
        self.history=[];self.events=[];self.last_tick=None;self.last_packet=None

    def initialize(self,feedback):
        if self.feedback is not None:raise ValueError('Session already initialized')
        self.feedback=copy.deepcopy(feedback)
        self.capabilities=copy.deepcopy(feedback.get('capabilities',dict(coordinateFrame='model-local-xy',fps=30)))
        self.homes={s:nodes_goal(feedback['arms'][s]['actual']) for s in SIDES}
        for side in SIDES:
            if 'projection' in feedback['arms'][side]:self.homes[side]['projection']=copy.deepcopy(feedback['arms'][side]['projection'])
        self.home_context=np.array(feedback['headBody'],float)

    def describe(self):
        if self.feedback is None:raise ValueError('Session not initialized')
        return copy.deepcopy(self.capabilities)

    def _context(self,values):
        a=np.asarray(values,float)
        if a.shape!=(6,) or not np.isfinite(a).all() or np.any(abs(a)>CONTEXT_LIMITS):
            raise ValueError('headBody must be six finite values within the declared input range')
        return a.copy()

    def _accept(self,identifier,kind,now,deadline,head_body,reason='superseded'):
        if self.feedback is None:raise ValueError('Actual feedback required before accepting a request')
        if any(x['id']==identifier for x in self.history):raise ValueError('Duplicate request id')
        if not isinstance(identifier,str) or not identifier:raise ValueError('Request id required')
        start=float(self.feedback['time'])
        if not np.isfinite([now,deadline]).all() or now<start-1e-8 or deadline<=start:
            raise ValueError('Invalid action-clock deadline')
        if self.active is not None:
            self.active['history'].update(endedAt=float(now),endedReason=reason)
        record=dict(id=identifier,kind=kind,acceptedAt=float(now),observationTime=start,
                    deadline=float(deadline),sourceSamplesDelivered=0)
        self.history.append(record)
        context=np.array(self.feedback['headBody'],float)
        velocity=np.zeros(6)
        if self.previous_feedback is not None:
            dt=start-self.previous_feedback['time']
            if dt>0:velocity=(context-self.previous_feedback['headBody'])/dt
        self.active=dict(id=identifier,kind=kind,start=start,deadline=float(deadline),
            initialContext=context,contextVelocity=velocity,headBody=self._context(head_body),history=record)
        self.events.append(dict(time=float(now),requestId=identifier,event='accepted',kind=kind))

    def submit_pose(self,identifier,arms,head_body,deadline,now,kind='pose',reason='superseded'):
        if set(arms)!=set(SIDES):raise ValueError('Both arm identities required')
        for side in SIDES:
            value=copy.deepcopy(arms[side]);value.update(time=deadline,mode='hold')
            Target(**value)  # Reject moving terminal states on a hold request.
        self._context(head_body)
        self._accept(identifier,kind,now,deadline,head_body,reason)
        self.active['arms']=copy.deepcopy(arms)

    def submit_route(self,identifier,samples,entry_seconds,now,settle_seconds=.6):
        if not samples or entry_seconds<DT or settle_seconds<DT or abs(entry_seconds/DT-round(entry_seconds/DT))>1e-7:raise ValueError('Invalid route envelope')
        times=np.array([x['time'] for x in samples],float)
        if abs(times[0])>1e-8 or not np.isfinite(times).all() or not np.allclose(np.diff(times),DT,atol=1e-8):
            raise ValueError('Route requires contiguous original 30Hz times beginning at zero')
        for sample in samples:
            self._context(sample['headBody'])
            if set(sample['arms'])!=set(SIDES):raise ValueError('Both arm identities required')
            for s in SIDES:
                value=copy.deepcopy(sample['arms'][s]);value.update(time=now,mode='track');Target(**value)
        start=self.feedback['time'] if self.feedback is not None else now
        self._accept(identifier,'route',now,start+entry_seconds,samples[0]['headBody'])
        self.active.update(samples=copy.deepcopy(samples),routeStart=start+entry_seconds+DT,
            sourceEnd=start+entry_seconds+DT+times[-1],settleSeconds=settle_seconds)
        self.active['history'].update(sourceSamplesPlanned=len(samples),sourceTimesDelivered=[])

    def freeze(self,identifier,now,deadline):
        if self.feedback is None:raise ValueError('Actual feedback required')
        arms={s:nodes_goal(self.feedback['arms'][s]['actual']) for s in SIDES}
        for side in SIDES:
            if 'projection' in self.feedback['arms'][side]:arms[side]['projection']=copy.deepcopy(self.feedback['arms'][side]['projection'])
        self.submit_pose(identifier,arms,self.feedback['headBody'],deadline,now,kind='freeze',reason='interrupted')

    def release(self,identifier,now,deadline):
        self.submit_pose(identifier,self.homes,self.home_context,deadline,now,kind='release',reason='released')

    def _context_connection(self,now):
        p=self.active;duration=p['deadline']-p['start'];u=np.clip((now-p['start'])/duration,0,1)
        # Quintic boundary connection with inherited actual input velocity;
        # zero final velocity. Source samples themselves bypass this connector.
        delta=p['headBody']-p['initialContext'];v=p['contextVelocity']*duration
        raw=p['initialContext']+v*u+(10*delta-6*v)*u**3+(-15*delta+8*v)*u**4+(6*delta-3*v)*u**5
        bounded=np.clip(raw,-CONTEXT_LIMITS,CONTEXT_LIMITS)
        return bounded.tolist(),['head-body-connection-bound'] if np.max(abs(raw-bounded))>1e-8 else []

    def packet(self,now):
        if self.active is None:raise ValueError('No active body request')
        if self.last_tick is not None and abs(now-self.last_tick-DT)>1e-7:raise ValueError('Noncontiguous action clock')
        p=self.active;source_time=None;reasons=[]
        if p['kind']=='route' and now>=p['routeStart']-1e-8 and now<=p['sourceEnd']+1e-8:
            index=round((now-p['routeStart'])/DT);sample=p['samples'][index]
            targets=copy.deepcopy(sample['arms'])
            for side in SIDES:targets[side].update(time=float(now),mode='track')
            context=copy.deepcopy(sample['headBody']);source_time=sample['time'];phase='source'
            p['history']['sourceTimesDelivered'].append(source_time)
            p['history']['sourceSamplesDelivered']+=1
        elif p['kind']=='route' and now>p['sourceEnd']:
            targets={s:hold_goal(p['samples'][-1]['arms'][s],p['sourceEnd']+p['settleSeconds']) for s in SIDES}
            context=copy.deepcopy(p['samples'][-1]['headBody']);phase='hold-after-source'
        else:
            arms=p['samples'][0]['arms'] if p['kind']=='route' else p['arms']
            targets={s:hold_goal(arms[s],p['deadline']) for s in SIDES}
            context,reasons=self._context_connection(now)
            phase='enter' if p['kind']=='route' else p['kind'] if now<p['deadline'] else 'hold'
            if p['kind']=='release' and now>=p['deadline'] and self.last_packet is not None and self.last_packet['session']['requestId']==p['id'] and self.feedback.get('settled',False):phase='idle'
        for s in SIDES:Target(**targets[s])
        self.last_tick=float(now)
        self.last_packet=dict(stage=p['id']+'/'+phase,sourceTime=source_time,targets=targets,headBody=context,
            session=dict(requestId=p['id'],kind=p['kind'],phase=phase,acceptedAt=p['history']['acceptedAt'],
                observationTime=self.feedback['time'],requestReasons=reasons))
        return copy.deepcopy(self.last_packet)

    def observe(self,row,head_body_actual):
        if self.last_packet is None or abs(row['time']-self.last_tick)>1e-7:raise ValueError('Feedback/command pairing mismatch')
        arms={};reasons=['request:'+x for x in self.last_packet['session']['requestReasons']]
        for side in SIDES:
            s=row[side];old=np.array(self.feedback['arms'][side]['actual']);actual=np.array(s['actual'])
            velocity=(actual-old)/DT
            arms[side]=dict(actual=s['actual'],applied=s['applied'],velocity=velocity.tolist(),status=s['status'],
                residuals={k:s[k] for k in ('wristError','elbowError','palmErrorDeg')},reasons=s['reasons'],
                terminalStateMet=s['terminalStateMet'],positionReached=s['positionReached'])
            if 'projectionMeasured' in s:
                arms[side].update(projection=s['projectionMeasured'],projectionActuators=s['projectionApplied'],
                    projectionStateMet=s['projectionStateMet'],projectionResidual=s['projectionResidual'])
            reasons.extend(side+':'+x for x in s['reasons'])
        settled=all(x['positionReached'] and x['terminalStateMet'] and x.get('projectionStateMet',True) for x in arms.values())
        context_error=float(np.max(abs(np.array(head_body_actual)-self.last_packet['headBody'])))
        if context_error>.05:reasons.append('head-body-input-residual')
        settled=settled and context_error<=.05 and self.last_packet['session']['phase']!='source'
        status='failed' if any(x['status']=='failed' for x in arms.values()) else 'limited' if reasons else 'settled' if settled else 'executing'
        report=self.last_packet['session']|dict(time=row['time'],executionStatus=status,settled=settled,
            arms=arms,headBodyActual=list(head_body_actual),physicalContext=copy.deepcopy(row.get('physicalContext',{})),
            headBodyInputResidual=context_error,reasons=sorted(set(reasons)))
        self.previous_feedback=self.feedback
        self.feedback=dict(time=row['time'],arms=copy.deepcopy(arms),headBody=np.array(head_body_actual,float),settled=settled)
        return report
