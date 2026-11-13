"""Restricted native basic-hand pose executor, in authored parameter units.

Five IDs are measured visible channels, not anatomical joint angles. A target is
a pose to reach by a deadline, not a finger trajectory. SDK applied values are
the state; command generation never advances that state on its own.
"""
import numpy as np

FINGER_IDS=('Param102','Param103','Param105','Param106','Param107')


def stop_cap(distance,acceleration_step,maximum_step):
    """Largest current step whose exact discrete braking fits remaining distance."""
    lo=np.zeros_like(np.asarray(distance,float));hi=np.full_like(lo,maximum_step)
    for _ in range(40):
        step=(lo+hi)/2;n=np.ceil(step/acceleration_step)
        travel=n*step-acceleration_step*n*(n-1)/2
        fits=travel<=np.maximum(distance,0)
        lo=np.where(fits,step,lo);hi=np.where(fits,hi,step)
    return lo


class FingerController:
    def __init__(self,applied,speed=90.,acceleration=540.,dt=1/30):
        self.q=self.vector(applied);self.velocity=np.zeros(5);self.dt=dt
        self.speed=speed;self.acceleration=acceleration;self.pending=None;self.last=None
        self.lo=np.full(5,-20.);self.hi=np.full(5,20.)

    @staticmethod
    def vector(value):
        q=np.asarray(value,float)
        if q.shape!=(5,) or not np.isfinite(q).all():raise ValueError('five finite native basic-finger values required')
        return q.copy()

    def command(self,values,deadline,now):
        goal=self.vector(values)
        if not np.isfinite([deadline,now]).all():raise ValueError('finite finger clocks required')
        reference=np.clip(goal,self.lo,self.hi);error=reference-self.q;dt=self.dt
        a=max(0.,self.acceleration-.1)*dt*dt;v=self.speed*dt
        lower=np.maximum.reduce([np.full(5,-v),self.velocity*dt-a,-stop_cap(self.q-self.lo,a,v)])
        upper=np.minimum.reduce([np.full(5,v),self.velocity*dt+a,stop_cap(self.hi-self.q,a,v)])
        desired=np.sign(error)*stop_cap(abs(error),a,v)
        # Float32 SDK readback can put a discrete stop face a few ULPs
        # outside its planned box. Reserve .1 acceleration units, and collapse
        # only sub-5e-6-unit roundoff gaps; actual budgets stay unchanged.
        feasible=bool(np.all(lower<=upper+5e-6))
        if feasible:lower=np.minimum(lower,upper)
        delta=np.clip(desired,lower,upper) if feasible else np.clip(-self.velocity*dt,-a,a)+self.velocity*dt
        command=np.clip(self.q+delta,self.lo,self.hi)
        self.pending=dict(values=goal,reference=reference,deadline=float(deadline))
        self.last=dict(command=command,now=float(now),feasible=feasible)
        return command.copy()

    def observe(self,applied):
        q=self.vector(applied);velocity=(q-self.q)/self.dt
        metrics=dict(maxStep=float(max(abs(q-self.q))),maxSpeed=float(max(abs(velocity))),
            maxAcceleration=float(max(abs(velocity-self.velocity))/self.dt),
            commandResidual=float(max(abs(q-self.last['command']))))
        residual=q-self.pending['values'];reasons=[]
        reached=bool(max(abs(residual))<=.15);settled=bool(reached and max(abs(velocity))<=.5)
        if np.any(self.pending['values']!=self.pending['reference']):reasons.append('finger-pose-range-limited')
        if self.last['now']>=self.pending['deadline'] and not settled:reasons.append('finger-deadline-missed')
        if not self.last['feasible']:reasons.append('finger-feedback-stop-conflict')
        if metrics['maxSpeed']>self.speed+.001 or metrics['maxAcceleration']>self.acceleration+.001:reasons.append('actual-finger-continuity-violation')
        if metrics['commandResidual']>.001:reasons.append('actual-finger-command-residual')
        status='failed' if any(r in reasons for r in ('finger-feedback-stop-conflict','actual-finger-continuity-violation')) else 'limited' if reasons else 'settled' if settled else 'executing'
        self.q=q;self.velocity=velocity
        return dict(ids=list(FINGER_IDS),requested=self.pending['values'].tolist(),feasibleReference=self.pending['reference'].tolist(),
            applied=q.tolist(),residual=residual.tolist(),velocity=velocity.tolist(),metrics=metrics,settled=settled,
            status=status,reasons=reasons,deadline=self.pending['deadline'],time=self.last['now'],
            scope='native basic-hand authored pose channels; no anatomical joint/finger-plane guarantee')
