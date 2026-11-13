"""Two authored eye parameters with applied-state feedback, not a 3D sight ray."""
import numpy as np
from tools.control.finger_control import stop_cap

GAZE_IDS=('ParamEyeBallX','ParamEyeBallY')


class GazeController:
    def __init__(self,applied):
        self.q=self.vector(applied);self.velocity=np.zeros(2);self.dt=1/30
        self.speed=3.;self.acceleration=15.;self.last=None

    @staticmethod
    def vector(value):
        q=np.asarray(value,float)
        if q.shape!=(2,) or not np.isfinite(q).all():raise ValueError('two finite authored eye values required')
        return q.copy()

    def command(self,values,deadline,now):
        goal=self.vector(values)
        if not np.isfinite([deadline,now]).all():raise ValueError('finite gaze clocks required')
        reference=np.clip(goal,-.8,.8);a=(self.acceleration-.01)*self.dt**2;v=self.speed*self.dt
        lo=np.maximum.reduce([np.full(2,-v),self.velocity*self.dt-a,-stop_cap(self.q+.8,a,v)])
        hi=np.minimum.reduce([np.full(2,v),self.velocity*self.dt+a,stop_cap(.8-self.q,a,v)])
        feasible=bool(np.all(lo<=hi+5e-7));lo=np.minimum(lo,hi) if feasible else lo
        error=reference-self.q;desired=np.sign(error)*stop_cap(abs(error),a,v)
        step=np.clip(desired,lo,hi) if feasible else self.velocity*self.dt-np.sign(self.velocity)*a
        command=np.clip(self.q+step,-.8,.8)
        self.last=dict(goal=goal,reference=reference,command=command,deadline=float(deadline),now=float(now),feasible=feasible)
        return command.copy()

    def observe(self,applied):
        q=self.vector(applied);velocity=(q-self.q)/self.dt;residual=q-self.last['goal']
        metrics=dict(maxSpeed=float(max(abs(velocity))),maxAcceleration=float(max(abs(velocity-self.velocity))/self.dt),
            commandResidual=float(max(abs(q-self.last['command']))))
        settled=bool(max(abs(residual))<=.02 and max(abs(velocity))<=.05)
        reasons=[]
        if np.any(self.last['goal']!=self.last['reference']):reasons.append('gaze-range-limited')
        if self.last['now']>=self.last['deadline'] and not settled:reasons.append('gaze-deadline-missed')
        if not self.last['feasible']:reasons.append('gaze-stop-conflict')
        if metrics['maxSpeed']>3.001 or metrics['maxAcceleration']>15.001:reasons.append('actual-gaze-continuity-violation')
        if metrics['commandResidual']>.001:reasons.append('actual-gaze-command-residual')
        status='failed' if any(x in reasons for x in ('gaze-stop-conflict','actual-gaze-continuity-violation')) else 'limited' if reasons else 'settled' if settled else 'executing'
        self.q=q;self.velocity=velocity
        return dict(ids=list(GAZE_IDS),requested=self.last['goal'].tolist(),reference=self.last['reference'].tolist(),applied=q.tolist(),
            residual=residual.tolist(),velocity=velocity.tolist(),metrics=metrics,status=status,reasons=reasons,settled=settled,
            scope='authored eye input and final mesh markers; no 3D gaze-ray guarantee')
