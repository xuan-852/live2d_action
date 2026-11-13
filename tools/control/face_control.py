"""Bounded existing face inputs with SDK feedback; no audio/lip-sync inference."""
import numpy as np
from tools.control.finger_control import stop_cap

FACE_IDS=('ParamEyeLOpen','ParamEyeROpen','ParamMouthOpenY','ParamMouthForm','ParamEyeLSmile','ParamEyeRSmile')
NEUTRAL=[1.,1.,0.,0.,0.,0.]


class FaceController:
    lo=np.array([0.,0.,0.,-1.,-1.,-1.])
    hi=np.ones(6)

    def __init__(self,applied):
        self.q=self.vector(applied);self.velocity=np.zeros(6)

    @staticmethod
    def vector(value):
        q=np.array(value,float)
        if q.shape!=(6,) or not np.isfinite(q).all():raise ValueError('six finite face inputs required')
        return q

    def command(self,values,now):
        goal=self.vector(values)
        if np.any(goal<self.lo) or np.any(goal>self.hi):raise ValueError('face goal out of range')
        a=17.9/900;v=3/30
        low=np.maximum.reduce([np.full(6,-v),self.velocity/30-a,-stop_cap(self.q-self.lo,a,v)])
        high=np.minimum.reduce([np.full(6,v),self.velocity/30+a,stop_cap(self.hi-self.q,a,v)])
        self.conflict=bool(np.any(low>high+1e-6))
        if self.conflict:raise ValueError('face stopping constraints conflict')
        error=goal-self.q
        step=np.clip(np.sign(error)*stop_cap(abs(error),a,v),np.minimum(low,high),high)
        self.goal=goal;self.commanded=np.clip(self.q+step,self.lo,self.hi)
        return self.commanded.copy()

    def observe(self,applied):
        actual=self.vector(applied);velocity=(actual-self.q)*30
        acceleration=(velocity-self.velocity)*30
        reasons=[]
        if max(abs(velocity))>3.001 or max(abs(acceleration))>18.001:reasons.append('actual-face-continuity-violation')
        if max(abs(actual-self.commanded))>.001:reasons.append('actual-face-command-residual')
        settled=bool(max(abs(actual-self.goal))<.02 and max(abs(velocity))<.05)
        self.q=actual;self.velocity=velocity
        return dict(requested=self.goal.tolist(),applied=actual.tolist(),velocity=velocity.tolist(),
                    maxAcceleration=float(max(abs(acceleration))),reasons=reasons,settled=settled,
                    status='limited' if reasons else 'settled' if settled else 'executing')
