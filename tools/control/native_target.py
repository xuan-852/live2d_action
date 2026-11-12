"""Measured five-axis native IK behind actual-state frame constraints."""
import json
from pathlib import Path
import numpy as np
from tools.whole.recovered.native import NativeArm
from tools.control.timed_arm import TimedArmController, task_error, task_jacobian, response_jacobian, heading
from tools.control.stateful_arm import constrained_step

NATIVE_KEYS = ['Param94','Param95','Param97','Param98','Param99']
NATIVE_ROUTE_LIMITS = dict(speed=2700.,acceleration=162000.,node_speed=3.,node_acceleration=180.,feedback_gain=.488)


class NativeTargetController(TimedArmController):
    @staticmethod
    def endpoint_diagnostics(target,nodes):
        residuals=dict(wrist=float(np.linalg.norm(nodes[2]-target.wrist)),
            elbow=None if target.elbow is None else float(np.linalg.norm(nodes[1]-target.elbow)),
            palmDegrees=None if target.palm_heading is None else float(abs((heading(nodes)-target.palm_heading+180)%360-180)))
        reasons=[name+'-target-limited' for name,key,limit in
            [('wrist','wrist',.005),('elbow','elbow',.005),('palm','palmDegrees',4.)]
            if residuals[key] is not None and residuals[key]>limit]
        return residuals,reasons

    def __init__(self, forward, model, anchors, q, nodes, **limits):
        calibration=json.loads((Path(__file__).resolve().parents[1]/'whole/recovered/calibration.json').read_text())['sides']['anatomicalLeft']
        self.measured=NativeArm(model,anchors,calibration)
        lo=[float(model.lo[model.idx[k]])+.01 for k in NATIVE_KEYS]
        hi=[float(model.hi[model.idx[k]])-.01 for k in NATIVE_KEYS]
        super().__init__('native',forward,lo,hi,q,nodes,**limits)
        self.reference_goal=None
        self.exact_native_stop=True

    def _fit_endpoint(self,target,predict,q):
        reference=self.reference_goal if self.reference_goal is not None else q
        seeds=[reference.copy(),q.copy()]
        if target.elbow is not None:
            # Use the actual shoulder context, never assume a 3-D body skeleton.
            root=predict(q)[0]; upper=target.elbow-root; fore=target.wrist-target.elbow
            angle=lambda v:float(np.degrees(np.arctan2(v[1],v[0])))
            record=dict(upperHeadingDeg=angle(upper),forearmHeadingDeg=angle(fore),
                upperProjectedRatio=float(np.linalg.norm(upper)/self.measured.lengths['upper']),
                forearmProjectedRatio=float(np.linalg.norm(fore)/self.measured.lengths['forearm']),
                bendUnwrappedDeg=float((angle(fore)-angle(upper)+180)%360-180))
            measured_seeds=[]
            for previous in (dict(zip(NATIVE_KEYS,reference)),None):
                u=self.measured._seed_upper(record,previous);f=self.measured._seed_bend(record,previous)
                measured_seeds.append(np.array([u[0],u[1],f[0],f[1],reference[4]]))
            seeds=measured_seeds+seeds
        seeds.append(np.zeros(5))
        solutions=[]
        for seed in seeds:
            z=np.clip(seed,self.lo,self.hi)
            for _ in range(40):
                nodes,raw=response_jacobian(predict,z,self.lo,self.hi)
                error=task_error(nodes,target);cost=float(error@error)
                if np.max(abs(error))<.001:break
                j=task_jacobian(nodes,raw,target)
                # Clipping an unconstrained Gauss-Newton step discards how the
                # other axes must compensate when one axis is on its boundary.
                # Solve the bounded local step itself, including a trust box.
                h=j.T@j+1e-5*np.eye(5);g=j.T@error
                a=np.vstack([np.eye(5),-np.eye(5)])
                b=np.r_[np.minimum(self.hi-z,10.),-np.maximum(self.lo-z,-10.)]
                step,feasible=constrained_step(h,g,a,b)
                if not feasible:break
                accepted=False
                for scale in (1.,.5,.25,.1,.01):
                    proposal=np.clip(z+scale*step,self.lo,self.hi)
                    e=task_error(predict(proposal),target)
                    if float(e@e)<cost-1e-10:
                        z=proposal;accepted=True;break
                if not accepted:break
            nodes=predict(z);cost=float(np.linalg.norm(task_error(nodes,target)))
            solutions.append((cost,z.copy(),nodes.copy()))
            if cost<.001:break
        best=min(x[0] for x in solutions)
        eligible=[x for x in solutions if x[0]<=best+min(.05,best*.05)]
        cost,z,nodes=min(eligible,key=lambda x:float(np.linalg.norm((x[1]-reference)/(self.hi-self.lo))))
        self.reference_goal=z.copy()
        return z,nodes,cost

    def _objective(self,target,now,q,actual,predict,jac,desired):
        self.finite_target_step=None
        if target.mode!='track':
            h,g=super()._objective(target,now,q,actual,predict,jac,desired)
            if target.mode=='hold' and now>=target.time:
                # Physics/body context can change after the original endpoint
                # fit. Correct using current actual nodes, without restarting
                # the goal clock or replacing the pre-deadline connector.
                goal,nodes,residual=self._fit_endpoint(target,predict,q)
                components,limited=self.endpoint_diagnostics(target,nodes)
                self._policy_info.update(referenceActuators=goal.tolist(),referenceNodes=nodes.tolist(),
                    endpointActuators=goal.tolist(),endpointResidual=residual,currentContextCorrection=True,
                    residualComponents=components,reasons=limited+(['endpoint-solution-limited'] if residual>1 else []))
                return h,h@(.5*(goal-q))
            return h,g
        goal,nodes,residual=self._fit_endpoint(target,predict,q)
        self.finite_target_step=goal-q
        components,limited=self.endpoint_diagnostics(target,nodes)
        self.plan=None
        self._policy_info=dict(kind='measured-five-axis-route-ik',mode='track',
            endpointActuators=goal.tolist(),referenceNodes=nodes.tolist(),endpointResidual=residual,
            residualComponents=components,reasons=limited+(['endpoint-solution-limited'] if residual>1 else []),deadlineControlsPlan=False)
        h=np.diag(1/(self.hi-self.lo)**2)
        return h,h@(goal-q)
