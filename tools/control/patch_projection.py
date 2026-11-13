"""Frontal length control using the existing approved patch mesh deformation.

The forward model mirrors ShoulderConnection on its bound material triangles;
SDK final mesh/area readback remains authoritative. No new topology or wrist.
"""
import ctypes as C
import numpy as np
from tools.control.stateful_arm import constrained_step
from tools.control.timed_arm import TimedArmController,task_error,task_jacobian,response_jacobian,heading

PATCH_KEYS=['shoulder','elbow','wrist','upperLengthPercent','forearmLengthPercent','palmLengthPercent']
PALM_PLANE_KEYS=PATCH_KEYS+['palmWidthPercent','palmShearPercent']
RATIO_TOL=.03
PALM_DIRECTION_WEIGHT=3.


def projection_guard(raw,unprojected_angular,expected_floor):
    return np.minimum.reduce([np.asarray(raw)/np.maximum(expected_floor,1e-6),
        np.asarray(unprojected_angular),np.asarray(raw)*(.55/.30)])


def projected_shape_guard(raw,projection_only,angular_at_projection,expected_floor,raw_floor=.30):
    return np.minimum.reduce([np.asarray(angular_at_projection),np.asarray(projection_only)/np.maximum(expected_floor,1e-6),
        np.asarray(raw)*(.55/np.asarray(raw_floor))])


def rotate(v,angle):
    a=np.radians(angle);c,s=np.cos(a),np.sin(a)
    return np.asarray(v)@np.array([[c,s],[-s,c]])


def rigid_mls(vertices,source,destination):
    result=[]
    for v in vertices:
        d=np.sum((source-v)**2,axis=1)
        if np.min(d)<1e-12:
            result.append(destination[np.argmin(d)]);continue
        w=1/d;w/=sum(w);p=w@source;q=w@destination
        a=source-p;b=destination-q
        real=np.sum(w*np.sum(a*b,axis=1));imag=np.sum(w*(a[:,0]*b[:,1]-a[:,1]*b[:,0]))
        norm=np.hypot(real,imag)
        if norm<1e-12:raise ValueError('Indeterminate patch attachment')
        delta=v-p
        result.append(q+np.array([real*delta[0]-imag*delta[1],imag*delta[0]+real*delta[1]])/norm)
    return np.array(result)


def bind(vertices,triangles,point):
    # Same max-min barycentric material binding as the SDK wrapper, at exactly
    # the approved four nodes. This is not a fresh pivot estimate.
    vertices=np.asarray(vertices,np.float32);point=np.asarray(point,np.float32)
    best=None;quality=-np.inf
    cross=lambda a,b:a[0]*b[1]-a[1]*b[0]
    for ids in triangles:
        x=vertices[ids[0]];u=vertices[ids[1]]-x;v=vertices[ids[2]]-x;p=point-x;area=cross(u,v)
        if abs(area)<1e-10:continue
        wb=cross(p,v)/area;wc=cross(u,p)/area;wa=np.float32(1)-wb-wc
        score=min(wa,wb,wc)
        if score>quality:quality=score;best=(ids.copy(),np.array([wa,wb,wc],float))
    if best is None or max(abs(best[1]))>4:raise ValueError('Unsafe material binding')
    return best


class PatchProjectionPlant:
    def __init__(self,model,approved_nodes,rooted_upper=False):
        self.rooted_upper=rooted_upper
        self.model=model;model.set({});self.rest={}
        counts=np.ctypeslib.as_array(model.api('GetDrawableIndexCounts',C.POINTER(C.c_int),[C.c_void_p])(model.ptr),shape=(model.n,))
        pointers=model.api('GetDrawableIndices',C.POINTER(C.POINTER(C.c_ushort)),[C.c_void_p])(model.ptr)
        self.bindings=[]
        for mesh,point in zip(('ArtMesh251','ArtMesh261','ArtMesh253','ArtMesh253'),approved_nodes):
            i=model.names.index(mesh);v=model.mesh[i].copy();self.rest[mesh]=v.copy()
            triangles=np.ctypeslib.as_array(pointers[i],shape=(int(counts[i]),)).reshape(-1,3)
            self.bindings.append((mesh,*bind(v,triangles,point)))
        self.body_indices=np.argsort(-self.rest['ArtMesh251'][:,0],kind='stable')[:2]
        self.context=None;self.set_context({})

    def set_context(self,context):
        if self.context==context:return
        self.context=dict(context);self.model.set(context)
        self.source={k:self.model.mesh[self.model.names.index(k)].copy().astype(float) for k in self.rest}
        self.raw_nodes=np.array([weights@self.source[mesh][ids] for mesh,ids,weights in self.bindings])
        self.controls=np.vstack([self.source['ArtMesh251'][self.body_indices],self.raw_nodes[:2]])

    def forward(self,q):
        q=np.asarray(q,float)
        shoulder,elbow,wrist=q[:3];upper,fore,palm=1+np.asarray(q[3:6])/100
        width,shear=(1+q[6]/100,q[7]/100) if len(q)==8 else (1.,0.)
        socket,elbow_base,wrist_base,palm_base=self.raw_nodes
        ep=socket+rotate((elbow_base-socket)*upper,shoulder)
        wp=ep+rotate((wrist_base-elbow_base)*fore,shoulder+elbow)
        pa=(palm_base-wrist_base);pa/=np.linalg.norm(pa)
        fa=(wrist_base-elbow_base);fa/=np.linalg.norm(fa)
        targets=self.controls.copy();targets[3]=ep
        out=[]
        identity=np.max(abs(q))<1e-12
        for mesh,ids,weights in self.bindings:
            v=self.source[mesh][ids]
            if identity:deformed=v
            elif mesh=='ArtMesh253':
                delta=v-wrist_base;side=np.array([-pa[1],pa[0]])
                side=np.array([-pa[1],pa[0]])
                projected=delta+np.outer(delta@pa,pa)*(palm-1)+np.outer(delta@side,pa)*shear+np.outer(delta@side,side)*(width-1)+np.outer(delta@side,pa)*shear+np.outer(delta@side,side)*(width-1)
                deformed=wp+rotate(projected,shoulder+elbow+wrist)
            elif mesh=='ArtMesh261':
                delta=v-wrist_base;scaled=delta+np.outer(delta@fa,fa)*(fore-1)
                # Garment cuff keeps its longitudinal/wrist deformation.
                # Transverse palm projection belongs to the bare hand only.
                projected=delta+np.outer(delta@pa,pa)*(palm-1)
                t=np.clip((.09-self.rest[mesh][ids,1])/.08,0,1);t=t*t*(3-2*t)
                local=scaled*(1-t[:,None])+projected*t[:,None]
                deformed=np.array([wp+rotate(x,shoulder+elbow+blend*wrist) for x,blend in zip(local,t)])
            elif self.rooted_upper:
                # Existing SDK-wrapper rootedUpper mode, same approved socket.
                # Source/rest weights mirror SandboxArmGeometry.RootWeight.
                root=np.array([-.105,.310]);axis=np.array([-.145,.205])-root;length=np.linalg.norm(axis);axis/=length
                t=(self.rest[mesh][ids]-root)@axis/length;weight=np.clip((t+.6)/1.6,0,1)
                attached=rigid_mls(v,self.controls,self.controls)
                upper_axis=elbow_base-socket;upper_axis/=np.linalg.norm(upper_axis)
                local=attached-socket;local+=np.outer(local@upper_axis,upper_axis)*(upper-1)*weight[:,None]
                deformed=np.array([socket+rotate(x,shoulder*w) for x,w in zip(local,weight)])
            else:deformed=rigid_mls(v,self.controls,targets)
            out.append(weights@deformed)
        return np.array(out)


class PatchProjectionController(TimedArmController):
    def __init__(self,forward,source_nodes,q,nodes,**limits):
        self.palm_plane=len(q)==8
        self.coordinated_finite_step=self.palm_plane
        super().__init__('patch',forward,[-56,0,-100,-25,-45,-60]+([-65,-95] if self.palm_plane else []),[56,145,30,0,0,0]+([0,95] if self.palm_plane else []),q,nodes,**limits)
        self.source_nodes=source_nodes;self.actual_source_nodes=None;self.reference_goal=None
        self.actuator_acceleration_reserve=.1
        self.supports_projection=True
        self.recover_mesh_margin=True
        # Coupled transverse projection has stronger nonlinear/context sensitivity.
        # Reserve more area; actual .50/.28 failure thresholds remain unchanged.
        self.planned_mesh_floor=.55

    def nonlinear_actuator_valid(self,q):
        if not self.palm_plane:return True
        r,w=1+q[5]/100,1+q[6]/100;s=q[7]/100
        return s*s <= max(0.,(1-r*r)*(1-w*w))+1e-6

    def bound_shape(self,q):
        z=q.copy()
        if self.palm_plane:
            r,w=1+z[5]/100,1+z[6]/100
            bound=np.sqrt(max(0.,(1-r*r)*(1-w*w)))*100
            z[7]=np.clip(z[7],-bound,bound)
        return z

    def basis_reference(self,target):
        basis=np.array([1.,0.]) if target.palm_basis is None else target.palm_basis
        # Preserve the upper route/longitudinal request. If the source wants a
        # near-edge single-face view, retreat the transverse appearance first;
        # do not shorten the whole hand or block the arm to chase it.
        r=float(np.clip(self.desired_ratios(target)[2],.4,1.))
        width=float(np.clip(max(basis[0],.38/r),.35,1.))
        bound=np.sqrt(max(0.,(1-r*r)*(1-width*width)))
        shear=float(np.clip(basis[1],-bound,bound))
        return np.array([width,shear])

    def basis_goal(self,target):
        return (self.basis_reference(target)-np.array([1.,0.]))*100

    def full_error(self,nodes,target,q):
        error=self._error(nodes,target)
        return np.r_[error,(self.basis_goal(target)-q[6:])/3.] if self.palm_plane else error

    def full_jacobian(self,j):
        return np.vstack([j,np.eye(self.dof)[6:]/3.]) if self.palm_plane else j

    def ratios(self,nodes,source=None):
        source=self.source_nodes() if source is None else source
        return np.linalg.norm(np.diff(nodes,axis=0),axis=1)/np.maximum(np.linalg.norm(np.diff(source,axis=0),axis=1),1e-8)

    def desired_ratios(self,target):
        if target.projection is not None:return target.projection
        source=self.source_nodes();base=np.linalg.norm(np.diff(source,axis=0),axis=1)
        upper=np.linalg.norm(target.elbow-source[0])/base[0] if target.elbow is not None else 1.
        fore=np.linalg.norm(target.wrist-target.elbow)/base[1] if target.elbow is not None else 1.
        return np.array([upper,fore,1.])

    def _error(self,nodes,target):
        error=task_error(nodes,target)
        if target.palm_heading is not None:error[-1]*=PALM_DIRECTION_WEIGHT
        return np.r_[error,(self.desired_ratios(target)-self.ratios(nodes))/RATIO_TOL]

    def task_jacobian(self,nodes,raw,target):
        j=task_jacobian(nodes,raw,target)
        if target.palm_heading is not None:j[-1]*=PALM_DIRECTION_WEIGHT
        return j

    def ratio_jacobian(self,nodes,node_jacobian):
        rows=[];source=self.source_nodes();lengths=np.linalg.norm(np.diff(source,axis=0),axis=1)
        for i in range(3):
            d=nodes[i+1]-nodes[i]
            delta=node_jacobian[(i+1)*2:(i+2)*2]-node_jacobian[i*2:(i+1)*2]
            rows.append(d@delta/max(np.linalg.norm(d)*lengths[i],1e-10)/RATIO_TOL)
        return np.array(rows)

    def _fit_endpoint(self,target,predict,q):
        reference=q if self.reference_goal is None else self.reference_goal
        source=self.source_nodes();ratios=self.desired_ratios(target)
        seed=np.zeros(self.dof);seed[3:6]=(ratios-1)*100
        if self.palm_plane:seed[6:]=self.basis_goal(target)
        if target.elbow is not None:
            angle=lambda v:float(np.degrees(np.arctan2(v[1],v[0])))
            up=angle(target.elbow-source[0]);fore=angle(target.wrist-target.elbow)
            wrap=lambda x:(x+180)%360-180
            seed[:3]=[wrap(up-angle(source[1]-source[0])),wrap(fore-up-angle(source[2]-source[1])+angle(source[1]-source[0])),
                wrap((target.palm_heading if target.palm_heading is not None else heading(source))-fore-heading(source)+angle(source[2]-source[1]))]
        candidates=[]
        for seed in (reference,seed,q,np.zeros(self.dof)):
            z=self.bound_shape(np.clip(seed,self.lo,self.hi))
            for _ in range(30):
                nodes,raw=response_jacobian(predict,z,self.lo,self.hi);error=self.full_error(nodes,target,z)
                if max(abs(error))<.001:break
                j=self.task_jacobian(nodes,raw,target)
                j=self.full_jacobian(np.vstack([j,self.ratio_jacobian(nodes,raw)]));h=j.T@j+1e-5*np.eye(self.dof);g=j.T@error
                step,ok=constrained_step(h,g,np.vstack([np.eye(self.dof),-np.eye(self.dof)]),
                    np.r_[np.minimum(self.hi-z,10),-np.maximum(self.lo-z,-10)])
                if not ok:break
                improved=False
                for scale in (1.,.5,.25,.1):
                    proposal=self.bound_shape(np.clip(z+scale*step,self.lo,self.hi))
                    if np.linalg.norm(self.full_error(predict(proposal),target,proposal))<np.linalg.norm(error)-1e-9:
                        z=proposal;improved=True;break
                if not improved:break
            nodes=predict(z);cost=float(np.linalg.norm(self.full_error(nodes,target,z)));candidates.append((cost,z.copy(),nodes.copy()))
            if cost<.001:break
        cost,goal,nodes=min(candidates,key=lambda x:x[0]);self.reference_goal=goal.copy()
        return goal,nodes,cost

    def _objective(self,target,now,q,actual,predict,jac,desired):
        self.finite_target_step=None
        self.task_linear_constraints=None
        ratios=self.desired_ratios(target);source=self.source_nodes()
        if target.palm_heading is not None:
            a=np.radians(target.palm_heading)
            desired[3]=target.wrist+np.linalg.norm(source[3]-source[2])*ratios[2]*np.array([np.cos(a),np.sin(a)])
        if self.palm_plane or target.mode=='track' or (target.mode=='hold' and now>=target.time):
            goal,nodes,residual=self._fit_endpoint(target,predict,q)
            self.finite_target_step=goal-q if target.mode=='track' or self.palm_plane else None
            self._policy_info=dict(kind='mesh-consistent-frontal-projection',referenceNodes=nodes.tolist(),
                endpointActuators=goal.tolist(),endpointResidual=residual,projectionRequested=ratios.tolist(),
                palmDirectionWeight=PALM_DIRECTION_WEIGHT,
                **(dict(palmBasisFeasibleReference=self.basis_reference(target).tolist(),palmPlanePriority='retain wrist/elbow/longitudinal route; retreat unsupported transverse view') if self.palm_plane else {}),
                reasons=['endpoint-solution-limited'] if residual>1 else [])
            if target.mode=='hold':
                h=np.diag(1/np.maximum(abs(goal-q),1)**2)
                return h,h@(.5*(goal-q))
            # If the complete candidate is geometrically blocked, retain the
            # actual task-space metric rather than following each parameter by
            # the same fractional amount (which can sacrifice palm direction).
            j=self.full_jacobian(np.vstack([self.task_jacobian(actual,jac,target),self.ratio_jacobian(actual,jac)]))
            error=self.full_error(actual,target,q)
            if target.palm_heading is not None:
                # Planned 3.5 degrees leaves half a degree for SDK-context error;
                # the unchanged actual acceptance remains4 degrees. If infeasible
                # the frame solver explicitly relaxes this task, never safety.
                angular=task_jacobian(actual,jac,target)[-1]*4.
                angle_error=float((target.palm_heading-heading(actual)+180)%360-180)
                self.task_linear_constraints=(np.array([angular,-angular]),np.array([angle_error+3.5,-angle_error+3.5]))
                self._policy_info.update(palmDirectionConstraint='enforced',plannedPalmErrorDegrees=3.5)
            h=j.T@j+1e-5*np.eye(self.dof)
            return h,j.T@error*(self.feedback_gain if target.mode=='track' else .5)
        return super()._objective(target,now,q,actual,predict,jac,desired)

    def observe(self,applied_q,nodes,geometry=None):
        row=super().observe(applied_q,nodes,geometry)
        measured=self.ratios(np.array(nodes),self.actual_source_nodes)
        requested=self.desired_ratios(self.pending);error=measured-requested
        row.update(projectionApplied=(1+np.array(applied_q[3:6])/100).tolist(),projectionMeasured=measured.tolist(),
            projectionRequested=requested.tolist(),projectionResidual=error.tolist(),projectionStateMet=bool(max(abs(error))<=RATIO_TOL))
        if self.palm_plane:
            applied_basis=np.array([1+applied_q[6]/100,applied_q[7]/100])
            requested_basis=np.array([1.,0.]) if self.pending.palm_basis is None else self.pending.palm_basis
            basis_error=applied_basis-requested_basis
            row.update(palmBasisRequested=requested_basis.tolist(),palmBasisApplied=applied_basis.tolist(),
                palmBasisResidual=basis_error.tolist(),palmBasisStateMet=bool(max(abs(basis_error))<=.03))
            if max(abs(basis_error))>.03:
                extra=['palm-plane-limited']+(['deadline-missed'] if self.last['now']>=self.pending.time else [])
                row['reasons']=sorted(set(row['reasons']+extra))
                if row['status']!='failed':row['status']='limited'
                row['goalCompleted']=False
            if requested_basis[0]<=0:
                row['reasons']=sorted(set(row['reasons']+['palm-back-face-unsupported']))
        if max(abs(error))>RATIO_TOL:
            row['reasons']=sorted(set(row['reasons']+['projection-target-limited']))
            if row['status']!='failed':row['status']='limited'
            row['goalCompleted']=False
        if row['palmErrorDeg'] is not None and row['palmErrorDeg']>4:
            row['reasons']=sorted(set(row['reasons']+['palm-direction-limited']))
            if row['status']!='failed':row['status']='limited'
        if np.any(requested<np.array([.75,.55,.4])-RATIO_TOL) or np.any(requested>1+RATIO_TOL):
            row['reasons']=sorted(set(row['reasons']+['projection-range-limited']))
            if row['status']!='failed':row['status']='limited'
        if geometry is not None and geometry.get('rawMinTriangleAreaRatio',1)<.28:
            row['reasons']=sorted(set(row['reasons']+['actual-raw-mesh-area-violation']));row['status']='failed'
        return row
