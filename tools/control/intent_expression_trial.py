"""Three newly authored expression tasks; a diagnostic scenario, not a learned policy.

Source actuator seeds ONLY author feasible material-node references. The live
executor receives nodes/velocities, hand pose, head/body and gaze requests.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import sys,json,ctypes as C,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from tools.control.pilot import Plant
from tools.control.patch_projection import bind
from tools.control.gesture_exploration import POSES
from tools.control.native_target import NATIVE_KEYS,NATIVE_ROUTE_LIMITS
from tools.control.body_session import nodes_goal,HEAD_BODY_IDS
from tools.control.timed_arm import heading
from tools.control.validate_stateful_arm import run,APPROVED
from tools.control.finger_control import FINGER_IDS
from tools.expression.unity import write_spec

POINT=np.array([15.,0.,6.,0.,15.])
OFFER=np.array([12.,0.,5.,0.,-12.])
BEAT=np.array([18.,0.,8.,0.,-5.])
NEAR=np.array([36.581066,6.681845,19.701235,-2.462181,20.])
ZERO=np.zeros(5)
POINT_HEAD=np.array([14.,-3.,0.,0.,0.,0.])
NEAR_HEAD=np.array([-2.,1.,-3.,0.,0.,0.])


def build_markers(plant):
    m=plant.model;m.set({});a=plant.anchors.read();v=m.mesh[m.names.index('ArtMesh231')]
    distance=np.linalg.norm(v-a['wrist'],axis=1)
    def vertex(name,index):return dict(id=name,mesh='ArtMesh231',indices=[int(index)],weights=[1.])
    markers=[vertex('indexBase',np.argmin(distance)),vertex('indexTip',np.argmax(distance))]
    i=m.names.index('ArtMesh128')
    counts=np.ctypeslib.as_array(m.api('GetDrawableIndexCounts',C.POINTER(C.c_int),[C.c_void_p])(m.ptr),shape=(m.n,))
    ptr=m.api('GetDrawableIndices',C.POINTER(C.POINTER(C.c_ushort)),[C.c_void_p])(m.ptr)
    tri=np.ctypeslib.as_array(ptr[i],shape=(int(counts[i]),)).reshape(-1,3)
    ids,weights=bind(m.mesh[i],tri,[.07,.50])
    markers.append(dict(id='headSide',mesh='ArtMesh128',indices=ids.tolist(),weights=weights.tolist()))
    for label,mesh in [('irisLeft','ArtMesh105'),('irisRight','ArtMesh319')]:
        count=len(m.mesh[m.names.index(mesh)])
        markers.append(dict(id=label,mesh=mesh,indices=list(range(count)),weights=[1/count]*count))
    return markers


def read_markers(model,markers):
    return {x['id']:np.asarray(x['weights'])@model.mesh[model.names.index(x['mesh'])][x['indices']] for x in markers}


def plan(variant='coordinated'):
    if variant not in ('coordinated','neutral-fingers'):raise ValueError('unknown hand ablation')
    p=Plant();m=p.model;markers=build_markers(p)
    patch=nodes_goal(json.loads((APPROVED/'protocol.json').read_text(encoding='utf-8'))['newNodes'])
    def evaluate(q,hb,fingers=None):
        context=dict(zip(NATIVE_KEYS,map(float,q)))|dict(zip(HEAD_BODY_IDS,map(float,hb)))
        if fingers is not None:context|=dict(zip(FINGER_IDS,map(float,fingers)))
        m.set(context);a=p.anchors.read()
        return np.array([a[k] for k in ('shoulder','elbow','wrist','palm')]),read_markers(m,markers)
    point_nodes,point_markers=evaluate(POINT,POINT_HEAD,POSES['index-emphasis'])
    axis=point_markers['indexTip']-point_markers['indexBase'];axis/=np.linalg.norm(axis)
    point_target=point_markers['indexTip']+.15*axis
    near_nodes,near_markers=evaluate(NEAR,NEAR_HEAD,POSES['soft-curl'])
    fixed=dict(pointTarget=point_target.tolist(),pointMaterialAxis='indexBase -> indexTip, 2D proxy only',
        nearHeadPalmOffset=(near_nodes[3]-near_markers['headSide']).tolist(),nearHeadGap=[.02,.09])
    frames=[];phases=[];q=ZERO.copy();hb=np.zeros(6)
    def add(task,phase,count,target_q,target_hb,hand,gaze,hold=False):
        nonlocal q,hb
        start=len(frames);duration=count/30;initial_q=q.copy();initial_hb=hb.copy()
        finger_deadline=(start+min(count,24))/30;gaze_deadline=(start+min(count,15))/30
        for j in range(count):
            u=(j+1)/count;w=u*u*u*(10-15*u+6*u*u)
            author_q=initial_q+(np.asarray(target_q)-initial_q)*w
            context=initial_hb+(np.asarray(target_hb)-initial_hb)*w
            nodes,_=evaluate(author_q,context)
            clock=(len(frames)+1)/30
            native=nodes_goal(nodes)|dict(time=clock,mode='track')
            if hold:native|=dict(time=start/30,mode='hold')
            selected=POSES[hand] if variant=='coordinated' else POSES['neutral']
            frames.append(dict(stage=task+'/'+phase,sourceTime=None,headBody=context.tolist(),
                targets=dict(native=native,patch=patch|dict(mode='hold',time=clock+.8)),
                nativeFingerGoal=dict(values=selected.copy(),deadline=finger_deadline),
                gazeGoal=dict(values=list(gaze),deadline=gaze_deadline),
                intent=dict(task=task,phase=phase,kind='newly-authored diagnostic request',hand=hand,variant=variant,
                    pointTarget=fixed['pointTarget'] if task=='point' else None,
                    nearHeadOffset=fixed['nearHeadPalmOffset'] if task=='near-head' else None)))
        phases.append(dict(task=task,phase=phase,startFrame=start+1,endFrame=len(frames),frames=count,hold=hold,hand=hand,gaze=list(gaze),
            authoringSeedOnly=list(map(float,target_q)),headBody=list(map(float,target_hb))))
        q=np.asarray(target_q,float).copy();hb=np.asarray(target_hb,float).copy()
    add('idle','baseline',30,ZERO,[0]*6,'neutral',[0,0],True)
    add('point','prepare',15,ZERO,POINT_HEAD,'soft-curl',[.7,-.25])
    add('point','reach',30,POINT,POINT_HEAD,'index-emphasis',[.7,-.25])
    add('point','express',36,POINT,POINT_HEAD,'index-emphasis',[.7,-.25],True)
    add('point','withdraw',27,ZERO,[0]*6,'neutral',[0,0])
    add('point','rest',18,ZERO,[0]*6,'neutral',[0,0],True)
    add('explain','prepare',12,ZERO,[1,0,1,0,0,0],'soft-curl',[0,0])
    add('explain','offer',27,OFFER,[2,0,1,0,0,0],'open-accent',[0,0])
    add('explain','emphasize',18,BEAT,[2,0,2,0,0,0],'soft-curl',[0,0])
    add('explain','reopen',18,OFFER,[1,0,1,0,0,0],'open-accent',[0,0])
    add('explain','express',24,OFFER,[1,0,1,0,0,0],'open-accent',[0,0],True)
    add('explain','withdraw',27,ZERO,[0]*6,'neutral',[0,0])
    add('explain','rest',18,ZERO,[0]*6,'neutral',[0,0],True)
    add('near-head','prepare',12,ZERO,NEAR_HEAD,'soft-curl',[0,.15])
    add('near-head','approach',42,NEAR,NEAR_HEAD,'soft-curl',[.15,.2])
    add('near-head','express',36,NEAR,NEAR_HEAD,'soft-curl',[.15,.2],True)
    add('near-head','withdraw',36,ZERO,[0]*6,'neutral',[0,0])
    add('near-head','rest',24,ZERO,[0]*6,'neutral',[0,0],True)
    # Same executed-interval convention as existing native track control.
    previous,previous_h=evaluate(ZERO,np.zeros(6));previous_angle=heading(previous)
    for packet in frames:
        target=packet['targets']['native'];nodes=np.array([previous[0],target['elbow'],target['wrist'],[0,0]],float)
        angle=target['palm_heading'];rate=((angle-previous_angle+180)%360-180)*30
        if target['mode']=='track':
            target['velocity']=((np.array(target['wrist'])-previous[2])*30).tolist();target['angular_velocity']=float(rate)
        previous[1]=target['elbow'];previous[2]=target['wrist'];previous_angle=angle
    return dict(frames=frames,metadata=dict(scope='three hand-authored expression tasks, not learned upper policy',
        handK='not used: user clarified it contains no concrete hand/finger motion',
        variant=variant,sourceTiming='new fixed 30Hz authored scenario, no posthoc retiming',fixed=fixed,
        patch='neutral reference; no finger execution',palmProjection='identity')),phases,markers,fixed


def execute(root,variant='coordinated'):
    root.mkdir(parents=True,exist_ok=False);stream,phases,markers,fixed=plan(variant)
    write_spec(root/'acceptance-protocol.json',dict(fps=30,physics=False,variant=variant,phases=phases,fixed=fixed,
        arm=dict(wrist=.005,elbow=.005,palmDegrees=4),finger=dict(error=.15,settledSpeed=.5,speed=90,acceleration=540),
        gaze=dict(error=.02,settledSpeed=.05,speed=3,acceleration=15,scope='screen-direction proxy, no 3D eye ray'),
        pointing=dict(angleDegrees=6,phase='last15 point/express frames',targetFrozenBeforeSDK=True),
        nearHead=dict(palmToHeadSideDistance=[.02,.09],phase='last15 near-head/express frames',contact=False),
        evaluation='last15 express/rest holds; all transitions, deadlines and failures retained',
        visualAcceptance=False,source='new authoring authorized by user; approved gesture execution anchor retained'))
    write_spec(root/'task-markers.json',markers);shutil.copy2(__file__,root/Path(__file__).name)
    run(root/'sdk',request_stream=stream,native_five=True,native_limits=NATIVE_ROUTE_LIMITS.copy(),
        native_fingers=True,native_gaze=True,task_markers=markers)


if __name__=='__main__':execute(Path(sys.argv[1]),'neutral-fingers' if '--neutral-fingers' in sys.argv else 'coordinated')
