"""Few candidate basic-hand poses through live SDK feedback, not upper learning."""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import sys,json,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from tools.control.pilot import Plant
from tools.control.body_session import nodes_goal
from tools.control.validate_stateful_arm import run,APPROVED
from tools.control.native_target import NATIVE_ROUTE_LIMITS
from tools.expression.unity import write_spec

POSES={
    'neutral':[0,0,0,0,0],
    'soft-curl':[-8,-10,-10,-10,-10],
    'gather':[-15,-20,-20,-20,-20],
    'index-emphasis':[-10,20,-18,-18,-18],
    'two-finger':[-10,18,18,-18,-18],
    'open-accent':[10,15,10,5,3],
}


def make_stream():
    p=Plant();m=p.model;home=np.array(json.loads((APPROVED/'protocol.json').read_text())['newNodes'])
    patch=nodes_goal(home);packets=[];stages=[]
    def add(name,count,pose_name,arm=(0,0,0),finger_values=None):
        start=len(packets);deadline=(start+24)/30
        m.set(dict(zip(('Param94','Param97','Param99'),arm)));a=p.anchors.read()
        native=nodes_goal([a[k] for k in ('shoulder','elbow','wrist','palm')])
        values=POSES[pose_name] if finger_values is None else finger_values
        for _ in range(count):
            packets.append(dict(stage=name,sourceTime=None,headBody=[0.]*6,
                nativeFingerGoal=dict(values=list(values),deadline=deadline),
                targets=dict(native=native|dict(mode='hold',time=deadline),patch=patch|dict(mode='hold',time=deadline))))
        stages.append(dict(name=name,startFrame=start+1,endFrame=len(packets),frames=count,pose=pose_name,
            requestedFingerValues=list(values),armFixture=list(arm),deadline=deadline,
            label='candidate only; gesture meaning/naturalness unverified'))
    add('neutral',30,'neutral')
    for name in list(POSES)[1:]:add('shape/'+name,60,name)
    add('shape/neutral-return',30,'neutral')
    add('wrist/index-positive',60,'index-emphasis',(0,0,15))
    add('wrist/index-negative',60,'index-emphasis',(0,0,-15))
    add('raised/gather',90,'gather',(15,6,0))
    add('raised/index',60,'index-emphasis',(15,6,15))
    add('raised/open',60,'open-accent',(15,6,-10))
    add('interrupt/soft',12,'soft-curl',(15,6,0))
    add('interrupt/index',15,'index-emphasis',(15,6,0))
    add('interrupt/gather',18,'gather',(15,6,0))
    add('interrupt/neutral',45,'neutral',(15,6,0))
    add('unavailable',30,'neutral',(15,6,0),[40]*5)
    add('recover',60,'neutral',(15,6,0))
    add('release',90,'neutral')
    return dict(frames=packets,metadata=dict(scope='five candidate hand poses and local composition; no programmed upper policy/learning',
        source='previous isolated visible-channel inventory',candidateSemantics='unverified; not full fist/OK/physical finger joints',
        patch='static reference, no finger execution',palmProjection='identity')),stages


def execute(root):
    root.mkdir(parents=True,exist_ok=False);stream,stages=make_stream()
    write_spec(root/'acceptance-protocol.json',dict(stages=stages,fps=30,physics=False,
        fingerCriteria=dict(maxError=.15,maxSettledSpeed=.5,speed=90,acceleration=540,range=[-20,20],
            evaluation='last15 of holds >=45 frames; short interrupts exempt from arrival, all continuity retained'),
        armCriteria=dict(wrist=.005,elbow=.005,palmDegrees=4,mode='existing native controller; composition not inferred from finger parameter arrival'),
        knownFailures=['40-unit target outside declared20 range must remain limited','short interrupted requests need not reach; no stale target after switch'],
        geometry='record Core replay at full applied SDK context and same-context neutral-finger reference; no naturalness inferred from area',
        palmProjection='identity, no palm width/length/shear actuation',visualAcceptance=False))
    shutil.copy2(__file__,root/Path(__file__).name)
    run(root/'sdk',physics=False,request_stream=stream,native_five=True,native_limits=NATIVE_ROUTE_LIMITS.copy(),native_fingers=True)


if __name__=='__main__':execute(Path(sys.argv[1]))
