"""Few fixed frontal-shortening tasks, identical node requests for baseline/new."""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import sys,json,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from tools.control.pilot import Plant
from tools.control.patch_projection import PatchProjectionPlant
from tools.control.body_session import nodes_goal
from tools.control.validate_stateful_arm import run,APPROVED
from tools.control.stateful_arm import DT
from tools.control.native_target import NATIVE_ROUTE_LIMITS
from tools.expression import unity


def execute(root,projection=True,rooted=False):
    root.mkdir(parents=True,exist_ok=False);p=Plant();m=p.model;m.set({});a=p.anchors.read()
    native=nodes_goal([a[k] for k in ('shoulder','elbow','wrist','palm')])
    home=np.array(json.loads((APPROVED/'protocol.json').read_text())['newNodes']);patch=PatchProjectionPlant(m,home,rooted_upper=rooted)
    stages=[('acquire',30),('upper-shorten',60),('forearm-shorten',60),('palm-shorten',60),('combined',60),('wrist-under-projection',60),('return',60)]
    packets=[];stimuli=[]
    for stage,count in stages:
        for j in range(count):
            q=np.array([-15.,35.,0.,0.,0.,0.]);u=j/max(1,count-1);w=np.sin(np.pi*u)**2
            if stage=='upper-shorten':q[3]=-20*w
            elif stage=='forearm-shorten':q[4]=-25*w
            elif stage=='palm-shorten':q[5]=-55*w
            elif stage=='combined':q[3:]=np.array([-15,-20,-30])*w
            elif stage=='wrist-under-projection':q[2]=20*np.sin(2*np.pi*u);q[5]=-20*w
            elif stage=='return':q*=0
            nodes=patch.forward(q);clock=(len(packets)+1)*DT
            target=nodes_goal(nodes)|dict(mode='hold' if stage in ('acquire','return') else 'track',
                time=1. if stage=='acquire' else 12. if stage=='return' else clock,projection=(1+q[3:]/100).tolist())
            packets.append(dict(stage=stage,sourceTime=None,headBody=[0.]*6,
                targets=dict(native=native|dict(mode='hold',time=.6),patch=target)))
            stimuli.append(dict(stage=stage,qUsedOnlyToConstructNodeTarget=q.tolist()))
    unity.write_spec(root/'acceptance-protocol.json',dict(stages=stages,fps=30,physics=False,
        task='angles fixed while independently shortening upper/forearm/palm, then combined, wrist+/-, return',
        minimumRequestedRatios=[.8,.75,.45],ratioError=.03,wrist=.005,elbow=.005,palmDegrees=4,
        safety='same fast budgets; projected guard>=.5 AND raw area>=.28, flips/pair/restore0; fixed backend retains raw>=.5',
        hold='last15 return frames settled; keep all tracking frames and all limited cases',visualAcceptance=False))
    unity.write_spec(root/'fixture.json',stimuli);shutil.copy2(__file__,root/'patch_projection_trial.py')
    stream=dict(frames=packets,metadata=dict(scope='direct frontal projection acceptance fixture, not gesture learning',
        wristBinding=str(APPROVED),rawTargets='same original material-node requests for fixed/projection backend'))
    run(root/'sdk',request_stream=stream,native_five=True,native_limits=NATIVE_ROUTE_LIMITS.copy(),patch_projection=projection,patch_rooted=rooted)


if __name__=='__main__':execute(Path(sys.argv[1]),projection='--fixed-length' not in sys.argv,rooted='--patch-rooted' in sys.argv)
