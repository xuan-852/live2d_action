"""Fixed direct palm-plane tests; shape targets do not change wrist route."""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import json,sys,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from tools.control.pilot import Plant
from tools.control.patch_projection import PatchProjectionPlant
from tools.control.palm_plane import palm_basis
from tools.control.validate_stateful_arm import run,APPROVED
from tools.control.body_session import nodes_goal
from tools.control.native_target import NATIVE_ROUTE_LIMITS
from tools.expression.unity import write_spec


def execute(root):
    root.mkdir(parents=True,exist_ok=False)
    plant=Plant();a=plant.anchors.read();native=nodes_goal([a[k] for k in ('shoulder','elbow','wrist','palm')])
    approved=np.array(json.loads((APPROVED/'protocol.json').read_text())['newNodes'])
    patch=PatchProjectionPlant(plant.model,approved,rooted_upper=True)
    q=np.array([-15.,35.,0.,-10.,-10.,-40.]);nodes=patch.forward(q)
    stages=[('acquire',30,[1.,0.]),('oblique-positive',60,palm_basis(.6,.48,1).tolist()),
        ('flat-return',30,[1.,0.]),('oblique-negative',60,palm_basis(.6,.48,-1).tolist()),
        ('edge-limit',60,palm_basis(.6,.01,-1).tolist()),('recover',60,[1.,0.])]
    packets=[]
    for stage,count,basis in stages:
        deadline=(len(packets)+30)/30
        for i in range(count):
            packets.append(dict(stage=stage,sourceTime=None,headBody=[0.]*6,targets=dict(
                native=native|dict(mode='hold',time=.6),patch=nodes_goal(nodes)|dict(mode='hold',time=deadline,projection=[.9,.9,.6],palm_basis=basis))))
    write_spec(root/'acceptance-protocol.json',dict(stages=stages,fps=30,
        criteria=dict(wrist=.005,elbow=.005,palmDegrees=4,basis=.03,nodeSpeed=.72,nodeAcceleration=16.2,parameterSpeed=135,parameterAcceleration=1620,guard=.5,raw=.28,flips=0),
        knownFailure='edge target beyond single-face area/width capability; report limited, then recover',
        evaluation='last15 frames of each hold; all transitions retained',visualAcceptance=False))
    shutil.copy2(__file__,root/'palm_plane_trial.py')
    run(root/'sdk',request_stream=dict(frames=packets,metadata=dict(scope='palm plane direct acceptance, not upper choreography')),
        native_five=True,native_limits=NATIVE_ROUTE_LIMITS.copy(),patch_projection=True,patch_rooted=True,palm_plane=True)


if __name__=='__main__':execute(Path(sys.argv[1]))
