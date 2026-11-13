"""Direct inventory of CURRENT controller hand capabilities, not raw parameter playback."""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import sys,json,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from tools.control.pilot import Plant
from tools.control.body_session import nodes_goal
from tools.control.patch_projection import PatchProjectionPlant
from tools.control.validate_stateful_arm import run,APPROVED
from tools.control.native_target import NATIVE_ROUTE_LIMITS
from tools.control.palm_plane import palm_basis
from tools.expression.unity import write_spec

def execute(root):
    root.mkdir(parents=True,exist_ok=False);p=Plant();m=p.model
    home=np.array(json.loads((APPROVED/'protocol.json').read_text())['newNodes'])
    patch=PatchProjectionPlant(m,home,rooted_upper=True)
    m.set({});a=p.anchors.read();native_home=np.array([a[k] for k in ('shoulder','elbow','wrist','palm')])
    stages=[('neutral',45,0,1.,[1.,0.]),('wrist-plus',60,20,1.,[1.,0.]),
        ('wrist-minus',60,-20,1.,[1.,0.]),('palm-shorten',60,0,.65,[1.,0.]),
        ('oblique-positive',60,0,.65,palm_basis(.65,.52,1).tolist()),
        ('oblique-negative',60,0,.65,palm_basis(.65,.52,-1).tolist()),
        ('edge-limit',60,0,.65,palm_basis(.65,.01,-1).tolist()),('recover',90,0,1.,[1.,0.])]
    packets=[];descriptions=[]
    for stage,count,wrist,length,basis in stages:
        # Native fixture parameters construct node goals only, never supplied to controller.
        m.set({'Param99':wrist});a=p.anchors.read();nn=np.array([a[k] for k in ('shoulder','elbow','wrist','palm')])
        q=np.array([0.,0.,wrist,0.,0.,(length-1)*100]);pn=patch.forward(q)
        deadline=(len(packets)+30)/30
        native=nodes_goal(nn)|dict(time=deadline,mode='hold')
        if stage not in ('neutral','wrist-plus','wrist-minus','recover'):
            native=nodes_goal(native_home)|dict(time=deadline,mode='hold')
        pt=nodes_goal(pn)|dict(time=deadline,mode='hold',projection=[1.,1.,length],palm_basis=basis)
        for _ in range(count):packets.append(dict(stage=stage,sourceTime=None,headBody=[0.]*6,targets={'native':native,'patch':pt}))
        descriptions.append({'stage':stage,'frames':count,'nativeTask':'planar wrist heading' if stage.startswith('wrist') else 'neutral reference only',
            'patchTask':{'wristDegrees':wrist,'palmLength':length,'basis':basis},
            'note':'Native does NOT expose palm-plane/independent hand-shape control; neutral there is not task success'})
    write_spec(root/'acceptance-protocol.json',{'stages':descriptions,'physics':False,'fps':30,
        'task':'fixed proximal arm: planar wrist, shortening, oblique plane, side limit, recovery',
        'unsupported':['native palm-plane request API','finger articulation/gesture in both current controller backends','true hand back-face switching'],
        'criteria':'same current controller budgets/guards; evaluate last15 each stage plus all-frame safety; visual pending',
        'scope':'ability inventory, no naturalness/gesture success inferred from material nodes'})
    shutil.copy2(__file__,root/'frozen-hand-trial.py')
    run(root/'sdk',physics=False,request_stream={'frames':packets,'metadata':{'scope':'direct hand capability inventory','stageDescriptions':descriptions}},
        native_five=True,native_limits=NATIVE_ROUTE_LIMITS.copy(),patch_projection=True,patch_rooted=True,palm_plane=True)

if __name__=='__main__':execute(Path(sys.argv[1]))
