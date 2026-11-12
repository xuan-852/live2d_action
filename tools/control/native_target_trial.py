"""Unchanged overhead MMD node goal, then unchanged source-route verification."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from tools.control.pilot import Plant
from tools.control.mmd_timed_controller import make_stream,execute
from tools.control.validate_stateful_arm import APPROVED,run
from tools.control.stateful_arm import speed_limits
from tools.control.timed_arm import heading
from tools.control.native_target import NATIVE_ROUTE_LIMITS


def static(root,source):
    plant=Plant();plant.model.set({});a=plant.anchors.read()
    homes=dict(native=np.array([a[k] for k in ('shoulder','elbow','wrist','palm')]),
               patch=np.array(json.loads((APPROVED/'protocol.json').read_text())['newNodes']))
    doc=json.loads(source.read_text(encoding='utf-8-sig'));stream=make_stream(doc,homes)
    packet=stream['frames'][30+28];packets=[]
    def home_target(side):
        p=homes[side]
        return dict(wrist=p[2].tolist(),elbow=p[1].tolist(),palm_heading=heading(p),mode='hold',time=5.)
    for i in range(180):
        goal=dict(packet['targets']['native']) if i<90 else home_target('native')
        goal.update(mode='hold',time=2. if i<90 else 5.,angular_velocity=0.)
        goal.pop('velocity',None)
        packets.append(dict(stage='overhead-hold' if i<90 else 'neutral-return',sourceTime=.9333333333333333 if i<90 else None,
            headBody=packet['headBody'] if i<90 else [0.]*6,
            targets=dict(native=goal,patch=home_target('patch'))))
    stream=dict(frames=packets,metadata=dict(sourceFile=doc['sourceFile'],sourceFrame=28,
        target='EXACT previous highest wrist request, no scaling or coordinate change',
        acceptance='last15frames each phase: wrist/elbow .005, palm4deg, all-node stop .02, palm rate8; safety unchanged fast budgets'))
    run(root,physics=True,request_stream=stream,native_five=True)


if __name__=='__main__':
    root,source=map(Path,sys.argv[1:3])
    if '--mmd' in sys.argv:
        limits=speed_limits('fast')
        if '--responsive-native' in sys.argv:
            limits=NATIVE_ROUTE_LIMITS.copy()
        execute(root,source,native_five=True,native_limits=limits)
    else:static(root,source)
