"""Upper node-sequence -> independent lower adapters -> SDK. No policy training."""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import sys,json,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from tools.control.pilot import Plant
from tools.control.dual_task import rotate
from tools.whole.batch import markers
from tools.expression import unity


def chain(home,angles):
    p=np.asarray(home).copy()
    for j in range(1,4):p[j]=p[j-1]+rotate(home[j]-home[j-1],sum(angles[:j]))
    return p


def inverse_patch(home,target):
    a=np.degrees(np.arctan2(np.diff(home,axis=0)[:,1],np.diff(home,axis=0)[:,0]))
    b=np.degrees(np.arctan2(np.diff(target,axis=0)[:,1],np.diff(target,axis=0)[:,0]))
    total=(b-a+180)%360-180
    # Relative joint rotations also need wrapping. Subtracting two wrapped
    # absolute headings can otherwise invent a +/-360 degree joint turn.
    relative=np.r_[total[0],np.diff(total)]
    return (relative+180)%360-180


def run(root,approved):
    root.mkdir(parents=True,exist_ok=False);plant=Plant();m=plant.model
    shutil.copy2(__file__,root/'frozen-upper-node-trial.py')
    patch_home=np.array(json.loads((approved/'protocol.json').read_text())['newNodes'])
    m.set({});a=plant.anchors.read();native_home=np.array([a[k] for k in ('shoulder','elbow','wrist','palm')])
    count=241;targets=[];controls=[];q=np.zeros(3);keys=['Param94','Param97','Param99']
    def native(z):
        m.set(dict(zip(keys,z)));p=plant.anchors.read();return np.array([p[k] for k in ('shoulder','elbow','wrist','palm')])
    for i in range(count):
        t=i/30;envelope=np.sin(np.pi*t/8)**2
        n=chain(native_home,np.array([-15,18,-12])*envelope)
        p=chain(patch_home,np.array([-35,55,-25])*envelope)
        targets.append([n.tolist(),p.tolist()])
        for _ in range(3):
            base=native(q);jac=np.stack([((native(q+np.eye(3)[j]*.1)-base)/.1).ravel() for j in range(3)],axis=1)
            step=np.linalg.solve(jac.T@jac+1e-5*np.eye(3),jac.T@(n-base).ravel())
            q=np.clip(q+np.clip(step,-1.5,1.5),[-10,-12,-25],[25,12,25])
        controls.append(np.r_[q,inverse_patch(patch_home,p)].tolist())
    # Adapter cannot access generating angles: takes only the exported node packet.
    unity.write_spec(root/'node-targets.json',{'fps':30,'nodes':targets,'sides':['native','patch'],'order':['shoulder','elbow','wrist','palm'],
        'scope':'programmed node trajectory; not learned upper generator'})
    unity.write_spec(root/'protocol.json',{'frames':count,'fps':30,'physics':False,
        'approvedWristSource':str(approved),'upper':'fixed-root two-arm node trajectory, programmed NOT trained',
        'lower':'native Core least-squares node fitting, patch relative segment-angle inverse',
        'limitations':'native node chain may be unreachable; residuals retained; no head/body or shoulder translation certification',
        'purpose':'upper/lower node interface smoke test, not naturalness or training advantage'})
    c=np.array(controls)
    curves=unity.write_spec(root/'curves.json',{'fps':30,'duration':8,'physics':False,'ids':keys,'channels':[{'values':c[:,j].tolist()} for j in range(3)]})
    poses=[{'shoulder':float(v[3]),'elbow':float(v[4]),'wrist':float(v[5]),
        'projection':{'upperLength':1,'forearmLength':1,'palmLength':1,'palmWidth':1,'palmShear':0},
        'girdle':{'rootX':0,'rootY':0,'socketX':0,'socketY':0}} for v in c]
    patch=unity.write_spec(root/'patch.json',{'poses':poses,'nativeMarkers':markers(m,plant.anchors),'correctedPatchNodes':patch_home.ravel().tolist()})
    unity.run('LabMotion.RunMotion',root/'capture',{'ACTION_LAB_MODEL':unity.model3_path(),'ACTION_LAB_CURVES':curves,'ACTION_LAB_PATCH':patch,'ACTION_LAB_LAYER_FRAME':''},root/'capture.log')
    frames=[json.loads((root/'capture'/f'{i:04}.json').read_text(encoding='utf-8-sig')) for i in range(count)]
    actual=np.array([[np.array(f['wholeArms'][k]).reshape(4,2) for k in ('nativeChain','patchChain')] for f in frames]);error=actual-np.array(targets)
    unity.write_spec(root/'results.json',{'nativeRMS':float(np.sqrt(np.mean(error[:,0]**2))),'patchRMS':float(np.sqrt(np.mean(error[:,1]**2))),
        'maxFlips':max(f['wholeArms']['flippedTriangles'] for f in frames),'minAreaRatio':min(f['wholeArms']['minTriangleAreaRatio'] for f in frames),
        'maxPairing':max(f['pairingMaxDelta'] for f in frames),'maxRestore':max(f['wholeArms']['restoredMeshError'] for f in frames),'trained':False})
    print((root/'results.json').read_text())

if __name__=='__main__':run(Path(sys.argv[1]),Path(sys.argv[2]))
