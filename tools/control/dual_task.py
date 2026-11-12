"""Fixed wrist-region SDK baseline qualification before policy-data collection."""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,threading,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from tools.control.pilot import Plant,LO,HI
from tools.control.task_objective import evaluate,hands
from tools.expression import unity
from tools.whole.batch import markers


def rotate(v,angle):
    a=np.radians(angle);c,s=np.cos(a),np.sin(a)
    return np.array([[c,-s],[s,c]])@v


def patch_predict(chain,current,candidate):
    p=np.asarray(chain).reshape(-1,2)
    delta=np.asarray(candidate)-current
    return p[0]+rotate(p[1]-p[0],delta[0])+rotate(p[2]-p[1],sum(delta))


def wrist_step(predict,q,target,lo,hi,slew):
    base=predict(q);columns=[]
    for j in range(2):
        h=.1 if q[j]+.1<=hi[j] else -.1
        columns.append((predict(q+np.eye(2)[j]*h)-base)/h)
    jac=np.stack(columns,axis=1)
    delta=np.linalg.solve(jac.T@jac+1e-6*np.eye(2),jac.T@(target-base))
    return np.clip(q+np.clip(delta,-slew,slew),lo,hi)


def run(root):
    root.mkdir(parents=True,exist_ok=False);plant=Plant();count=241
    before=root/'before';before.mkdir()
    for file in [Path(__file__),Path('preview/unity/LabMotion.cs'),Path('preview/unity/WholeArmSession.cs')]:shutil.copy2(file,before/file.name)
    (root/'git-before.txt').write_text(__import__('subprocess').check_output(['git','status','--short'],text=True))
    pose=lambda a:{'shoulder':float(a[0]),'elbow':float(a[1]),'wrist':0,
        'projection':{'upperLength':1,'forearmLength':1,'palmLength':1,'palmWidth':1,'palmShear':0},
        'girdle':{'rootX':0,'rootY':0,'socketX':0,'socketY':0}}
    patch=unity.write_spec(root/'patch.json',{'nativeMarkers':markers(plant.model,plant.anchors),'poses':[pose([0,0]) for _ in range(count)]})
    rng=np.random.default_rng(9147)
    native=rng.uniform([6,-6],[20,6],(2,2));patch_angles=rng.uniform([8,-12],[28,20],(2,2))
    unity.write_spec(root/'protocol.json',{'seed':9147,'qualificationOnly':True,'policyTrained':False,
        'nativeGoalSource':native.tolist(),'patchGoalSource':patch_angles.tolist(),'fps':30,'frames':count,
        'phaseFrames':80,'holdFrames':15,'radiusShoulderFraction':.1,
        'scope':'fixed two-hand wrist regions; neutral head/body, SDK physics on',
        'goals':'neutral-context actual native Core wrists; patch neutral chain kinematic candidates, reachability tested not assumed',
        'split':'qualification only, never heldout policy evaluation','visual':'unknown until reviewed; not successful training data',
        'constraints':'native bounds [-10,25]/[-12,12],slew1.5; patch +/-35deg,slew1deg',
        'resource':'one serial Unity task; CPU/memory peaks not measured'})
    output=root/'baseline';bridge=root/'bridge';bridge.mkdir();errors=[]
    def capture():
        try:unity.run('LabMotion.RunControl',output,{'ACTION_LAB_MODEL':unity.model3_path(),'ACTION_LAB_PATCH':patch,
            'ACTION_LAB_LAYER_FRAME':'','ACTION_LAB_CONTROL_FRAMES':count,'ACTION_LAB_CONTROL_BRIDGE':bridge},root/'baseline.log')
        except Exception as exc:errors.append(exc)
    thread=threading.Thread(target=capture,daemon=True);thread.start()
    q=np.zeros(2);a=np.zeros(2);last=None;goals=None;frames=[]
    try:
        for i in range(count):
            phase=min(i//80,2)
            if last is None:target=plant.observe([0,0],0)[2:].copy()
            else:
                values={p['id']:p['value'] for p in last['applied']}
                def observe(z):
                    for k,v in (values|dict(zip(('Param94','Param97'),z))).items():plant.model.params[plant.model.idx[k]]=v
                    plant.model.update(plant.model.ptr);return plant.anchors.read()['wrist'].copy()
                target=goals[phase][0]
                q=wrist_step(observe,q,target,LO,HI,1.5)
                chain=last['wholeArms']['patchChain']
                old=a.copy()
                a=wrist_step(lambda candidate:patch_predict(chain,old,candidate),a,goals[phase][1],np.full(2,-35),np.full(2,35),1)
            # RunControl's legacy goal field is elbow+wrist, only wrist is task-relevant.
            elbow=plant.anchors.read()['elbow']
            cmd=bridge/f'{i:04}.command.json';temp=cmd.with_suffix('.tmp')
            unity.write_spec(temp,{'q':q.tolist(),'body':[0,0],'head':[0,0,0],'patch':pose(a),'goal':np.r_[elbow,target].tolist()});temp.replace(cmd)
            begin=time.monotonic()
            while not (bridge/f'{i:04}.ready').exists():
                if errors:raise errors[0]
                if time.monotonic()-begin>180:raise TimeoutError(str(i))
                time.sleep(.01)
            last=json.loads((output/f'{i:04}.json').read_text(encoding='utf-8-sig'));frames.append(last)
            if goals is None:
                home=hands(last);chain=last['wholeArms']['patchChain']
                goals=[np.array([plant.observe(n,0)[2:],patch_predict(chain,[0,0],p)]) for n,p in zip(native,patch_angles)]+[home]
                radius=.1*np.linalg.norm(np.array(last['wholeArms']['nativeChain'][:2])-last['wholeArms']['patchChain'][:2])
                unity.write_spec(root/'goals.json',{'goals':[g.tolist() for g in goals],'radius':float(radius),'frozenAfterNeutralBeforeMovement':True})
        thread.join()
        if errors:raise errors[0]
    except Exception as exc:
        unity.write_spec(root/'failure.json',{'error':repr(exc),'note':'worker may wait until SDK handshake timeout; no data promoted'})
        raise
    results=[{'phase':phase,**evaluate(np.array([hands(f) for f in frames[phase*80:phase*80+80]]),goals[phase],radius,15)} for phase in range(3)]
    unity.write_spec(root/'results.json',{'phases':results,'maxPairing':max(f['pairingMaxDelta'] for f in frames),
        'maxRestore':max(f['wholeArms']['restoredMeshError'] for f in frames),'maxFlips':max(f['wholeArms']['flippedTriangles'] for f in frames),
        'policyTrained':False,'visualQualified':False})
    print(json.dumps(results,indent=2))


if __name__=='__main__':run(Path(sys.argv[1]))
