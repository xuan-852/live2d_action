"""Dual-arm/head/body coordination smoke experiment, procedural NOT learned."""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,threading,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from tools.control.pilot import Plant
from tools.control.closed_loop import increment
from tools.expression import unity
from tools.whole.batch import markers


def heading(chain):
    p=np.asarray(chain).reshape(-1,2)
    v=np.diff(p[:3],axis=0)
    return np.degrees(np.arctan2(v[:,1],v[:,0]))


def angle_error(target,actual):return (np.asarray(target)-actual+180)%360-180


def run(root):
    root.mkdir(parents=True,exist_ok=False);plant=Plant();count=241
    before=root/'before';before.mkdir()
    for p in (Path(__file__),Path('preview/unity/LabMotion.cs'),Path('preview/unity/WholeArmSession.cs')):shutil.copy2(p,before/p.name)
    pose=lambda a:{'shoulder':float(a[0]),'elbow':float(a[1]),'wrist':0,
        'projection':{'upperLength':1,'forearmLength':1,'palmLength':1,'palmWidth':1,'palmShear':0},
        'girdle':{'rootX':0,'rootY':0,'socketX':0,'socketY':0}}
    patch=unity.write_spec(root/'patch.json',{'nativeMarkers':markers(plant.model,plant.anchors),'poses':[pose([0,0]) for _ in range(count)]})
    # Distinct asymmetric goals; no source-motion or semantic-success claims.
    tasks=[{'native':[16,-5],'patch':[10,20],'head':[10,2,-3],'body':[2,0]},
           {'native':[4,5],'patch':[24,-12],'head':[-10,2,3],'body':[-2,1]},
           {'native':[12,-3],'patch':[18,8],'head':[0,4,0],'body':[0,1]},
           {'native':[0,0],'patch':[0,0],'head':[0,0,0],'body':[0,0]}]
    unity.write_spec(root/'protocol.json',{'tasks':tasks,'frames':count,'fps':30,'phaseFrames':60,
        'methods':['simultaneous','head-leading'],'headLeadFrames':12,
        'scope':'procedural goal scheduling with actual arm feedback, NOT trained coordination',
        'constraints':'native slew1.5; patch slew1deg; head slew.6; body slew.15; patch bounded +/-35deg',
        'metrics':'native current-context target RMS; patch material headings; actual head/body tracking; pairing/restore/flips',
        'limitations':'body input tracking not visual body-axis certification; patch headings not visible hand task certification; no naturalness pass'})
    summary=[]
    for method in ('simultaneous','head-leading'):
        folder=root/method;bridge=root/(method+'-bridge');bridge.mkdir();errors=[]
        def capture():
            try:unity.run('LabMotion.RunControl',folder,{'ACTION_LAB_MODEL':unity.model3_path(),'ACTION_LAB_PATCH':patch,
                'ACTION_LAB_LAYER_FRAME':'','ACTION_LAB_CONTROL_FRAMES':count,'ACTION_LAB_CONTROL_BRIDGE':bridge},root/(method+'.log'))
            except Exception as exc:errors.append(exc)
        thread=threading.Thread(target=capture);thread.start()
        q=np.zeros(2);a=np.zeros(2);head=np.zeros(3);body=np.zeros(2);last=None;rest=None;rows=[]
        for i in range(count):
            phase=min(i//60,3);task=tasks[phase];armPhase=phase
            if method=='head-leading' and i%60<12:armPhase=max(phase-1,0) if phase else -1
            arm=tasks[armPhase] if armPhase>=0 else {'native':[0,0],'patch':[0,0]}
            head+=np.clip(np.array(task['head'])-head,-.6,.6)
            body+=np.clip(np.array(task['body'])-body,-.15,.15)
            values={}
            if last is not None:values={p['id']:p['value'] for p in last['applied']}
            def observe(z):
                for k,v in (values|dict(zip(('Param94','Param97'),z))).items():plant.model.params[plant.model.idx[k]]=v
                plant.model.update(plant.model.ptr);p=plant.anchors.read();return np.r_[p['elbow'],p['wrist']]
            target=observe(arm['native']).copy()
            if last is not None:
                actual=np.array(last['wholeArms']['nativeChain'][2:6]);base=observe(q)
                jac=np.stack([(observe(q+np.eye(2)[j]*.1)-base)/.1 for j in range(2)],axis=1)
                q+=increment(jac,target-actual,q)
                current=heading(last['wholeArms']['patchChain'])
                desired=rest+np.array([arm['patch'][0],sum(arm['patch'])])
                err=angle_error(desired,current)
                a+=np.clip([err[0],err[1]-err[0]],-1,1);a=np.clip(a,-35,35)
            cmd=bridge/f'{i:04}.command.json';temp=cmd.with_suffix('.tmp')
            unity.write_spec(temp,{'q':q.tolist(),'body':body.tolist(),'head':head.tolist(),'patch':pose(a),'goal':target.tolist()});temp.replace(cmd)
            begin=time.monotonic()
            while not (bridge/f'{i:04}.ready').exists():
                if errors:raise errors[0]
                if time.monotonic()-begin>180:raise TimeoutError(method+str(i))
                time.sleep(.01)
            last=json.loads((folder/f'{i:04}.json').read_text(encoding='utf-8-sig'))
            if rest is None:rest=heading(last['wholeArms']['patchChain'])
            desired=rest+np.array([arm['patch'][0],sum(arm['patch'])])
            rows.append({'frame':i,'phase':phase,'armPhase':armPhase,'nativeRMS':float(np.sqrt(np.mean((np.array(last['wholeArms']['nativeChain'][2:6])-target)**2))),
                'patchHeadingMaxError':float(max(abs(angle_error(desired,heading(last['wholeArms']['patchChain']))))),
                'pairing':last['pairingMaxDelta'],'restore':last['wholeArms']['restoredMeshError'],
                'flips':last['wholeArms']['flippedTriangles'],'patchAngles':a.tolist()})
        thread.join()
        if errors:raise errors[0]
        unity.write_spec(root/(method+'-track.json'),rows)
        result={'method':method,'phaseMetrics':[{'nativeRMS':float(np.mean([r['nativeRMS'] for r in rows[k*60+50:k*60+60]])),
            'patchHeadingMaxError':float(np.mean([r['patchHeadingMaxError'] for r in rows[k*60+50:k*60+60]]))} for k in range(4)],
            'maxPairing':max(r['pairing'] for r in rows),'maxRestore':max(r['restore'] for r in rows),'maxFlips':max(r['flips'] for r in rows)}
        summary.append(result);unity.write_spec(root/'results.json',summary);print(result,flush=True)


if __name__=='__main__':run(Path(sys.argv[1]))
