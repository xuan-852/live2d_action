"""Real SDK feedback via an atomic file handshake, not precomputed animation."""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,threading,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from tools.control.pilot import Plant,LO,HI,STEP
from tools.expression import unity
from tools.whole.batch import markers


def increment(jac,error,q):
    delta=np.linalg.solve(jac.T@jac+1e-5*np.eye(2),jac.T@error)
    return np.clip(np.clip(delta,-STEP,STEP),LO-q,HI-q)


def run(root,source):
    root.mkdir(parents=True,exist_ok=False)
    plant=Plant();count=181
    before=root/'before';before.mkdir()
    for p in (Path(__file__),Path('preview/unity/LabMotion.cs')):shutil.copy2(p,before/p.name)
    shutil.copy2(source/'models.npz',root/'frozen-models.npz')
    weights=np.load(root/'frozen-models.npz')
    mean,scale,w=[weights['True-history-'+str(i)] for i in range(3)]
    rng=np.random.default_rng(8031)
    goals=[rng.uniform([-2,-7],[19,7],(2,2)) for _ in range(2)]
    unity.write_spec(root/'protocol.json',{'seed':8031,'frames':count,'fps':30,'methods':['numerical','history'],
        'goalParameters':[g.tolist() for g in goals],'goalSwitchFrames':[0,60,120],
        'bounds':[LO.tolist(),HI.tolist()],'slew':STEP,'physics':True,
        'frozenModelSource':str(source),'scope':'2 new exposed test episodes, not naturalness certification',
        'criterion':'last 10 frames of each 60-frame phase RMS < .005; report per phase',
        'fairness':'same steps/slew; numerical baseline uses full applied parameter context and extra Core queries; not equal sensing/compute',
        'resource':'serial Unity; peak resources unmeasured'})
    patch=unity.write_spec(root/'patch.json',{'nativeMarkers':markers(plant.model,plant.anchors),
        'poses':[{'shoulder':0,'elbow':0,'wrist':0,'projection':{'upperLength':1,'forearmLength':1,'palmLength':1,'palmWidth':1,'palmShear':0},
                  'girdle':{'rootX':0,'rootY':0,'socketX':0,'socketY':0}} for _ in range(count)]})
    results=[]
    for ep in range(2):
        targets=[plant.observe(g,0).copy() for g in goals[ep]]+[plant.observe([0,0],0).copy()]
        for method in ('numerical','history'):
            label=f'ep{ep}-{method}';output=root/label;bridge=root/(label+'-bridge');bridge.mkdir()
            errors=[]
            def capture():
                try:unity.run('LabMotion.RunControl',output,{'ACTION_LAB_MODEL':unity.model3_path(),'ACTION_LAB_PATCH':patch,
                    'ACTION_LAB_LAYER_FRAME':'','ACTION_LAB_CONTROL_FRAMES':count,'ACTION_LAB_CONTROL_BRIDGE':bridge},root/(label+'.log'))
                except Exception as exc:errors.append(exc)
            thread=threading.Thread(target=capture);thread.start()
            q=np.zeros(2);previous=None;state=None;prior=np.zeros(4);applied=None;rows=[]
            for i in range(count):
                body=np.array([2*np.sin(i/30*1.1+ep),2*np.sin(i/30*.9)])*min(i/15,1)
                target=targets[min(i//60,2)]
                if state is not None:
                    if method=='numerical':
                        def predict(z):
                            for key,value in (applied|dict(zip(('Param94','Param97'),z))).items():
                                plant.model.params[plant.model.idx[key]]=value
                            plant.model.update(plant.model.ptr)
                            a=plant.anchors.read();return np.r_[a['elbow'],a['wrist']]
                    else:
                        def predict(z):
                            x=np.r_[z,body,state,state-previous,prior]
                            return state+np.r_[1,(x-mean)/scale]@w
                    base=predict(q)
                    jac=np.stack([(predict(q+np.eye(2)[j]*(.1 if q[j]<HI[j]-.1 else -.1))-base)/(.1 if q[j]<HI[j]-.1 else -.1) for j in range(2)],axis=1)
                    q=q+increment(jac,target-base,q)
                command=bridge/f'{i:04}.command.json';temp=command.with_suffix('.tmp')
                unity.write_spec(temp,{'q':q.tolist(),'body':body.tolist(),'goal':target.tolist()});temp.replace(command)
                start=time.monotonic()
                while not (bridge/f'{i:04}.ready').exists():
                    if errors:raise errors[0]
                    if time.monotonic()-start>180:raise TimeoutError(label+str(i))
                    time.sleep(.01)
                frame=json.loads((output/f'{i:04}.json').read_text(encoding='utf-8-sig'))
                actual=np.array(frame['wholeArms']['nativeChain'][2:6])
                previous=state.copy() if state is not None else actual.copy();state=actual
                applied={p['id']:p['value'] for p in frame['applied']}
                prior=np.r_[q,body]
                rows.append({'frame':i,'error':float(np.sqrt(np.mean((actual-target)**2))),'q':q.tolist(),
                    'goal':target.tolist(),'actual':actual.tolist(),'pairing':frame['pairingMaxDelta'],
                    'restore':frame['wholeArms']['restoredMeshError']})
            thread.join()
            if errors:raise errors[0]
            unity.write_spec(root/(label+'-track.json'),rows)
            phases=[float(np.mean([r['error'] for r in rows[a:b]])) for a,b in ((50,60),(110,120),(170,180))]
            results.append({'episode':ep,'method':method,'phaseRMS':phases,'success':[v<.005 for v in phases],
                'maxPairing':max(r['pairing'] for r in rows),'maxRestore':max(r['restore'] for r in rows)})
            unity.write_spec(root/'results.json',results);print(results[-1],flush=True)


if __name__=='__main__':run(Path(sys.argv[1]),Path(sys.argv[2]))
