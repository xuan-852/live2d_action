"""SDK-physics response prediction, episode-held-out; not a policy training claim."""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,datetime,hashlib,subprocess,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from tools.expression import unity
from tools.control.pilot import Plant
from tools.whole.batch import markers

def clock():return datetime.datetime.now().astimezone().isoformat()
def ridge(x,y):
    mean=x.mean(0);scale=x.std(0);scale[scale<1e-8]=1
    z=np.c_[np.ones(len(x)),(x-mean)/scale]
    w=np.linalg.solve(z.T@z+1e-3*np.eye(z.shape[1]),z.T@y)
    return lambda v:np.c_[np.ones(len(v)),(v-mean)/scale]@w,(mean,scale,w)

def transitions(u,s):
    """Past/current observations only; command at i+1 is the selected action."""
    if len(u)!=121 or s.shape!=(121,4) or not np.isfinite(u).all() or not np.isfinite(s).all():
        raise ValueError('Expected 121 finite paired samples')
    return [{'command':u[i+1], 'state':s[i], 'previous':s[i-1],
             'priorInput':u[i], 'target':s[i+1]} for i in range(2,len(s)-1)]

def run(root):
    root.mkdir(parents=True,exist_ok=False);start=time.monotonic();plant=Plant()
    before=root/'before';before.mkdir()
    for file in ['preview/unity/LabMotion.cs','preview/unity/WholeArmSession.cs','tools/control/pilot.py','tools/control/physics_probe.py']:
        shutil.copy2(file,before/Path(file).name)
    (root/'git-before.txt').write_text(subprocess.check_output(['git','status','--short'],text=True))
    protocol={'started':clock(),'trainEpisodes':[0,1,2,3],'testEpisodes':[4,5],
      'seed':4911,'fps':30,'framesPerEpisode':121,'contexts':'ParamBodyAngleX/Y inputs, NOT direct X2 output',
      'methods':['Core no-physics model','current-command ridge','state-feedback delta ridge','history delta ridge','persistence','constant-velocity'],
      'prediction':'next frame elbow+wrist 4 coordinates; next arm commands/body input available; physics outputs not given',
      'frozenBeforeCapture':True,'scope':'one-step prediction; not closed-loop policy or expression quality',
      'budget':'12 serial Unity tasks <=900s each; BLAS1; no API; no generation',
      'resourceIdentity':{'moc':plant.model.sha},'resourceLimits':'serial tasks; peak memory/CPU not measured',
      'successCriterion':'report RMS per complete heldout episode, not random frame split; history must beat constant velocity too'}
    unity.write_spec(root/'protocol.json',protocol)
    rng=np.random.default_rng(4911);ids=['Param94','Param97','ParamBodyAngleX','ParamBodyAngleY']
    poses=[{'shoulder':0,'elbow':0,'wrist':0,'projection':{'upperLength':1,'forearmLength':1,'palmLength':1,'palmWidth':1,'palmShear':0},'girdle':{'rootX':0,'rootY':0,'socketX':0,'socketY':0}} for _ in range(121)]
    patch=unity.write_spec(root/'neutral-patch.json',{'poses':poses,'nativeMarkers':markers(plant.model,plant.anchors),'layerPolicy':'legacy-z'})
    capture=[]
    for ep in range(6):
        t=np.arange(121)/30;phase=rng.uniform(0,6.28,4)
        amp=1 if ep<5 else 1.4
        values=[7+12*np.sin(t*1.3+phase[0]),8*np.sin(t*1.7+phase[1]),amp*2*np.sin(t*1.1+phase[2]),amp*2*np.cos(t*.9+phase[3])]
        # Explicit common neutral ramp in/out, keeping each episode independent.
        envelope=np.minimum(np.minimum(t/.5,(4-t)/.5),1).clip(0,1)
        values=[v*envelope for v in values]
        for physics in (False,True):
            label=f'ep{ep}-'+('physics' if physics else 'off');folder=root/label
            cf=unity.write_spec(root/(label+'.curves.json'),{'fps':30,'duration':4,'physics':physics,'excludeX2':False,'ids':ids,'channels':[{'values':v.tolist()} for v in values]})
            ts=time.monotonic()
            unity.run('LabMotion.RunMotion',folder,{'ACTION_LAB_MODEL':unity.model3_path(),'ACTION_LAB_CURVES':cf,'ACTION_LAB_PATCH':patch,'ACTION_LAB_LAYER_FRAME':''},root/(label+'.log'))
            capture.append({'episode':ep,'physics':physics,'seconds':time.monotonic()-ts,'finished':clock()})
            unity.write_spec(root/'capture-time.json',capture)
            print(label,'captured',flush=True)
    datasets={};checks=[]
    for ep in range(6):
        for physics in (False,True):
            label=f'ep{ep}-'+('physics' if physics else 'off')
            fs=[json.loads(p.read_text(encoding='utf-8-sig')) for p in sorted((root/label).glob('[0-9][0-9][0-9][0-9].json'))]
            u=np.array([[next(p['value'] for p in f['requested'] if p['id']==k) for k in ids] for f in fs])
            s=np.array([f['wholeArms']['nativeChain'][2:6] for f in fs])
            a=[{p['id']:p['value'] for p in f['applied']} for f in fs]
            if [f['frame'] for f in fs]!=list(range(121)) or any(abs(f['actionTime']-i/30)>1e-5 for i,f in enumerate(fs)):
                raise ValueError('Nonuniform or missing frame/action time')
            if any(not (root/label/f'{i:04}.png').is_file() for i in range(121)):
                raise ValueError('Missing paired PNG')
            rows=transitions(u,s)
            datasets[(ep,physics)]=rows
            checks.append({'label':label,'pairing':max(f['pairingMaxDelta'] for f in fs),'restore':max(f['wholeArms']['restoredMeshError'] for f in fs),
              'armWriterMaxDelta':max(abs(a[i][k]-u[i,j]) for i in range(len(fs)) for j,k in enumerate(ids[:2])),
              'X2Range':[min(x['ParamBodyAngleX2'] for x in a),max(x['ParamBodyAngleX2'] for x in a)],
              'Y2Range':[min(x['ParamBodyAngleY2'] for x in a),max(x['ParamBodyAngleY2'] for x in a)]})
    def xrow(r,mode):
        if mode=='command':return r['command']
        v=np.r_[r['command'],r['state']]
        return v if mode=='feedback' else np.r_[v,r['state']-r['previous'],r['priorInput']]
    metrics=[];models={}
    for physics in (False,True):
        train=[r for ep in range(4) for r in datasets[(ep,physics)]]
        for mode in ('command','feedback','history'):
            x=np.array([xrow(r,mode) for r in train]);y=np.array([r['target']-(r['state'] if mode!='command' else 0) for r in train])
            predict,weights=ridge(x,y);models[str(physics)+'-'+mode]=weights
            for ep in (4,5):
                rows=datasets[(ep,physics)];pred=predict(np.array([xrow(r,mode) for r in rows]))
                if mode!='command':pred+=np.array([r['state'] for r in rows])
                truth=np.array([r['target'] for r in rows]);err=pred-truth
                metrics.append({'physics':physics,'episode':ep,'method':mode,'rms':float(np.sqrt(np.mean(err**2))),'maxPointError':float(np.linalg.norm(err.reshape(-1,2,2),axis=2).max())})
        for ep in (4,5):
            rows=datasets[(ep,physics)];truth=np.array([r['target'] for r in rows])
            for name,pred in [('persistence',np.array([r['state'] for r in rows])),
                              ('constant-velocity',np.array([2*r['state']-r['previous'] for r in rows])),
                              ('Core-static',np.array([plant.observe(r['command'][:2],0) for r in rows]))]:
                metrics.append({'physics':physics,'episode':ep,'method':name,'rms':float(np.sqrt(np.mean((pred-truth)**2)))})
    np.savez(root/'models.npz',**{k+'-'+str(i):v for k,parts in models.items() for i,v in enumerate(parts)})
    unity.write_spec(root/'checks.json',checks);unity.write_spec(root/'metrics.json',metrics)
    unity.write_spec(root/'summary.json',{'finished':clock(),'wallSeconds':time.monotonic()-start,'frames':1452,'trainingTransitionsPerCondition':472,'testTransitionsPerEpisode':118,'metrics':metrics,'humanAccepted':False,'policyTested':False})
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':run(Path(sys.argv[1]))
