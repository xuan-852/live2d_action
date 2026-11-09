"""Frozen relational adaptation experiment; local inputs, Core and sandbox only."""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'): os.environ[k]='1'
import sys,json,time,hashlib,shutil,subprocess,datetime,ctypes
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from tools.expression import unity
from tools.whole.batch import sample_document,markers,read,CASES,summarize
from tools.whole.recovered.core_model import Model
from tools.whole.recovered.anchors import Anchors
from tools.whole.review import source_at,source_image
import imageio_ffmpeg
from PIL import Image,ImageDraw

def stamp(): return datetime.datetime.now().astimezone().isoformat()

def target(points):
    a,b=np.array(points['左腕']),np.array(points['右腕'])
    centre=(a+b)/2; width=np.linalg.norm(a[:2]-b[:2])
    return np.array([(np.array(points[k])[:2]-centre[:2])/width for k in ('左ひじ','左手首','右ひじ','右手首')])

def rotate(v,a):
    a=np.deg2rad(a);return np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])@v

def bounded_solve(residual,seed,lo,hi):
    q=np.clip(seed,lo,hi);r=residual(q);calls=1;damping=.01
    for _ in range(24):
        jac=np.empty((len(r),len(q)))
        for j in range(len(q)):
            probe=q.copy();probe[j]=q[j]+.25 if q[j]+.25<=hi[j] else q[j]-.25
            jac[:,j]=(residual(probe)-r)/(probe[j]-q[j]);calls+=1
        step=np.linalg.solve(jac.T@jac+damping*np.eye(len(q)),-jac.T@r)
        nxt=np.clip(q+np.clip(step,-8,8),lo,hi);nr=residual(nxt);calls+=1
        if nr@nr<r@r:q,r=nxt,nr;damping=max(1e-7,damping*.5)
        else:damping=min(100,damping*3)
        if np.linalg.norm(step)<1e-4:break
    return q,float(r@r/2),calls

def patch_chain(q,proj):
    s=np.array([-.105,.310]);e=s+rotate(np.array([-.04,-.105])*proj['upperLength'],q[0])
    w=e+rotate(np.array([-.054,-.180])*proj['forearmLength'],q[0]+q[1])
    return e,w

def prepare(root):
    root.mkdir(parents=True,exist_ok=False)
    before=root/'before';before.mkdir()
    subprocess.run(['git','status','--short'],stdout=(root/'git-before.txt').open('w'),check=True)
    for folder in ('preview/unity','tools/whole','tests'):
        shutil.copytree(unity.ROOT/folder,before/folder,ignore=shutil.ignore_patterns('__pycache__'))
    old=Path('E:/FuXuanLab/runs/20261030-generated-observation-04/compiled')
    # All were generated/compiled previously. Only original 8 used in sandbox tuning.
    extra=['novel-03','novel-06','novel-07','novel-11']
    cases=CASES+extra
    history={'regression':CASES,'development': ['novel-01','novel-08'],
        'heldoutThisTrial':extra,'history':'previously generated/compiled in October; no sandbox use found in docs/tools/tests search; not globally unseen',
        'selection':'first four numerically ordered previously compiled successes outside original eight; no render selection',
        'revisionBudget':1,'revisionsUsed':0,'frozenAt':stamp(),
        'hypothesis':'body-relative elbow/wrist relational targets replace direction-only transfer',
        'visibility':'shared comparison-visible diagnostic distal foreground; not realistic depth inference',
        'budgets':{'generation':'disabled','analysis':'BLAS 1 thread; sequential','unity':'sequential; 900s each task','encoding':'1 thread','paidAPI':0}}
    for c in cases:
        dst=root/'inputs'/c;dst.mkdir(parents=True)
        for fn in (c+'.json',c+'.json.reference.json','map.json'):
            shutil.copy2(old/c/fn,dst/fn)
    hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'inputs').rglob('*') if p.is_file()}
    history['inputHashes']=hashes
    unity.write_spec(root/'protocol.json',history)
    return cases

def adapt(root,c,doc,m,anchors):
    from copy import deepcopy
    out=deepcopy(doc); diagnostic=[]
    # Normalisation explicitly retains positions; no first-frame subtraction.
    m.set({});neutral=anchors.read();ns=neutral['shoulder'];ps=np.array([-.105,.310])
    centre=(ns+ps)/2;width=np.linalg.norm(ns-ps)
    prior=None
    for i,row in enumerate(out['frames']):
        if i==0 or i==len(out['frames'])-1: continue
        points,_=source_at(root,c,row['t']); desired=target(points)*width+centre
        support={k:v for k,v in row['native'].items() if k not in ('Param94','Param95','Param97','Param98')}
        keys=['Param94','Param95','Param97','Param98']
        seed=np.array([row['native'][k] for k in keys]+[row['shoulder'],row['elbow']])
        lo=np.array([-29.5,-29.5,-29.8,-29.5,-56,0]);hi=np.array([59.5,29.5,29.8,29.5,56,145])
        # Warm-start continuous state, but no rate clamp that suppresses target excursions.
        seed=np.clip(seed if prior is None else prior,lo,hi)
        def residual(q):
            m.set(support|dict(zip(keys,q[:4])))
            n=anchors.read();pe,pw=patch_chain(q[4:],row['projection'])
            actual=np.array([n['elbow'],n['wrist'],pe,pw])
            pos=(actual-desired)/width
            relation=((actual[1]-actual[3])-(desired[1]-desired[3]))/width
            # Position + elbow route + bilateral relation; no clip-dependent weights.
            return np.r_[pos.ravel(),relation]
        q,cost,calls=bounded_solve(residual,seed,lo,hi);prior=q
        row['native'].update({k:float(v) for k,v in zip(keys,q[:4])})
        row['shoulder'],row['elbow']=map(float,q[4:]);diagnostic.append({'key':i,'cost':cost,'nfev':calls})
    return out,diagnostic

def execute(root):
    cases=prepare(root) if not (root/'protocol.json').exists() else CASES+read(root/'protocol.json')['heldoutThisTrial']
    m=Model(unity.model3_path().parent,(unity.PROJECT/'Assets/Live2D/Cubism/Plugins/Windows/x86_64/Live2DCubismCore.dll').resolve());a=Anchors(m)
    # Fail early on source schema before exposing heldout rendering.
    from tools.whole.recovered.core_model import N
    results={};start=time.monotonic()
    for c in cases:
        doc=read(root/'inputs'/c/(c+'.json'));begin=stamp()
        b,diag=adapt(root,c,doc,m,a);unity.write_spec(root/f'{c}-solve.json',diag)
        results[c]={'started':begin,'manualEdits':0,'caseBranches':0}
        for label,d in [('A',doc),('B',b)]:
            grid,curves,poses=sample_document(d)
            cf=unity.write_spec(root/f'{c}-{label}-curves.json',{'fps':30,'duration':float(grid[-1]),'physics':True,'excludeX2':False,'ids':list(curves),'channels':[{'values':v} for v in curves.values()]})
            pf=unity.write_spec(root/f'{c}-{label}-patch.json',{'poses':poses,'nativeMarkers':markers(m,a),'layerPolicy':'comparison-visible'})
            folder=root/f'{c}-{label}'
            t=time.monotonic();unity.run('LabMotion.RunMotion',folder,{'ACTION_LAB_MODEL':unity.model3_path(),'ACTION_LAB_CURVES':cf,'ACTION_LAB_PATCH':pf,'ACTION_LAB_LAYER_FRAME':''},root/f'{c}-{label}.log')
            results[c][label]=summarize(folder)|{'wallSeconds':time.monotonic()-t}
        results[c]['finished']=stamp();unity.write_spec(root/'results.json',results)
        print(c,'complete',flush=True)
    unity.write_spec(root/'stage.json',{'finished':stamp(),'wallSeconds':time.monotonic()-start})

if __name__=='__main__': execute(Path(sys.argv[1]))
