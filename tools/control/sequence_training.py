"""Upper model-space arm-node continuation, complete source-heldout imitation."""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from tools.expression import source
from tools.control.upper_node_trial import chain

def feature(history):return np.r_[history[-1],np.diff(history,axis=0).ravel()]

def model_nodes(home,angles):
    angles=np.degrees(angles)
    return np.array([chain(home[s],[angles[s*2],angles[s*2+1]-angles[s*2],0]) for s in range(2)])

def run(root,references,interface):
    root.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,root/'frozen-sequence-training.py')
    home=np.array(json.loads((interface/'node-targets.json').read_text())['nodes'][0])
    files=sorted(references.glob('*.json'));rng=np.random.default_rng(22109);order=rng.permutation(len(files))
    split={str(files[i]):('train' if j<12 else 'validation' if j<15 else 'test') for j,i in enumerate(order)}
    protocol={'seed':22109,'split':split,'task':'30-frame observed prefix -> autoregressive remaining arm-node sequence',
        'representation':'two sides upper/forearm heading changes in radians, decoded to current model node lengths; fixed shoulders, neutral relative palm',
        'loss':'next-frame angular velocity MSE; whole-sequence node/velocity/acceleration errors evaluated separately',
        'training':'MLP 20x64x4, 2000 fixed minibatches Adam; no validation/test weight selection',
        'scope':'generated-reference imitation, NOT naturalness certification, no semantic task labels, no SDK training',
        'limitations':'no head/body orientation labels; no depth or translation; fixed model roots; source retargeting candidate not approval',
        'testStatus':'complete source-file split, previously exposed sources NOT semantically unseen',
        'fps':30,'prefixFrames':30}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2));np.save(root/'home.npy',home)
    data=[];xs=[];ys=[];hashes=[]
    for file in files:
        tr=source.load(file);times=np.arange(int(min(tr['keys'][-1][0],6)*30)+1)/30
        angles=[]
        for t in times:
            f=source.at_time(tr,float(t));a=[]
            for side in ('left','right'):
                p=np.array([f[side+n] for n in ('Shoulder','Elbow','Wrist')]);v=np.diff(p,axis=0)
                a.extend(np.arctan2(v[:,1],-v[:,0]))
            angles.append(a)
        angles=np.unwrap(np.array(angles),axis=0);angles-=angles[0]
        data.append((file.name,split[str(file)],angles));hashes.append({'file':str(file),'sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
        if split[str(file)]=='train':
            for i in range(5,len(angles)):
                xs.append(feature(angles[i-5:i]));ys.append(angles[i]-angles[i-1])
    (root/'sources.json').write_text(json.dumps(hashes,indent=2))
    x=np.array(xs);y=np.array(ys);mean=x.mean(0);scale=x.std(0);scale[scale<1e-5]=1
    x=(x-mean)/scale;outscale=np.maximum(y.std(0),.001);y=y/outscale
    params=[rng.normal(0,.1,(20,64)),np.zeros(64),rng.normal(0,.1,(64,4)),np.zeros(4)]
    initial=[p.copy() for p in params];mom=[np.zeros_like(p) for p in params];var=[p.copy() for p in mom]
    for step in range(1,2001):
        ids=rng.choice(len(x),min(256,len(x)),replace=False);z=x[ids];h=np.tanh(z@params[0]+params[1]);pred=h@params[2]+params[3]
        e=2*(pred-y[ids])/pred.size;dh=(e@params[2].T)*(1-h*h)
        grads=[z.T@dh,dh.sum(0),h.T@e,e.sum(0)]
        for j,g in enumerate(grads):
            mom[j]=.9*mom[j]+.1*g;var[j]=.999*var[j]+.001*g*g
            params[j]-=.001*(mom[j]/(1-.9**step))/(np.sqrt(var[j]/(1-.999**step))+1e-8)
    np.savez(root/'weights.npz',mean=mean,scale=scale,outscale=outscale,**{f'p{i}':p for i,p in enumerate(params)})
    metrics=[];tracks=[]
    for name,group,truth in data:
        if group=='train':continue
        reference=np.array([model_nodes(home,a) for a in truth])
        for method in ('hold','velocity','untrained','trained'):
            generated=list(truth[:30].copy());p=initial if method=='untrained' else params
            for i in range(30,len(truth)):
                if method=='hold':delta=np.zeros(4)
                elif method=='velocity':delta=generated[-1]-generated[-2]
                else:
                    z=(feature(np.array(generated[-5:]))-mean)/scale
                    delta=(np.tanh(z@p[0]+p[1])@p[2]+p[3])*outscale
                generated.append(generated[-1]+np.clip(delta,-.08,.08))
            nodes=np.array([model_nodes(home,a) for a in generated]);err=nodes[30:]-reference[30:]
            metrics.append({'source':name,'split':group,'method':method,'sequenceNodeRMS':float(np.sqrt(np.mean(err**2))),
                'velocityRMS':float(np.sqrt(np.mean((np.diff(nodes[29:],axis=0)-np.diff(reference[29:],axis=0))**2))),
                'maxNodeStep':float(np.linalg.norm(np.diff(nodes,axis=0),axis=-1).max()),
                'accelerationRMS':float(np.sqrt(np.mean(np.diff(nodes[29:],n=2,axis=0)**2)))})
            if group=='test':tracks.append({'source':name,'method':method,'nodes':nodes.tolist(),'reference':reference.tolist()})
    (root/'results.json').write_text(json.dumps(metrics,indent=2));(root/'tracks.json').write_text(json.dumps(tracks))
    print(json.dumps({m:float(np.mean([r['sequenceNodeRMS'] for r in metrics if r['split']=='test' and r['method']==m])) for m in ('hold','velocity','untrained','trained')},indent=2))

if __name__=='__main__':run(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]))
