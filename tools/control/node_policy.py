"""Upper-layer node trajectory imitation. No parameter/control labels in policy IO."""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from tools.control.dual_task import rotate


def demonstration(home,angles,phase):
    """Synthetic minimum-jerk chain teacher, NOT human motion supervision."""
    w=10*phase**3-15*phase**4+6*phase**5
    result=home.copy()
    for side in range(2):
        result[side,1]=home[side,0]+rotate(home[side,1]-home[side,0],angles[side,0]*w)
        result[side,2]=result[side,1]+rotate(home[side,2]-home[side,1],sum(angles[side])*w)
    return result


def observation(current,home,wrists,phase,scale):
    origin=home[:,0].mean(0)
    return np.r_[((current-origin)/scale).ravel(),((wrists-origin)/scale).ravel(),phase]


class Network:
    def __init__(self,seed=42):
        rng=np.random.default_rng(seed)
        self.params=[rng.normal(0,.15,(17,64)),np.zeros(64),rng.normal(0,.1,(64,8)),np.zeros(8)]

    def predict(self,x):
        a,b,c,d=self.params
        return np.tanh(x@a+b)@c+d

    def fit(self,x,y,iterations=1200):
        moments=[np.zeros_like(p) for p in self.params];velocity=[p.copy() for p in moments]
        rng=np.random.default_rng(633)
        for step in range(1,iterations+1):
            ids=rng.choice(len(x),min(256,len(x)),replace=False);z=x[ids];target=y[ids]
            a,b,c,d=self.params;hidden=np.tanh(z@a+b);pred=hidden@c+d
            error=2*(pred-target)/pred.size
            dh=(error@c.T)*(1-hidden**2)
            grads=[z.T@dh,dh.sum(0),hidden.T@error,error.sum(0)]
            for i,g in enumerate(grads):
                moments[i]=.9*moments[i]+.1*g;velocity[i]=.999*velocity[i]+.001*g*g
                self.params[i]-=.002*(moments[i]/(1-.9**step))/(np.sqrt(velocity[i]/(1-.999**step))+1e-8)


def decode(raw,home,scale):
    result=home.copy();result[:,1:]=home[:,1:]+raw.reshape(2,2,2)*scale
    return result


def run(root,neutral):
    root.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,root/'frozen-node-policy.py')
    frame=json.loads(neutral.read_text(encoding='utf-8-sig'))
    home=np.array([frame['wholeArms'][key][:6] for key in ('nativeChain','patchChain')]).reshape(2,3,2)
    scale=np.linalg.norm(home[0,0]-home[1,0]);rng=np.random.default_rng(11091)
    tasks=rng.uniform([-20,-25],[25,25],(128,2,2))
    protocol={'seed':11091,'trainEpisodes':list(range(96)),'validationEpisodes':list(range(96,112)),'testEpisodes':list(range(112,128)),
        'input':'actual/current six nodes, two desired wrist endpoints, phase; no actuator parameters, no target elbow',
        'output':'next six node positions; shoulders fixed to home, learned elbow/wrist positions',
        'teacher':'synthetic rotated chains with minimum-jerk timing; not human/naturalness evidence',
        'training':'supervised node-position MSE, numpy tanh MLP17x64x8; 1200 fixed minibatches; no test selection',
        'scope':'node-space upper model only; SDK execution NOT tested by this script',
        'neutralSHA':hashlib.sha256(neutral.read_bytes()).hexdigest(),'neutralSource':str(neutral),
        'phaseSteps':60,'constraints':'no learned shoulder motion, head/body, physical feedback or goal switching yet',
        'metrics':'teacher-forced validation and autoregressive whole-test rollouts; per episode endpoint and chain length errors'}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    np.savez(root/'tasks.npz',home=home,tasks=tasks)
    xs=[];ys=[];validation=[]
    for episode,angles in enumerate(tasks[:112]):
        wrists=demonstration(home,angles,1)[:,2]
        for i in range(60):
            current=demonstration(home,angles,i/60);target=demonstration(home,angles,(i+1)/60)
            x=observation(current,home,wrists,(i+1)/60,scale)
            y=((target[:,1:]-home[:,1:])/scale).ravel()
            if episode<96:xs.append(x);ys.append(y)
            else:validation.append((x,y))
    net=Network();initial=[p.copy() for p in net.params]
    net.fit(np.array(xs),np.array(ys))
    np.savez(root/'weights.npz',**{f'p{i}':p for i,p in enumerate(net.params)})
    vx,vy=map(np.array,zip(*validation))
    metrics=[];tracks=[]
    trained=[p.copy() for p in net.params]
    for episode in range(112,128):
        angles=tasks[episode];wrists=demonstration(home,angles,1)[:,2]
        for method in ('stay','untrained','trained'):
            current=home.copy();trajectory=[current.tolist()]
            net.params=initial if method=='untrained' else trained
            for i in range(60):
                if method!='stay':current=decode(net.predict(observation(current,home,wrists,(i+1)/60,scale)),home,scale)
                trajectory.append(current.tolist())
            lengths=np.linalg.norm(np.diff(np.array(trajectory),axis=2),axis=-1)
            rest=np.linalg.norm(np.diff(home,axis=1),axis=-1)
            metrics.append({'episode':episode,'method':method,'endpointWristRMS':float(np.sqrt(np.mean((current[:,2]-wrists)**2))),
                'maxLengthRelativeError':float(abs(lengths/rest-1).max())})
            tracks.append({'episode':episode,'method':method,'goalWrists':wrists.tolist(),'nodes':trajectory})
    net.params=trained
    (root/'results.json').write_text(json.dumps({'validationNormalizedMSE':float(np.mean((net.predict(vx)-vy)**2)),'tests':metrics},indent=2))
    (root/'tracks.json').write_text(json.dumps(tracks))
    print(json.dumps({'validationNormalizedMSE':float(np.mean((net.predict(vx)-vy)**2)),
        'meanTestEndpointRMS':{m:float(np.mean([x['endpointWristRMS'] for x in metrics if x['method']==m])) for m in ('stay','untrained','trained')}},indent=2))


if __name__=='__main__':run(Path(sys.argv[1]),Path(sys.argv[2]))
