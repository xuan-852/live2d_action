"""Restricted native-arm response learning and held-out feedback control.

No SDK physics here: Core is a quasi-static plant, not a robot simulator.
The action is a bounded increment applied at an explicit 30Hz action clock.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'): os.environ[key]='1'
import sys,json,time,datetime,hashlib,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from tools.expression import unity
from tools.whole.recovered.core_model import Model
from tools.whole.recovered.anchors import Anchors

KEYS=('Param94','Param97')
LO=np.array([-10.,-12.]); HI=np.array([25.,12.])
STEP=1.5

def features(x,quadratic=True):
    x=np.asarray(x);x=np.atleast_2d(x)/np.array([30.,30.,3.])
    cols=[np.ones(len(x))]+[x[:,i] for i in range(3)]
    if quadratic:cols += [x[:,i]*x[:,j] for i in range(3) for j in range(i,3)]
    return np.stack(cols,axis=1)

class Forward:
    def __init__(self,quadratic=True): self.quadratic=quadratic;self.weights=None
    def fit(self,x,y):
        f=features(x,self.quadratic)
        self.weights=np.linalg.solve(f.T@f+1e-7*np.eye(f.shape[1]),f.T@y)
    def predict(self,x):return features(x,self.quadratic)@self.weights

class Plant:
    def __init__(self):
        self.model=Model(unity.model3_path().parent,(unity.PROJECT/'Assets/Live2D/Cubism/Plugins/Windows/x86_64/Live2DCubismCore.dll').resolve())
        self.anchors=Anchors(self.model)
    def observe(self,q,context):
        self.model.set(dict(zip(KEYS,q))|{'ParamBodyAngleX2':float(context)})
        p=self.anchors.read()
        # Material landmarks, not certified anatomical joints. All four coordinates
        # retained: reaching a wrist with an incorrect elbow branch is penalised.
        return np.r_[p['elbow'],p['wrist']]

def step_control(q,context,goal,predict,observed):
    # Receding local model; use actual state feedback, model only predicts response.
    jac=np.stack([(predict(q+np.eye(2)[j]*.1,context)-predict(q,context))/.1 for j in range(2)],axis=1)
    delta=np.linalg.solve(jac.T@jac+1e-5*np.eye(2),jac.T@(goal-observed))
    return np.clip(np.clip(delta,-STEP,STEP),LO-q,HI-q)

def run(root):
    root.mkdir(parents=True,exist_ok=False);begin=time.monotonic()
    stamp=lambda:datetime.datetime.now().astimezone().isoformat()
    protocol={'started':stamp(),'checkpoint':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      'actuators':list(KEYS),'bounds':[LO.tolist(),HI.tolist()],'incrementBound':STEP,'fps':30,
      'context':'ParamBodyAngleX2; explicit Core intervention, NOT SDK physics',
      'training':'64 exploration episodes x 24 increments; seed=1911',
      'test':'12 new initial/goal pairs; seed=2901; fixed/changed/OOD context; no test-based revision',
      'methods':['stay','linear-trained','quadratic-trained','numerical-feedback'],
      'success':'last 10 steps four-coordinate RMS < .005 model units; return same tolerance',
      'scope':'quasi-static material control only; no semantic/naturalness/visibility or dynamic physics certification',
      'budgets':'serial CPU; BLAS1; no generation/GPU training/paid API; 120 steps per reach, 120 return'}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    plant=Plant();rng=np.random.default_rng(1911);data=[];xs=[];ys=[]
    for episode in range(64):
        q=rng.uniform(LO+2,HI-2);context=float(rng.uniform(-1,1))
        for frame in range(24):
            s=plant.observe(q,context);action=rng.uniform(-STEP,STEP,2);nq=np.clip(q+action,LO,HI)
            ns=plant.observe(nq,context)
            data.append({'episode':episode,'frame':frame,'q':q.tolist(),'state':s.tolist(),'action':(nq-q).tolist(),'context':context,'nextState':ns.tolist()})
            xs.append(np.r_[nq,context]);ys.append(ns);q=nq
    x=np.array(xs);y=np.array(ys)
    linear=Forward(False);quad=Forward(True);linear.fit(x,y);quad.fit(x,y)
    np.savez(root/'forward-models.npz',linear=linear.weights,quadratic=quad.weights)
    (root/'transitions.json').write_text(json.dumps(data))
    # Repeat and changed-history probe: same parameters, same Core output?
    q=np.array([7.,2.]);a=plant.observe(q,.3);plant.observe(np.array([23.,-10.]),-.8);b=plant.observe(q,.3)
    identity={'mocSHA':plant.model.sha,'sameInputAfterDifferentHistoryMaxDelta':float(np.max(abs(a-b))),
      'meaning':'Core-only recurrence probe; no physics history in this environment',
      'anchorEvidence':plant.anchors.evidence}
    (root/'plant-check.json').write_text(json.dumps(identity,indent=2))
    rng=np.random.default_rng(2901);results=[];prediction=[];tracks={}
    methods={'stay':None,'linear-trained':linear,'quadratic-trained':quad,'numerical-feedback':'oracle'}
    for case in range(12):
        initial=rng.uniform(LO+3,HI-3);goalq=rng.uniform(LO+3,HI-3)
        context=float(rng.uniform(-1,1)) if case<8 else float(rng.choice([-3,3]))
        split='fixed' if case<4 else 'changing' if case<8 else 'OOD-context'
        for method,model in methods.items():
            q=initial.copy();rows=[];home=plant.observe(initial,context)
            for t in range(240):
                ctx=context if split!='changing' else float(context+.5*np.sin(t/30))
                target=plant.observe(goalq if t<120 else initial,ctx)
                actual=plant.observe(q,ctx)
                if method=='stay':action=np.zeros(2)
                else:
                    predict=plant.observe if model=='oracle' else lambda z,c:model.predict(np.r_[z,c])[0]
                    action=step_control(q,ctx,target,predict,actual)
                nq=np.clip(q+action,LO,HI);nactual=plant.observe(nq,ctx)
                rows.append({'frame':t,'q':nq.tolist(),'context':ctx,'goal':target.tolist(),'actual':nactual.tolist(),
                    'error':float(np.sqrt(np.mean((nactual-target)**2))),'action':action.tolist()});q=nq
            reach=np.mean([r['error'] for r in rows[110:120]]);ret=np.mean([r['error'] for r in rows[-10:]])
            results.append({'case':case,'split':split,'method':method,'reachError':reach,'returnError':ret,
                'reachSuccess':bool(reach<.005),'returnSuccess':bool(ret<.005),'maxAction':max(max(abs(np.array(r['action']))) for r in rows)})
            tracks[f'{case}-{method}']=rows
        # True held-out predictions, not training fit.
        for model,label in [(linear,'linear'),(quad,'quadratic')]:
            truth=plant.observe(goalq,context);pred=model.predict(np.r_[goalq,context])[0]
            prediction.append({'case':case,'split':split,'model':label,'RMS':float(np.sqrt(np.mean((truth-pred)**2)))})
    for name,obj in [('results',results),('prediction',prediction),('tracks',tracks)]:
        (root/(name+'.json')).write_text(json.dumps(obj,indent=1))
    summary={m:{'reach':sum(r['reachSuccess'] for r in results if r['method']==m),'return':sum(r['returnSuccess'] for r in results if r['method']==m),'cases':12,
       'meanReachError':float(np.mean([r['reachError'] for r in results if r['method']==m]))} for m in methods}
    (root/'summary.json').write_text(json.dumps({'finished':stamp(),'wallSeconds':time.monotonic()-begin,'transitions':len(data),'results':summary,'physics':False,'humanAccepted':False},indent=2))
    print((root/'summary.json').read_text())

if __name__=='__main__':run(Path(sys.argv[1]))
