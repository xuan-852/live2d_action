"""MMD arm-expression stress test, source targets vs current model execution."""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import sys,json,shutil,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from tools.control.pilot import Plant
from tools.control.mmd_virtual_tracker import signals
from tools.control.calibrate_virtual_tracker import calibrate
from tools.control.mmd_two_experiments import IDS,LIMITS
from tools.control.upper_node_trial import inverse_patch
from tools.control.dual_task import rotate
from tools.expression import unity
from tools.whole.batch import markers

def target_chain(home,target):
    """Explicit fixed-root projection candidate; preserve absolute segment directions."""
    points=[home[0].copy()]
    for j,(heading,ratio) in enumerate([(target['upperHeadingDeg'],target['upperProjectedRatio']),
                                      (target['forearmHeadingDeg'],target['forearmProjectedRatio']),
                                      (target['palmHeadingDeg'],1)]):
        length=np.linalg.norm(home[j+1]-home[j])*ratio
        points.append(points[-1]+rotate(np.array([length,0]),heading))
    return np.array(points)

def solve_native(plant,target,previous,context):
    keys=['Param94','Param95','Param97','Param98','Param99']
    m=plant.model;lo=np.array([m.lo[m.idx[k]] for k in keys])+.05;hi=np.array([m.hi[m.idx[k]] for k in keys])-.05
    q=np.clip(previous,lo,hi)
    def observe(z):
        m.set(context|dict(zip(keys,z)));a=plant.anchors.read()
        return np.array([a[k] for k in ('shoulder','elbow','wrist','palm')])
    for _ in range(12):
        actual=observe(q);jac=[]
        for j in range(5):
            h=.2 if q[j]+.2<=hi[j] else -.2
            jac.append(((observe(q+np.eye(5)[j]*h)-actual)/h).ravel())
        jac=np.stack(jac,axis=1);err=(target-actual).ravel()
        delta=np.linalg.solve(jac.T@jac+1e-6*np.eye(5),jac.T@err)
        proposal=np.clip(q+np.clip(delta,-3,3),lo,hi)
        if np.linalg.norm(observe(proposal)-target)<np.linalg.norm(err):q=proposal
        else:break
    return q,observe(q)

def run(root,sources,approved):
    root.mkdir(parents=True,exist_ok=False);plant=Plant();m=plant.model
    before=root/'before';before.mkdir()
    for f in [Path(__file__),Path('preview/unity/WholeArmSession.cs'),Path('preview/unity/RecoveredShoulderConnection.cs')]:shutil.copy2(f,before/f.name)
    home_patch=np.array(json.loads((approved/'protocol.json').read_text())['newNodes'])
    m.set({});a=plant.anchors.read();home_native=np.array([a[k] for k in ('shoulder','elbow','wrist','palm')])
    names=['yayoi-right-hand-rise1','makoto-hair','makoto-guard1']
    protocol={'sources':names,'fps':30,'trained':False,'wristVersion':str(approved),
        'headBody':'reuse frozen calibrated mapping, full original SDK physics',
        'arms':'source absolute front-projection headings and projected lengths; current model neutral lengths; fixed shoulder roots',
        'native':'five-channel Core node least squares, 12 iterations/frame; no slew guarantee',
        'patch':'relative-angle inverse and projected-length ratios, bounded shoulder +/-56,elbow 0..145,wrist -100..30; upper/fore length .75..1',
        'limitations':['fixed shoulders lose source torso-relative translations','source palm heading is 2D only, no palm-normal/finger matching',
            'native solver plans without live SDK physics context, actual tracking mismatch retained','no temporal retiming or limit smoothing: source/solver jumps not hidden',
            'source maps previously exposed, not unseen tests','not maximum anatomical capability proof; converter can limit result'],
        'baseline':'same head/body, neutral arms; action short original time, no loop'}
    unity.write_spec(root/'protocol.json',protocol);summary=[]
    for name in names:
        case=root/name;case.mkdir();path=sources/name/'map.json';doc=json.loads(path.read_text(encoding='utf-8-sig'))
        shutil.copy2(path,case/'source-map.json')
        times,raw,_,_=signals(doc);hb=np.clip(calibrate(raw),-LIMITS,LIMITS)
        native_values=[];poses=[];targets=[];diagnostics=[];q=np.zeros(5)
        for i,t in enumerate(times):
            nt=doc['sides']['anatomicalLeft']['frames'][i]['target'];pt=doc['sides']['anatomicalRight']['frames'][i]['target']
            n=target_chain(home_native,nt);p=target_chain(home_patch,pt);targets.append([n.tolist(),p.tolist()])
            context=dict(zip(IDS,hb[i]));q,actual=solve_native(plant,n,q,context);native_values.append(q.tolist())
            angles=inverse_patch(home_patch,p);bounded=np.clip(angles,[-56,0,-100],[56,145,30])
            lengths=np.clip([pt['upperProjectedRatio'],pt['forearmProjectedRatio']],.75,1)
            poses.append({'shoulder':float(bounded[0]),'elbow':float(bounded[1]),'wrist':float(bounded[2]),
                'projection':{'upperLength':float(lengths[0]),'forearmLength':float(lengths[1]),'palmLength':1,'palmWidth':1,'palmShear':0},
                'girdle':{'rootX':0,'rootY':0,'socketX':0,'socketY':0}})
            diagnostics.append({'nativePlannedRMS':float(np.sqrt(np.mean((actual-n)**2))),
                'patchRequestedAngles':angles.tolist(),'patchAppliedAngles':bounded.tolist(),
                'patchAngleClipped':bool(np.any(abs(angles-bounded)>1e-6)),
                'patchProjectionClipped':bool(np.any(abs(lengths-[pt['upperProjectedRatio'],pt['forearmProjectedRatio']])>1e-6))})
        unity.write_spec(case/'targets.json',{'times':times.tolist(),'nodes':targets,'diagnostics':diagnostics,'sourceSHA':hashlib.sha256(path.read_bytes()).hexdigest()})
        qv=np.array(native_values);armids=['Param94','Param95','Param97','Param98','Param99']
        for variant in ('head-body-only','with-arms'):
            cv={k:hb[:,j].tolist() for j,k in enumerate(IDS)}
            if variant=='with-arms':cv.update({k:qv[:,j].tolist() for j,k in enumerate(armids)})
            curves=unity.write_spec(case/(variant+'-curves.json'),{'fps':30,'duration':float(times[-1]),'physics':True,'ids':list(cv),'channels':[{'values':v} for v in cv.values()]})
            patch=unity.write_spec(case/(variant+'-patch.json'),{'poses':poses,'nativeMarkers':markers(m,plant.anchors),'correctedPatchNodes':home_patch.ravel().tolist()}) if variant=='with-arms' else ''
            unity.run('LabMotion.RunMotion',case/variant,{'ACTION_LAB_MODEL':unity.model3_path(),'ACTION_LAB_CURVES':curves,
                'ACTION_LAB_PATCH':patch,'ACTION_LAB_LAYER_FRAME':'','ACTION_LAB_PHYSICS_OVERRIDE':''},case/(variant+'.log'))
        fs=[json.loads((case/'with-arms'/f'{i:04}.json').read_text(encoding='utf-8-sig')) for i in range(len(times))]
        achieved=np.array([[np.array(f['wholeArms'][k]).reshape(4,2) for k in ('nativeChain','patchChain')] for f in fs]);err=achieved-np.array(targets)
        summary.append({'source':name,'frames':len(times),'nativeNodeRMS':float(np.sqrt(np.mean(err[:,0]**2))),
            'patchNodeRMS':float(np.sqrt(np.mean(err[:,1]**2))),'maxFlips':max(f['wholeArms']['flippedTriangles'] for f in fs),
            'minAreaRatio':min(f['wholeArms']['minTriangleAreaRatio'] for f in fs),'maxPairing':max(f['pairingMaxDelta'] for f in fs),
            'maxRestore':max(f['wholeArms']['restoredMeshError'] for f in fs),
            'patchAngleClippedFrames':sum(d['patchAngleClipped'] for d in diagnostics),'patchProjectionClippedFrames':sum(d['patchProjectionClipped'] for d in diagnostics),
            'visualReview':'pending, no expression success inferred'})
        unity.write_spec(root/'results.json',summary);print(name,'captured',flush=True)

if __name__=='__main__':run(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]))
