"""Locate existing source/target/actuator discontinuities without recapture."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from tools.control.upper_node_trial import inverse_patch

def peak_motion(points):
    delta=np.linalg.norm(np.diff(points,axis=0),axis=-1)
    index=np.unravel_index(delta.argmax(),delta.shape)
    return {'frameBefore':int(index[0]),'timeBefore':float(index[0]/30),'indices':list(map(int,index[1:])),
            'distancePerFrame':float(delta[index])}

def run(root,source,visible,wrist):
    root.mkdir(parents=True,exist_ok=False)
    home=np.array(json.loads((wrist/'protocol.json').read_text())['newNodes'])
    results=[]
    for name in ('yayoi-right-hand-rise1','makoto-hair','makoto-guard1'):
        case=source/name;t=json.loads((case/'targets.json').read_text());targets=np.array(t['nodes'])
        curves=json.loads((case/'with-arms-curves.json').read_text());q=np.array([c['values'] for c in curves['channels']]).T
        fs=[json.loads((visible/name/'capture'/f'{i:04}.json').read_text(encoding='utf-8-sig')) for i in range(len(targets))]
        actual=np.array([[np.array(f['wholeArms'][k]).reshape(4,2) for k in ('nativeChain','patchChain')] for f in fs])
        changes=abs(np.diff(q,axis=0));idx=np.unravel_index(changes.argmax(),changes.shape)
        old=np.array([d['patchRequestedAngles'] for d in t['diagnostics']]);new=np.array([inverse_patch(home,p[1]) for p in targets])
        changed=np.where(np.any(abs(new-old)>1e-5,axis=1))[0]
        results.append({'case':name,'targetMotionPeak':peak_motion(targets),'actualMotionPeak':peak_motion(actual),
            'parameterPeak':{'frameBefore':int(idx[0]),'id':curves['ids'][idx[1]],'delta':float(changes[idx])},
            'wrapErrorFrames':changed.tolist(),'oldFirstAngles':old[0].tolist(),'correctedFirstAngles':new[0].tolist(),
            'correctedAnglesOutsideCurrentLimitsFrames':np.where(np.any((new<[-56,0,-100])|(new>[56,145,30]),axis=1))[0].tolist(),
            'targetFileSHA':hashlib.sha256((case/'targets.json').read_bytes()).hexdigest(),
            'scope':'no new visual pass or force/naturalness score; source/target vs actual motion have different scales'})
    (root/'diagnosis.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))

if __name__=='__main__':run(*(Path(s) for s in sys.argv[1:5]))
