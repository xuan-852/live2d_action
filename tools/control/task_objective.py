"""Explicit dual-hand task gates. Unknown visual evidence never becomes success."""
import json,sys
from pathlib import Path
import numpy as np


def longest_run(flags):
    best=current=0
    for flag in flags:
        current=current+1 if flag else 0
        best=max(best,current)
    return best


def evaluate(actual,goals,radius,hold_frames,visual=None):
    actual=np.asarray(actual,float);goals=np.asarray(goals,float)
    if actual.ndim!=3 or actual.shape[1:]!=(2,2) or goals.shape!=(2,2):
        raise ValueError('Expected frame x identified hand x xy and two goals')
    if not np.isfinite(actual).all() or not np.isfinite(goals).all() or radius<=0 or hold_frames<1:
        raise ValueError('Invalid task evidence')
    error=np.linalg.norm(actual-goals,axis=2)
    inside=(error<=radius).all(axis=1)
    run=longest_run(inside)
    material=run>=hold_frames
    if visual is not None and (len(visual)!=len(actual) or any(v not in (True,False,None) for v in visual)):
        raise ValueError('Visual judgments must pair with every frame')
    verified=np.array([v is True for v in visual]) if visual is not None else np.zeros(len(actual),bool)
    return {'materialTaskPassed':bool(material),'visibleTaskPassed':bool(longest_run(inside & verified)>=hold_frames),
        'visualEvidence':'unreviewed' if visual is None or any(v is None for v in visual) else 'reviewed',
        'longestJointHoldFrames':run,'requiredHoldFrames':hold_frames,'radius':float(radius),
        'perHandFinalDistance':error[-1].tolist(),
        'relationFinalError':float(np.linalg.norm((actual[-1,0]-actual[-1,1])-(goals[0]-goals[1]))),
        'positionLossMean':float(np.mean((error/radius)**2)),
        'note':'positionLossMean is diagnostic, never substitutes for hold/visibility gates'}


def hands(frame):
    arms=frame['wholeArms']
    return np.array([arms['nativeChain'][4:6],arms['patchChain'][4:6]],float)


def run(source,output):
    output.mkdir(parents=True,exist_ok=False)
    calibration=[json.loads((source/'simultaneous'/f'{i:04}.json').read_text(encoding='utf-8-sig')) for i in (59,119,179,239)]
    initial=json.loads((source/'simultaneous'/'0000.json').read_text(encoding='utf-8-sig'))
    width=float(np.linalg.norm(np.array(initial['wholeArms']['nativeChain'][:2])-initial['wholeArms']['patchChain'][:2]))
    goals=[hands(f).tolist() for f in calibration[:3]]+[hands(initial).tolist()]
    protocol={'source':str(source),'qualificationOnly':True,'testStatus':'all recordings previously exposed; targets from simultaneous baseline, NOT heldout',
        'goalFrame':'fixed model coordinates; native hand / patch hand identities, no swapping',
        'goals':goals,'radius':.1*width,'radiusMeaning':'0.1 initial inter-shoulder material distance, provisional NOT perceptual calibration',
        'holdFrames':15,'fps':30,'phaseFrames':60,'loss':'mean squared hand distance / radius^2; diagnostic, no training performed',
        'success':'both identified hands within radius for 15 consecutive frames AND same frames visually verified',
        'attention':'not certified: parameter writes do not establish visible attention direction',
        'visual':'unreviewed; visibleTaskPassed must remain false; no whole-task success claim',
        'limitations':'target material wrists are not certified visible hand centres; no hidden-hand reward'}
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    results=[]
    for method in ('simultaneous','head-leading'):
        frames=[json.loads((source/method/f'{i:04}.json').read_text(encoding='utf-8-sig')) for i in range(241)]
        for phase in range(4):
            actual=np.array([hands(f) for f in frames[phase*60:phase*60+60]])
            result=evaluate(actual,goals[phase],protocol['radius'],15)
            results.append({'method':method,'phase':phase,**result})
    (output/'results.json').write_text(json.dumps(results,indent=2))
    print(json.dumps(results,indent=2))


if __name__=='__main__':run(Path(sys.argv[1]),Path(sys.argv[2]))
