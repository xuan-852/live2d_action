"""Input-only and separated physics-output attenuation, same MMD source/time."""
import sys,json,hashlib,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from tools.expression import unity
from tools.control.mmd_virtual_tracker import signals

BODY={'ParamBodyAngleX2','ParamBodyAngleY2','ParamBodyAngleZ2','Param251'}

def calibrate(raw):
    gains=np.array([.65,-.35,-.45,.06,-.15,-.15])
    return raw*gains

def attenuate(original,body,secondary):
    data=json.loads(json.dumps(original))
    for group in data['PhysicsSettings']:
        for output in group['Output']:
            output['Weight']*=body if output['Destination']['Id'] in BODY else secondary
    return data

def run(root,previous):
    root.mkdir(parents=True,exist_ok=False);source=previous/'source-map.json'
    shutil.copy2(source,root/'source-map.json');shutil.copy2(__file__,root/'frozen-driver.py')
    shutil.copy2('preview/unity/LabMotion.cs',root/'frozen-LabMotion.cs')
    document=json.loads(source.read_text(encoding='utf-8-sig'));times,raw,_,_=signals(document);values=calibrate(raw)
    ids=['ParamAngleX','ParamAngleY','ParamAngleZ','ParamBodyAngleX','ParamBodyAngleY','ParamBodyAngleZ']
    limits=np.array([25,25,20,8,8,8])
    if (abs(values)>limits).any():raise ValueError('Calibration exceeds frozen safety range; no silent clipping')
    physicsPath=unity.model3_path().parent/'符玄.physics3.json'
    physics=json.loads(physicsPath.read_text(encoding='utf-8-sig'))
    variants=[('calibrated-full',1,1),('secondary-half',1,.5),('body-secondary-half',.5,.5)]
    unity.write_spec(root/'protocol.json',{'sourceSHA':hashlib.sha256(source.read_bytes()).hexdigest(),
        'physicsSHA':hashlib.sha256(physicsPath.read_bytes()).hexdigest(),'fps':30,'frames':len(times),
        'gains':[.65,-.35,-.45,.06,-.15,-.15],'noClipping':True,'peakRequests':abs(values).max(0).tolist(),
        'variants':variants,'bodyOutputs':sorted(BODY),'physicsMethod':'scale output Weight, not particle damping or direct mesh scaling',
        'scope':'same short MMD source; no smoothing, no arms, no expressions, no training',
        'limitations':'gains chosen for this calibration clip, not validation; non-body outputs labelled secondary may include face/other effects, NOT exclusively hair/cloth',
        'previousBaseline':str(previous),'sourceDuration':float(times[-1])})
    unity.write_spec(root/'signals.json',{'times':times.tolist(),'sourceDeltasDegrees':raw.tolist(),'requests':values.tolist()})
    curves=unity.write_spec(root/'curves.json',{'fps':30,'duration':float(times[-1]),'physics':True,
        'ids':ids,'channels':[{'values':values[:,i].tolist()} for i in range(6)]})
    checks=[]
    for label,body,secondary in variants:
        override=unity.write_spec(root/(label+'-physics.json'),attenuate(physics,body,secondary))
        unity.run('LabMotion.RunMotion',root/label,{'ACTION_LAB_MODEL':unity.model3_path(),'ACTION_LAB_CURVES':curves,
            'ACTION_LAB_PATCH':'','ACTION_LAB_LAYER_FRAME':'','ACTION_LAB_PHYSICS_OVERRIDE':override},root/(label+'.log'))
        frames=[json.loads((root/label/f'{i:04}.json').read_text(encoding='utf-8-sig')) for i in range(len(times))]
        applied=[{p['id']:p['value'] for p in f['applied']} for f in frames]
        checks.append({'variant':label,'maxPairing':max(f['pairingMaxDelta'] for f in frames),
            'bodyOutputRanges':{k:[min(r[k] for r in applied),max(r[k] for r in applied)] for k in sorted(BODY)}})
        unity.write_spec(root/'checks.json',checks);print(label,'captured',flush=True)

if __name__=='__main__':run(Path(sys.argv[1]),Path(sys.argv[2]))
