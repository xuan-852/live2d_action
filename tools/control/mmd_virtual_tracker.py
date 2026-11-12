"""Existing MMD body/head-local signals as a virtual performer, no training."""
import sys,json,shutil,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from tools.expression import unity

def signals(document):
    rows=document['coordinated'];times=np.array([f['seconds'] for f in document['sides']['anatomicalLeft']['frames']])
    if len(rows)!=len(times) or not np.allclose(np.diff(times),1/30,atol=1e-6):raise ValueError('Expected aligned 30Hz source')
    data=np.array([[r[g][k] for g in ('headLocal','body') for k in ('yaw','pitch','roll')] for r in rows])
    if not np.isfinite(data).all():raise ValueError('Nonfinite orientation')
    # Relative to initial pose, no Euler angle addition of head/body.
    data=np.degrees(np.unwrap(np.radians(data),axis=0));data-=data[0]
    mapped=data*np.array([1,-1,-1,1/3,-1/3,-1/3])
    limits=np.array([25,25,20,8,8,8])
    return times,data,np.clip(mapped,-limits,limits),np.abs(mapped)>limits

def run(root,source):
    root.mkdir(parents=True,exist_ok=False);shutil.copy2(source,root/'source-map.json');shutil.copy2(__file__,root/'frozen-driver.py')
    document=json.loads(source.read_text(encoding='utf-8-sig'));times,raw,mapped,clipped=signals(document)
    ids=['ParamAngleX','ParamAngleY','ParamAngleZ','ParamBodyAngleX','ParamBodyAngleY','ParamBodyAngleZ']
    unity.write_spec(root/'protocol.json',{'source':str(source),'sourceSHA':hashlib.sha256(source.read_bytes()).hexdigest(),
        'motionSource':document['sourceFile'],'motionSHA':document['sourceSha256'],'mocSHA':document['modelSha256'],
        'frames':len(times),'sourceDuration':float(times[-1]),'fps':30,'trained':False,
        'mapping':'headLocal delta -> AngleXYZ; body delta /3 -> BodyAngleXYZ; Y/Z sign reversed, candidate not calibrated',
        'limitations':'reuses historical extracted Euler orientation, FK not re-certified; no headWorld addition; no morphs, no arm motion; body/global translation omitted',
        'clippedSamplesPerChannel':clipped.sum(0).tolist(),'variants':['head-only','head-body-physics'],
        'purpose':'MMD as repeatable virtual performer for basic head/body liveliness, NOT autonomous behavior'})
    unity.write_spec(root/'signals.json',{'times':times.tolist(),'sourceDeltasDegrees':raw.tolist(),'appliedRequests':mapped.tolist(),'ids':ids})
    checks=[]
    for label in ('head-only','head-body-physics'):
        values=mapped.copy()
        if label=='head-only':values[:,3:]=0
        curves=unity.write_spec(root/(label+'-curves.json'),{'fps':30,'duration':float(times[-1]),'physics':label!='head-only',
            'ids':ids,'channels':[{'values':values[:,j].tolist()} for j in range(6)]})
        unity.run('LabMotion.RunMotion',root/label,{'ACTION_LAB_MODEL':unity.model3_path(),'ACTION_LAB_CURVES':curves,
            'ACTION_LAB_PATCH':'','ACTION_LAB_LAYER_FRAME':''},root/(label+'.log'))
        frames=[json.loads((root/label/f'{i:04}.json').read_text(encoding='utf-8-sig')) for i in range(len(times))]
        applied=[{p['id']:p['value'] for p in f['applied']} for f in frames]
        checks.append({'variant':label,'maxPairing':max(f['pairingMaxDelta'] for f in frames),
            'bodyOutputRanges':{k:[min(r[k] for r in applied),max(r[k] for r in applied)] for k in ('ParamBodyAngleX2','ParamBodyAngleY2','ParamBodyAngleZ2')}})
        unity.write_spec(root/'checks.json',checks);print(label,'captured',flush=True)

if __name__=='__main__':run(Path(sys.argv[1]),Path(sys.argv[2]))
