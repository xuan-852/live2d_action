"""Shared frozen mapping: five-source suitability and continuous join A/B."""
import sys,json,hashlib,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from tools.control.mmd_virtual_tracker import signals
from tools.control.calibrate_virtual_tracker import calibrate
from tools.expression import unity

IDS=['ParamAngleX','ParamAngleY','ParamAngleZ','ParamBodyAngleX','ParamBodyAngleY','ParamBodyAngleZ']
LIMITS=np.array([25,25,20,8,8,8])

def bridge(a,b,va,vb,frames=24):
    u=np.arange(1,frames+1)[:,None]/(frames+1);duration=(frames+1)/30
    return (2*u**3-3*u**2+1)*a+(u**3-2*u**2+u)*duration*va+(-2*u**3+3*u**2)*b+(u**3-u**2)*duration*vb

def join(clips,smooth):
    parts=[];labels=[]
    # Deliberately mid-clip cuts test non-neutral joining; full clips tested separately.
    for repeat in range(3):
        for name,v in clips:
            v=v[:max(3,round(len(v)*.8))]
            if parts:
                old=parts[-1];n=24
                transition=bridge(old[-1],v[0],(old[-1]-old[-2])*30,(v[1]-v[0])*30,n) if smooth else np.repeat(old[-1][None],n,axis=0)
                parts.append(transition);labels.extend([{'source':name,'kind':'bridge' if smooth else 'hold-then-cut','repeat':repeat}]*n)
            parts.append(v);labels.extend([{'source':name,'kind':'source-prefix','repeat':repeat}]*len(v))
    # Same total duration; explicit settling interval, physics keeps advancing.
    tail=bridge(parts[-1][-1],np.zeros(6),np.zeros(6),np.zeros(6),30)
    parts.extend([tail,np.zeros((60,6))]);labels.extend([{'source':'neutral','kind':'settle','repeat':0}]*90)
    return np.concatenate(parts),labels

def run(root,sources):
    root.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,root/'frozen-experiments.py')
    clips=[];manifest=[]
    for p in sorted(sources.glob('*/map.json')):
        d=json.loads(p.read_text(encoding='utf-8-sig'));t,raw,_,_=signals(d);requested=calibrate(raw)
        clipped=abs(requested)>LIMITS;v=np.clip(requested,-LIMITS,LIMITS)
        clips.append((p.parent.name,v));shutil.copy2(p,root/(p.parent.name+'-source.json'))
        manifest.append({'id':p.parent.name,'source':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
            'motionSHA':d['sourceSha256'],'duration':float(t[-1]),'frames':len(t),'clippedFramesPerChannel':clipped.sum(0).tolist(),
            'quality':'pending user review','scope':'head/body only, no arms/morph/global translation; existing Euler extraction'})
    if len(clips)!=5:raise ValueError('Expected five unique source clips')
    unity.write_spec(root/'protocol.json',{'mapping':'same gains as 20261112 calibration, frozen across five sources; hard safety clipping reported not hidden',
        'experiments':['five complete source suitability','three-pass prefix joins A/B'],
        'continuous':'single SDK session per variant, physics state never reset at boundaries; 24-frame bridge or equal-duration last-request hold then hard cut',
        'limitations':'only five source motions, not 10-20; repeated prefixes are not independent samples; Hermite overshoot can hit safety bounds; no naturalness auto-pass',
        'trained':False,'fps':30,'sources':manifest})
    tasks=[('sample-'+name,v,[{'source':name,'kind':'complete-source','repeat':0}]*len(v)) for name,v in clips]
    for smooth in (False,True):
        v,labels=join(clips,smooth);tasks.append(('continuous-smooth' if smooth else 'continuous-cut',v,labels))
    checks=[]
    for name,v,labels in tasks:
        pre=v.copy();v=np.clip(v,-LIMITS,LIMITS)
        unity.write_spec(root/(name+'-timeline.json'),{'labels':labels,'requests':v.tolist(),'preSafetyClipCount':int((pre!=v).sum())})
        curves=unity.write_spec(root/(name+'-curves.json'),{'fps':30,'duration':(len(v)-1)/30,'physics':True,
            'ids':IDS,'channels':[{'values':v[:,j].tolist()} for j in range(6)]})
        unity.run('LabMotion.RunMotion',root/name,{'ACTION_LAB_MODEL':unity.model3_path(),'ACTION_LAB_CURVES':curves,
            'ACTION_LAB_PATCH':'','ACTION_LAB_LAYER_FRAME':'','ACTION_LAB_PHYSICS_OVERRIDE':''},root/(name+'.log'))
        fs=[json.loads((root/name/f'{i:04}.json').read_text(encoding='utf-8-sig')) for i in range(len(v))]
        checks.append({'name':name,'frames':len(v),'duration':(len(v)-1)/30,'maxPairing':max(f['pairingMaxDelta'] for f in fs),
            'maxRequestStep':float(abs(np.diff(v,axis=0)).max()),'review':'pending'})
        unity.write_spec(root/'checks.json',checks);print(name,'captured',flush=True)

if __name__=='__main__':run(Path(sys.argv[1]),Path(sys.argv[2]))
