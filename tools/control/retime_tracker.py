"""Single-play return timing ablation; recapture SDK at uniform action times."""
import sys,json,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from tools.expression import unity

def time_map(t,end,mode,pivot=.5,stretch=2):
    t=np.asarray(t,float)
    if mode=='original':return np.minimum(t,end)
    duration=(end-pivot)*stretch
    u=np.clip((t-pivot)/duration,0,1)
    if mode=='slow-return':h=u
    elif mode=='ease-stop':
        a=stretch
        # ds/dt=1 at transition, zero at final stop; monotonic cubic.
        h=a*u+(3-2*a)*u*u+(a-2)*u*u*u
    else:raise ValueError(mode)
    return np.where(t<=pivot,t,pivot+(end-pivot)*h)

def run(root,previous):
    root.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,root/'frozen-retime.py')
    data=json.loads((previous/'signals.json').read_text());source=np.array(data['times']);values=np.array(data['requests']);end=source[-1]
    ids=['ParamAngleX','ParamAngleY','ParamAngleZ','ParamBodyAngleX','ParamBodyAngleY','ParamBodyAngleZ']
    protocol={'source':str(previous),'pivotSeconds':.5,'returnDurationMultiplier':2,
        'fixed':'identical calibrated amplitudes and original model physics, no smoothing of values',
        'mapping':'source-time warp; 30Hz new uniform action times, SDK physics dt unchanged',
        'scope':'0.5s peak is a diagnostic breakpoint, not certified semantic return onset; no looping',
        'diagnosis':{'maxRequestSpeedTime':float(source[np.argmax(np.linalg.norm(np.diff(values,axis=0),axis=1))]),
        'firstLastRequestDelta':(values[-1]-values[0]).tolist()},'variants':['original','slow-return','ease-stop']}
    unity.write_spec(root/'protocol.json',protocol)
    for mode in protocol['variants']:
        duration=end if mode=='original' else .5+2*(end-.5)
        t=np.arange(round(duration*30)+1)/30;s=time_map(t,end,mode)
        v=np.stack([np.interp(s,source,values[:,j]) for j in range(6)],axis=1)
        curves=unity.write_spec(root/(mode+'-curves.json'),{'fps':30,'duration':float(t[-1]),'physics':True,
            'ids':ids,'channels':[{'values':v[:,j].tolist()} for j in range(6)]})
        unity.write_spec(root/(mode+'-times.json'),{'actionTimes':t.tolist(),'sourceTimes':s.tolist(),'requests':v.tolist()})
        unity.run('LabMotion.RunMotion',root/mode,{'ACTION_LAB_MODEL':unity.model3_path(),'ACTION_LAB_CURVES':curves,
            'ACTION_LAB_PATCH':'','ACTION_LAB_LAYER_FRAME':'','ACTION_LAB_PHYSICS_OVERRIDE':''},root/(mode+'.log'))
        fs=[json.loads((root/mode/f'{i:04}.json').read_text(encoding='utf-8-sig')) for i in range(len(t))]
        unity.write_spec(root/(mode+'-checks.json'),{'frames':len(fs),'maxPairing':max(f['pairingMaxDelta'] for f in fs),
            'maxRequestStep':float(abs(np.diff(v,axis=0)).max()),'completed':True})
        print(mode,'captured',flush=True)

if __name__=='__main__':run(Path(sys.argv[1]),Path(sys.argv[2]))
