"""Full five-source enter/play/exit/settle in one persistent SDK physics session."""
import sys,json,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from tools.control.mmd_two_experiments import bridge,IDS,LIMITS
from tools.expression import unity

def timeline(clips):
    parts=[];labels=[]
    def add(v,name,phase):
        parts.append(v);labels.extend([{'source':name,'phase':phase}]*len(v))
    for name,v in clips:
        zero=np.zeros(6)
        add(np.zeros((30,6)),name,'stable-before')
        add(bridge(zero,v[0],zero,(v[1]-v[0])*30),name,'enter')
        add(v,name,'source')
        add(bridge(v[-1],zero,(v[-1]-v[-2])*30,zero),name,'exit')
        add(np.zeros((60,6)),name,'settle')
    return np.concatenate(parts),labels

def run(root,previous):
    root.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,root/'frozen-lifecycle.py')
    clips=[]
    for p in sorted(previous.glob('sample-*-timeline.json')):
        clips.append((p.name.removeprefix('sample-').removesuffix('-timeline.json'),np.array(json.loads(p.read_text())['requests'])))
    if len(clips)!=5:raise ValueError('Five complete previous signals required')
    v,labels=timeline(clips);clipcount=int((abs(v)>LIMITS).sum());v=np.clip(v,-LIMITS,LIMITS)
    unity.write_spec(root/'protocol.json',{'previous':str(previous),'frames':len(v),'fps':30,
        'session':'single model/physics instance, no reset across clips','mapping':'reuse exact previous complete-source requests',
        'phases':'30 stable,24 Hermite enter,complete source,24 Hermite exit,60 settle',
        'extraSafetyClippedValues':clipcount,'limitations':'Euler/source/mapping limitations unchanged; nominal Hermite continuity can be affected by safety clipping; not semantic behavior/training',
        'sourceFiles':[n for n,_ in clips]})
    unity.write_spec(root/'timeline.json',{'labels':labels,'requests':v.tolist()})
    curves=unity.write_spec(root/'curves.json',{'fps':30,'duration':(len(v)-1)/30,'physics':True,
        'ids':IDS,'channels':[{'values':v[:,j].tolist()} for j in range(6)]})
    unity.run('LabMotion.RunMotion',root/'capture',{'ACTION_LAB_MODEL':unity.model3_path(),'ACTION_LAB_CURVES':curves,
        'ACTION_LAB_PATCH':'','ACTION_LAB_LAYER_FRAME':'','ACTION_LAB_PHYSICS_OVERRIDE':''},root/'capture.log')
    frames=[json.loads((root/'capture'/f'{i:04}.json').read_text(encoding='utf-8-sig')) for i in range(len(v))]
    actual=np.array([[p['value'] for p in f['applied']] for f in frames])
    boundaries=[]
    for i in range(1,len(labels)):
        if labels[i]!=labels[i-1]:boundaries.append({'frame':i,'time':i/30,'from':labels[i-1],'to':labels[i],
            'maxRequestDelta':float(abs(v[i]-v[i-1]).max()),'maxAppliedParameterDelta':float(abs(actual[i]-actual[i-1]).max())})
    unity.write_spec(root/'checks.json',{'frames':len(v),'duration':(len(v)-1)/30,
        'maxPairing':max(f['pairingMaxDelta'] for f in frames),'boundaries':boundaries,
        'note':'parameter deltas are not whole-image continuity certification'})
    print(root/'checks.json')

if __name__=='__main__':run(Path(sys.argv[1]),Path(sys.argv[2]))
