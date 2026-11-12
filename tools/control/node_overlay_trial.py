"""Random parameter interventions with frozen bound-node overlay, no target control."""
import sys,json,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from tools.control.pilot import Plant
from tools.whole.batch import markers
from tools.expression import unity


def stimuli(seed=11809):
    rng=np.random.default_rng(seed);values=np.zeros((361,6))
    bounds=np.array([18,10,18,24,25,18])
    for section in range(4):
        targets=rng.uniform(-1,1,(4,6))*bounds
        targets[0]=0;targets[-1]=0
        mask=np.ones(6) if section==3 else np.array([int(j%3==section) for j in range(6)])
        for k in range(90):
            t=k/30;lo=min(int(t),2);w=t-lo;w=w*w*(3-2*w)
            values[section*90+k]=((1-w)*targets[lo]+w*targets[lo+1])*mask
    return values


def run(root):
    root.mkdir(parents=True,exist_ok=False);plant=Plant();values=stimuli()
    before=root/'before';before.mkdir()
    for file in ['tools/control/node_overlay_trial.py','tools/whole/recovered/anchors.py',
                 'preview/unity/LabMotion.cs','preview/unity/WholeArmSession.cs','preview/unity/RecoveredShoulderConnection.cs','preview/unity/RecoveredArmGeometry.cs']:
        shutil.copy2(file,before/Path(file).name)
    spec={'seed':11809,'mocSHA':plant.model.sha,'frames':361,'fps':30,'physics':False,
        'sections':['shoulder only','elbow only','wrist only','combined'],
        'nativeChannels':['Param94','Param97','Param99'],'patchChannels':['shoulder','elbow','wrist'],
        'nodeDefinition':'frozen existing material bindings from anchors.py and ShoulderConnection, no per-test fitting',
        'purpose':'visually test whether bound nodes coincide with actual rotation joints, NOT demonstrate target reaching',
        'limitations':'same-frame material tracking alone is not proof of pivot validity; full image review required; no node auto-pass',
        'nativeBinding':plant.anchors.evidence}
    unity.write_spec(root/'protocol.json',spec)
    unity.write_spec(root/'stimuli.json',{'values':values.tolist()})
    poses=[{'shoulder':float(v[3]),'elbow':float(v[4]),'wrist':float(v[5]),
        'projection':{'upperLength':1,'forearmLength':1,'palmLength':1,'palmWidth':1,'palmShear':0},
        'girdle':{'rootX':0,'rootY':0,'socketX':0,'socketY':0}} for v in values]
    patch=unity.write_spec(root/'patch.json',{'poses':poses,'nativeMarkers':markers(plant.model,plant.anchors),'layerPolicy':'legacy-z'})
    curves=unity.write_spec(root/'curves.json',{'fps':30,'duration':12,'physics':False,'ids':spec['nativeChannels'],
        'channels':[{'values':values[:,j].tolist()} for j in range(3)]})
    unity.run('LabMotion.RunMotion',root/'capture',{'ACTION_LAB_MODEL':unity.model3_path(),'ACTION_LAB_CURVES':curves,
        'ACTION_LAB_PATCH':patch,'ACTION_LAB_LAYER_FRAME':''},root/'capture.log')
    fs=[json.loads((root/'capture'/f'{i:04}.json').read_text(encoding='utf-8-sig')) for i in range(361)]
    unity.write_spec(root/'checks.json',{'frames':len(fs),'maxPairing':max(f['pairingMaxDelta'] for f in fs),
        'maxRestore':max(f['wholeArms']['restoredMeshError'] for f in fs),'maxFlips':max(f['wholeArms']['flippedTriangles'] for f in fs),
        'jointValidity':'not automatically certified; review overlay'})
    print(root/'checks.json')

if __name__=='__main__':run(Path(sys.argv[1]))
