"""Replay exact arm test with native distal foreground only; retain old captures."""
import sys,json,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.expression import unity

def run(root,source):
    root.mkdir(parents=True,exist_ok=False)
    shutil.copy2('preview/unity/WholeArmSession.cs',root/'frozen-WholeArmSession.cs')
    unity.write_spec(root/'protocol.json',{'source':str(source),'change':'nativeDistalFront=true only',
        'scope':'native ArtMesh231/232/233/234/235/236/238 foreground, preserve internal original order differences',
        'limitations':'explicit foreground test, not general depth-aware occlusion; hand can overlap face/clothes by design',
        'unchanged':'source time, all parameters, physics, patch deformation/projection/layers'})
    checks=[]
    for entry in json.loads((source/'results.json').read_text()):
        name=entry['source'];case=root/name;case.mkdir()
        curves=case/'curves.json';shutil.copy2(source/name/'with-arms-curves.json',curves)
        p=json.loads((source/name/'with-arms-patch.json').read_text());p['nativeDistalFront']=True
        patch=unity.write_spec(case/'patch.json',p)
        unity.run('LabMotion.RunMotion',case/'capture',{'ACTION_LAB_MODEL':unity.model3_path(),'ACTION_LAB_CURVES':curves,
            'ACTION_LAB_PATCH':patch,'ACTION_LAB_LAYER_FRAME':'','ACTION_LAB_PHYSICS_OVERRIDE':''},case/'capture.log')
        fs=[json.loads((case/'capture'/f'{i:04}.json').read_text(encoding='utf-8-sig')) for i in range(entry['frames'])]
        checks.append({'source':name,'frames':len(fs),'maxPairing':max(f['pairingMaxDelta'] for f in fs),
            'maxRestore':max(f['wholeArms']['restoredMeshError'] for f in fs),
            'nativeIds':fs[0]['wholeArms']['nativeDistalIds'],'orders':fs[0]['wholeArms']['nativeDistalSortingOrders']})
        unity.write_spec(root/'checks.json',checks);print(name,'captured',flush=True)

if __name__=='__main__':run(Path(sys.argv[1]),Path(sys.argv[2]))
