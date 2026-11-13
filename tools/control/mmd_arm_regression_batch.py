"""Frozen current controller across existing MMD sources, no per-case tuning."""
import sys,json,hashlib,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.control.palm_plane import bases_from_reference
from tools.control.mmd_timed_controller import execute


def run(root,sources,canonical_path):
    root.mkdir(parents=True,exist_ok=False)
    canonical=json.loads(canonical_path.read_text(encoding='utf-8-sig'))
    before=root/'before';before.mkdir()
    for name in ('stateful_arm.py','timed_arm.py','mmd_timed_controller.py','palm_plane.py','patch_projection.py','validate_stateful_arm.py'):
        shutil.copy2(Path(__file__).parent/name,before/name)
    jobs=[]
    for name in ('yayoi-right-hand-rise1','makoto-guard1','yukiho-mouth1','azusa-bow1'):
        path=sources/name/'map.json';doc=json.loads(path.read_text(encoding='utf-8-sig'))
        matches=[c for c in canonical['clips'] if c['sourceSha256']==doc['sourceSha256'] and c['file']==doc['sourceFile']]
        if len(matches)!=1:raise ValueError('Unique source provenance missing: '+name)
        clip=matches[0]
        reference={'kind':'mmd-fk','frames':[f['points'] for f in clip['frames']]}
        bases_from_reference(reference,doc)  # Refuse mismatched geometry before any capture.
        jobs.append((name,path,reference))
    protocol={'controller':'current working-tree snapshot, same settings for every source',
        'cases':[n for n,_,_ in jobs],'canonicalSHA':hashlib.sha256(canonical_path.read_bytes()).hexdigest(),
        'sourceTime':'all original frames; standard existing acquire/return only, no retiming',
        'options':['native-five','patch-projection','patch-rooted','matched-FK-palm-reference'],
        'scope':'cross-motion regression, previously exposed sources; no new training or semantic labels',
        'acceptance':'same existing evaluator, all limited/failed frames retained; visual pending'}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    status=[]
    for name,path,reference in jobs:
        ref=root/(name+'-fk.json');ref.write_text(json.dumps(reference))
        try:
            execute(root/name,path,patch_projection=True,patch_rooted=True,palm_reference=ref)
            status.append({'case':name,'capture':'completed','visual':'pending'})
        except Exception as exc:
            status.append({'case':name,'capture':'failed','error':repr(exc)})
        (root/'batch-status.json').write_text(json.dumps(status,indent=2));print(status[-1],flush=True)

if __name__=='__main__':run(*(Path(p) for p in sys.argv[1:4]))
