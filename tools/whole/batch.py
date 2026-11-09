"""Restore whole-motion evidence and compare one shared native-chain refinement.

No imports from the old worktree at runtime: required inputs are explicitly copied
into the new run. Baseline retains legacy compiled dual-arm tracks, not acceptance.
Candidate uses the recovered full-chain Core solver, never the hand-centroid grid.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import imageio_ffmpeg
from tools.expression import unity
from tools.whole.recovered.core_model import Model
from tools.whole.recovered.anchors import Anchors
from tools.whole.recovered.native import NativeArm

CASES = ['novel-01','novel-08','novel-12','novel-14','novel-16','novel-19','novel-21','novel-23']


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sample_document(doc, fps=30):
    keys=doc['frames']; times=np.array([f['t'] for f in keys])
    grid=np.arange(int(np.ceil(times[-1]*fps))+1)/fps
    curves={k:np.interp(grid,times,[f['native'][k] for f in keys]).tolist() for k in keys[0]['native']}
    poses=[]
    for t in grid:
        hi=min(max(int(np.searchsorted(times,t,side='right')),1),len(keys)-1)
        a,b=keys[hi-1],keys[hi]; w=float(np.clip((t-a['t'])/(b['t']-a['t']),0,1))
        p={k:a[k]+w*(b[k]-a[k]) for k in ('shoulder','elbow','wrist')}
        for obj in ('projection','girdle'):
            p[obj]={k:a[obj][k]+w*(b[obj][k]-a[obj][k]) for k in a[obj]}
        poses.append(p)
    return grid,curves,poses


def markers(model, anchors):
    result=[]
    for name,(mesh,indices,weights) in anchors.weights.items():
        result.append(dict(id=name,mesh=model.names[mesh],indices=indices.tolist(),weights=weights.tolist()))
    result.append(dict(id='palm',mesh=model.names[147],indices=list(range(len(model.mesh[147]))),
                       weights=[1/len(model.mesh[147])]*len(model.mesh[147])))
    return result


def refine(doc, mapping, model, anchors, native):
    """Same target upper/forearm headings, projection and rest-palm policy for every case.

    Previous pose seeds branch continuity; original output is a seed only. No gesture labels.
    Residuals retained. This does not guarantee natural route or feasible speed.
    """
    import copy
    out=copy.deepcopy(doc); rows=mapping['sides']['anatomicalLeft']['frames']
    diagnostics=[]; prior=None
    base_limits={k:list(v) for k,v in native.limits.items()}
    speeds={'Param94':120.,'Param95':90.,'Param97':120.,'Param98':90.,'Param99':120.}
    class Supported:
        support={}
        def set(self, values): model.set(self.support | values)
    supported=Supported(); native.model=supported
    for i,row in enumerate(out['frames']):
        if i==0:
            prior={k:row['native'][k] for k in ('Param94','Param95','Param97','Param98','Param99','Param100')}
            continue
        # Existing compiler puts two neutral/approach keys before source keys.
        phase=min(max(i-2,0),len(rows)-1)
        target=rows[phase]['target']
        supported.support={k:v for k,v in row['native'].items() if k not in ('Param94','Param95','Param97','Param98','Param99','Param100')}
        prior=prior or {k:row['native'][k] for k in ('Param94','Param95','Param97','Param98','Param99','Param100')}
        if i<2 or i>=len(rows)+2:
            measured=native.measure({k:row['native'][k] for k in ('Param94','Param95','Param97','Param98','Param99','Param100')})
            target={'upperHeadingDeg':measured['upperHeadingDeg'],'forearmHeadingDeg':measured['forearmHeadingDeg'],
                'upperProjectedRatio':measured['upperRatio'],'forearmProjectedRatio':measured['forearmRatio'],
                'bendUnwrappedDeg':measured['bendDeg'],'palmRelativeDeg':measured['palmRelativeDeg']}
        dt=row['t']-out['frames'][i-1]['t']
        for k,v in base_limits.items():
            rate=speeds[k]
            native.limits[k]=[max(v[0],prior[k]-rate*dt),min(v[1],prior[k]+rate*dt)]
        native.limits['Param99']=[max(-30,prior['Param99']-speeds['Param99']*dt),
                                  min(30,prior['Param99']+speeds['Param99']*dt)]
        solved=native.solve(target,prior)
        prior=solved['pose']; row['native'].update(prior)
        diagnostics.append({'key':i,'target':target,'status':solved['status'],
                            'residuals':solved['residuals'],'reasons':solved['reasons']})
    native.limits=base_limits
    return out,diagnostics


def summarize(folder):
    files=sorted(folder.glob('[0-9][0-9][0-9][0-9].json'))
    frames=[read(p) for p in files]; ar=[f['wholeArms'] for f in frames]
    native=np.array([a['nativeChain'] for a in ar]); patch=np.array([a['patchChain'] for a in ar])
    return {'frames':len(frames),'maxPrePatchCoreMeshDelta':max(f['pairingMaxDelta'] for f in frames),
            'patchRestoreError':max(a['restoredMeshError'] for a in ar),
            'flippedTrianglesMax':max(a['flippedTriangles'] for a in ar),
            'minTriangleAreaRatio':min(a['minTriangleAreaRatio'] for a in ar),
            'nativeMaxMarkerStep':float(np.max(np.linalg.norm(np.diff(native.reshape(-1,4,2),axis=0),axis=2))),
            'patchMaxMarkerStep':float(np.max(np.linalg.norm(np.diff(patch.reshape(-1,4,2),axis=0),axis=2))),
            'nativeWristRange':np.ptp(native[:,4:6],axis=0).tolist(),
            'patchWristRange':np.ptp(patch[:,4:6],axis=0).tolist(),
            'humanAccepted':False,'semanticPreservation':'unknown','naturalness':'pending',
            'scope':'actual rendered-mesh material markers; not anatomical certification'}


def execute(run, only=None):
    inputs=run/'inputs'; inputs.mkdir(exist_ok=True)
    (run/'logs').mkdir(exist_ok=True)
    protocol=run/'protocol.json'
    if not protocol.exists():
        old=Path('E:/FuXuanLab/runs/20261030-generated-observation-04/compiled')
        provenance=[]
        for name in CASES:
            for filename in (name+'.json','map.json',name+'.json.reference.json'):
                src=old/name/filename; dst=inputs/name/filename; dst.parent.mkdir(exist_ok=True)
                raw=src.read_bytes(); dst.write_bytes(raw)
                provenance.append({'path':str(src),'copy':str(dst.relative_to(run)),
                                   'sha256':hashlib.sha256(raw).hexdigest()})
        unity.write_spec(protocol,{'cases':CASES,'heldout':['novel-12','novel-14','novel-21'],
            'frozenAt':datetime.datetime.now().astimezone().isoformat(),'inputs':provenance,
            'baseline':'recovered whole compiled tracks and minimal mesh executor',
            'candidate':'same patch/body/time, recovered native whole-chain solve against Core',
            'unknown':['head orientation','finger pose','palm plane'],
            'intervention':'no per-gesture branches; no human acceptance inherited'})
    m=Model(unity.model3_path().parent,(unity.PROJECT/'Assets/Live2D/Cubism/Plugins/Windows/x86_64/Live2DCubismCore.dll').resolve())
    anchors=Anchors(m); cal=read(Path('tools/whole/recovered/calibration.json'))
    native=NativeArm(m,anchors,cal['sides']['anatomicalLeft']); marker=markers(m,anchors)
    unity.write_spec(run/'anchor-revalidation.json',{'modelSha':m.sha,'evidence':anchors.evidence,
        'neutral':{k:v.tolist() for k,v in anchors.read().items()},'note':'material markers remeasured; not anatomical ground truth'})
    results=read(run/'results.json') if (run/'results.json').exists() else {}
    for name in only or CASES:
        baseline=read(inputs/name/(name+'.json')); mapping=read(inputs/name/'map.json')
        candidate,diagnostics=refine(baseline,mapping,m,anchors,native)
        unity.write_spec(run/(name+'-native-solve.json'),diagnostics)
        results.setdefault(name,{})
        for label,doc in (('restored',baseline),('chain',candidate)):
            grid,curves,poses=sample_document(doc)
            spec=unity.write_spec(run/(name+'-'+label+'-curves.json'),{
                'fps':30,'duration':float(grid[-1]),'physics':True,'excludeX2':False,
                'ids':list(curves),'channels':[{'values':v} for v in curves.values()]})
            patch=unity.write_spec(run/(name+'-'+label+'-patch.json'),{'poses':poses,'nativeMarkers':marker})
            folder=run/(name+'-'+label)
            if not (folder/'complete.txt').exists():
                unity.run('LabMotion.RunMotion',folder,{'ACTION_LAB_MODEL':unity.model3_path(),
                    'ACTION_LAB_CURVES':spec,'ACTION_LAB_PATCH':patch},run/'logs'/(name+'-'+label+'.log'))
            results[name][label]=summarize(folder)
            unity.write_spec(run/'results.json',results)
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-filter_complex_threads','1',
            '-framerate','30','-i',str(run/(name+'-restored')/'%04d.png'),
            '-framerate','30','-i',str(run/(name+'-chain')/'%04d.png'),
            '-filter_complex','[0:v]scale=512:512[a];[1:v]scale=512:512[b];[a][b]hstack=inputs=2[v]',
            '-map','[v]','-c:v','libx264','-threads','1','-crf','18','-pix_fmt','yuv420p',
            '-movflags','+faststart',str(run/(name+'-whole-compare.mp4'))],
            check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        print(name,json.dumps(results[name]),flush=True)
    unity.write_spec(run/'stage-time.json',{'time':datetime.datetime.now().astimezone().isoformat(),'completed':list(results)})


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('run',type=Path); parser.add_argument('--cases',nargs='+')
    args=parser.parse_args(); execute(args.run,args.cases)
