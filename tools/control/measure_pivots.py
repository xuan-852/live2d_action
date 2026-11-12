"""Identify effective rotation centres from Core response, not visual guesses."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from tools.control.pilot import Plant


def fit_rotation(x,y):
    x=np.asarray(x,float);y=np.asarray(y,float)
    xc=x.mean(0);yc=y.mean(0)
    u,_,vt=np.linalg.svd((x-xc).T@(y-yc))
    correction=np.eye(2);correction[-1,-1]=np.linalg.det(u@vt)
    rotation=u@correction@vt
    translation=yc-xc@rotation
    angle=float(np.degrees(np.arctan2(rotation[0,1],rotation[0,0])))
    matrix=np.eye(2)-rotation.T
    if abs(angle)<.05:return None
    pivot=np.linalg.solve(matrix,translation)
    residual=float(np.sqrt(np.mean((x@rotation+translation-y)**2)))
    movement=float(np.sqrt(np.mean((x-y)**2)))
    return {'pivot':pivot.tolist(),'angleDegrees':angle,'rigidRMS':residual,
            'movementRMS':movement,'relativeRigidResidual':residual/max(movement,1e-12)}


def run(root):
    root.mkdir(parents=True,exist_ok=False);plant=Plant();model=plant.model
    model.set({});base=model.vertices();opacity=model.opacity.copy()
    params=['Param94','Param97','Param99','ParamAngleZ','ParamBodyAngleZ2',
            'ParamAngleX','ParamAngleY','ParamBodyAngleX2','ParamBodyAngleY2']
    amplitudes=[-6.,-3.,-1.,1.,3.,6.]
    protocol={'mocSHA':model.sha,'parameters':params,'amplitudes':amplitudes,
        'baseline':'Core parameter defaults; each intervention resets every parameter; SDK physics OFF',
        'fit':'proper 2D rigid least squares; pivot from (I-R^T)c=t',
        'qualification':'all six responses: abs angle>=.05deg; movementRMS>1e-5; rigid residual/movement<.02; centre scatter<.003 model units',
        'validation':'centre median from +/-1,+/-3; +/-6 response checks held out from centre estimation',
        'scope':'effective mesh rotation centres, not recovered editor hierarchy or anatomical joints; no invented 3D axes',
        'patch':'Core interventions do not measure added patch axes; those require patch execution identification separately'}
    (root/'protocol.json').write_text(json.dumps(protocol,indent=2))
    (root/'frozen-measure-pivots.py').write_bytes(Path(__file__).read_bytes())
    results=[]
    raw={'baseline_'+str(i):v for i,v in enumerate(base)}
    for param in params:
        samples=[]
        for value in amplitudes:
            model.set({param:value});vertices=model.vertices();samples.append(vertices)
            for index,vertices_mesh in enumerate(vertices):raw[f'{param}_{value}_{index}']=vertices_mesh
        meshes=[]
        for index,x in enumerate(base):
            if opacity[index]<.5 or len(x)<3:continue
            fits=[fit_rotation(x,y[index]) for y in samples]
            if any(f is None for f in fits):continue
            if min(f['movementRMS'] for f in fits)<1e-5:continue
            centres=np.array([f['pivot'] for f in fits]);centre=np.median(centres[1:5],axis=0)
            scatter=float(np.linalg.norm(centres-centre,axis=1).max())
            worst=max(f['relativeRigidResidual'] for f in fits)
            accepted=worst<.02 and scatter<.003
            meshes.append({'mesh':model.names[index],'centre':centre.tolist(),'maxCentreDeviation':scatter,
                'maxRelativeRigidResidual':worst,'qualified':bool(accepted),'samples':fits})
        meshes.sort(key=lambda m:(not m['qualified'],m['maxRelativeRigidResidual'],m['maxCentreDeviation']))
        results.append({'parameter':param,'qualifiedMeshes':sum(m['qualified'] for m in meshes),'meshes':meshes})
    model.set({})
    np.savez_compressed(root/'raw-mesh-responses.npz',**raw)
    (root/'results.json').write_text(json.dumps(results,indent=2))
    lines=['# Core effective rotation-centre measurements','',
        'Physics disabled. Mesh-specific effective centres; NOT anatomical nodes or full deformer hierarchy.','']
    for r in results:
        lines.append(f"## {r['parameter']}: {r['qualifiedMeshes']} qualified meshes")
        for m in r['meshes'][:6]:
            lines.append(f"- {m['mesh']}: qualified={m['qualified']}, centre={m['centre']}, scatter={m['maxCentreDeviation']:.6g}, relative rigid residual={m['maxRelativeRigidResidual']:.6g}")
    (root/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines))


if __name__=='__main__':run(Path(sys.argv[1]))
