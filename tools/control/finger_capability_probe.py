"""Targeted existing finger-channel inventory, NOT an expressive gesture controller.

Only the five visible basic-hand channels are driven. Proximal arm, physics,
patch projection and finger geometry stay unchanged. Preserve every SDK frame.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import json,sys,shutil,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import ctypes as C
import numpy as np
from PIL import Image,ImageDraw
import imageio_ffmpeg
from tools.control.pilot import Plant
from tools.control.validate_stateful_arm import APPROVED,pose
from tools.whole.batch import markers
from tools.expression import unity

VISIBLE_KEYS=('Param102','Param103','Param105','Param106','Param107')
AFFECTED=('ArtMesh230','ArtMesh231','ArtMesh233','ArtMesh234','ArtMesh235')


def make_probe():
    samples=[];stages=[]
    def append(name,values):
        start=len(samples);samples.extend(values)
        stages.append(dict(name=name,start=start,end=len(samples),frames=len(values)))
    append('neutral',[np.zeros(5) for _ in range(30)])
    # Fixed envelope before capture: .5s ramp, .5s negative hold, 1s sweep,
    # .5s positive hold, .5s return. +/-20 lies inside authored +/-30.
    wave=np.r_[np.linspace(0,-20,15),np.full(15,-20),np.linspace(-20,20,30),np.full(15,20),np.linspace(20,0,15)]
    for i,key in enumerate(VISIBLE_KEYS):append(key+'-isolated',[np.eye(5)[i]*v for v in wave])
    for sign in (-1,1):
        w=np.r_[np.linspace(0,sign*20,30),np.full(30,sign*20),np.linspace(sign*20,0,30)]
        append('combined-'+('negative' if sign<0 else 'positive'),[np.full(5,v) for v in w])
    append('neutral-return',[np.zeros(5) for _ in range(30)])
    return np.array(samples),stages


def core_audit(plant,samples):
    m=plant.model;m.set({});base=m.vertices()
    counts=np.ctypeslib.as_array(m.api('GetDrawableIndexCounts',C.POINTER(C.c_int),[C.c_void_p])(m.ptr),shape=(m.n,))
    ptr=m.api('GetDrawableIndices',C.POINTER(C.POINTER(C.c_ushort)),[C.c_void_p])(m.ptr)
    triangles={name:np.ctypeslib.as_array(ptr[m.names.index(name)],shape=(int(counts[m.names.index(name)]),)).reshape(-1,3).copy() for name in AFFECTED}
    cross=lambda a,b:a[:,0]*b[:,1]-a[:,1]*b[:,0]
    areas=lambda v,ids:cross(v[ids[:,1]]-v[ids[:,0]],v[ids[:,2]]-v[ids[:,0]])
    before={name:areas(base[m.names.index(name)],ids) for name,ids in triangles.items()}
    previous=base;rows=[]
    for frame,q in enumerate(samples):
        m.set(dict(zip(VISIBLE_KEYS,map(float,q))));minimum=1.;flips=0;step=0.;patch_displacement=0.
        for name,ids in triangles.items():
            i=m.names.index(name);valid=abs(before[name])>1e-9
            ratios=areas(m.mesh[i],ids)[valid]/before[name][valid]
            minimum=min(minimum,float(min(ratios)));flips+=int(sum(ratios<0))
            step=max(step,float(np.max(np.linalg.norm(m.mesh[i]-previous[i],axis=1))))
        for n in ('ArtMesh253','ArtMesh254','ArtMesh255','ArtMesh256','ArtMesh257','ArtMesh258','ArtMesh259'):
            i=m.names.index(n);patch_displacement=max(patch_displacement,float(np.max(np.linalg.norm(m.mesh[i]-base[i],axis=1))))
        rows.append(dict(frame=frame,minNativeFingerTriangleRatio=minimum,flippedNativeFingerTriangles=flips,
            maxNativeFingerVertexStep=step,patchCoreDisplacement=patch_displacement))
        previous=m.vertices()
    m.set({});return rows


def review(root,samples,stages,audit):
    capture=root/'sdk';frames=[json.loads((capture/f'{i:04}.json').read_text(encoding='utf-8-sig')) for i in range(len(samples))]
    if not(capture/'complete.txt').exists():raise ValueError('incomplete SDK capture')
    first=frames[0];crops={}
    for side in ('native','patch'):
        p=np.array(first[side+'NodeScreen']).reshape(4,2)[2:];center=p.mean(0)
        # Same fixed physical/pixel span for both hands; no per-frame rescaling.
        crops[side]=tuple(int(x) for x in [center[0]-110,center[1]-110,center[0]+110,center[1]+110])
    metrics=[]
    for i,f in enumerate(frames):
        requested={x['id']:x['value'] for x in f['requested']};applied={x['id']:x['value'] for x in f['applied']}
        metrics.append(dict(frame=i,maxAppliedError=max(abs(applied[k]-samples[i,j]) for j,k in enumerate(VISIBLE_KEYS)),
            maxPairing=f['pairingMaxDelta'],**{k:v for k,v in audit[i].items() if k!='frame'}))
    unity.write_spec(root/'assessment.json',dict(frames=len(frames),
        maxAppliedError=max(x['maxAppliedError'] for x in metrics),maxPairing=max(x['maxPairing'] for x in metrics),
        minFingerArea=min(x['minNativeFingerTriangleRatio'] for x in metrics),
        maxFlips=max(x['flippedNativeFingerTriangles'] for x in metrics),
        maxFingerVertexStep=max(x['maxNativeFingerVertexStep'] for x in metrics),
        maxPatchCoreDisplacement=max(x['patchCoreDisplacement'] for x in metrics),
        geometryScope='Core per-vertex audit at requested curve samples; SDK PNG/JSON independently verifies applied parameters and rendering; not a final-render vertex export',
        visualAcceptance=False,patchFingerExecution=False,crops=crops))
    unity.write_spec(root/'per-frame-audit.json',metrics)
    for filename,review in [('finger-probe-review.mp4',True),('finger-probe-raw.mp4',False)]:
        output=root/filename
        if output.exists():raise FileExistsError(output)
        size='1280x720' if review else '1024x1024'
        cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',size,'-r','30','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
        with subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE) as proc:
            for i,q in enumerate(samples):
                with Image.open(capture/f'{i:04}.png') as source:
                    raw=source.convert('RGB')
                    if review:
                        image=Image.new('RGB',(1280,720),'#eeeeee');d=ImageDraw.Draw(image)
                        stage=next(s['name'] for s in stages if s['start']<=i<s['end'])
                        d.text((10,6),f'EXISTING FINGER CHANNELS | {stage} | {i/30:.2f}s | physics OFF / arm fixed / no palm scaling',fill='black')
                        d.text((10,25),'102 / 103 / 105 / 106 / 107 actual '+ ' / '.join(f"{frames[i]['applied'][next(j for j,p in enumerate(frames[i]['applied']) if p['id']==k)]['value']:.1f}" for k in VISIBLE_KEYS),fill='black')
                        for col,side in enumerate(('native','patch')):
                            d.text((col*640+10,47),'NATIVE / authored finger response' if side=='native' else 'PATCH / STATIC REFERENCE: finger channels not connected',fill='black')
                            image.paste(raw.crop(crops[side]).resize((640,640)),(col*640,80))
                    else:image=raw
                    proc.stdin.write(image.tobytes())
            proc.stdin.close();error=proc.stderr.read();code=proc.wait()
            if code:raise RuntimeError(error.decode(errors='replace'))
        print(output)


def execute(root):
    root.mkdir(parents=True,exist_ok=False);samples,stages=make_probe();plant=Plant();m=plant.model
    audit=core_audit(plant,samples)
    unity.write_spec(root/'core-audit.json',audit)
    spec=unity.write_spec(root/'curves.json',dict(fps=30,duration=(len(samples)-1)/30,physics=False,excludeX2=False,
        ids=list(VISIBLE_KEYS),channels=[dict(values=samples[:,j].tolist()) for j in range(5)]))
    patch=unity.write_spec(root/'patch.json',dict(nativeMarkers=markers(m,plant.anchors),correctedPatchNodes=json.loads((APPROVED/'protocol.json').read_text())['newNodes'],
        nativeDistalFront=True,rootedUpper=True,poses=[pose([0,0,0]) for _ in samples]))
    # Flatten approved nodes for existing JsonUtility contract.
    p=json.loads(patch.read_text());p['correctedPatchNodes']=np.array(p['correctedPatchNodes']).ravel().tolist();unity.write_spec(patch,p)
    unity.write_spec(root/'protocol.json',dict(stages=stages,fps=30,frames=len(samples),range=[-20,20],physics=False,
        controlled=list(VISIBLE_KEYS),notDriven=['Param104 is anger; not a finger','Param111-115 target currently hidden alternate fingers; Param110 also couples visible forearm pieces'],
        handProjection='identity; existing shape retained',patch='static control reference; no existing SDK response to these channels',
        criteria=dict(appliedError=1e-4,pairing=1e-6,proximal='all non-finger requested parameters stay at default',
            scope='raw authored parameter ability; not integrated gesture control or naturalness acceptance'),visualAcceptance=False))
    frozen=root/'implementation';frozen.mkdir();shutil.copy2(__file__,frozen/Path(__file__).name)
    for name in unity.SCRIPTS:shutil.copy2(unity.ROOT/'preview/unity'/name,frozen/name)
    unity.run('LabMotion.RunMotion',root/'sdk',dict(ACTION_LAB_MODEL=unity.model3_path(),ACTION_LAB_CURVES=spec,ACTION_LAB_PATCH=patch,
        ACTION_LAB_PALM_PLANE='off',ACTION_LAB_PATCH_PROJECTION='off',ACTION_LAB_GEOMETRY_FEEDBACK='off',ACTION_LAB_PHYSICS_OVERRIDE='',ACTION_LAB_LAYER_FRAME=''),root/'capture.log',timeout=1200)
    review(root,samples,stages,audit)


if __name__=='__main__':execute(Path(sys.argv[1]))
