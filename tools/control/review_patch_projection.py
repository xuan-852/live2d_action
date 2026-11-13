"""Raw before/after frontal projection videos and independent SDK ratio checks."""
import sys,json,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image,ImageDraw
import imageio_ffmpeg
from tools.control.patch_projection import projection_guard,projected_shape_guard
from tools.expression.unity import write_spec


def read_case(root):
    p=json.loads((root/'protocol.json').read_text())
    rows=[json.loads(x) for x in (root/'track.jsonl').read_text().splitlines()]
    if len(rows)+1!=p['frames'] or not(root/'capture/complete.txt').exists():raise ValueError('Incomplete capture')
    ratios=[];guard_errors=[]
    for r in rows:
        f=json.loads((root/'capture'/f"{r['frame']:04}.json").read_text(encoding='utf-8-sig'))['wholeArms']
        actual=np.array(f['patchChain']).reshape(4,2);source=np.array(f['sourcePatchChain']).reshape(4,2)
        ratio=np.linalg.norm(np.diff(actual,axis=0),axis=1)/np.linalg.norm(np.diff(source,axis=0),axis=1)
        ratios.append(ratio)
        if f.get('projectionOnlyMeshAreaRatios'):
            floor=.30
            if p.get('palmPlane',{}).get('enabled'):
                # TargetIds order is frozen in RecoveredArmGeometry.cs.
                floor=[.36 if i in (3,4,5,6,7,8,9) else .30 for i in range(16)]
            expected=projected_shape_guard(f['rawMeshAreaRatios'],f['projectionOnlyMeshAreaRatios'],f['projectedAngularMeshAreaRatios'],f['projectionAreaFloors'],floor)
            guard_errors.append(float(np.max(abs(expected-f['meshAreaRatios']))))
        elif f.get('identityAngleMeshAreaRatios'):
            expected=projection_guard(f['rawMeshAreaRatios'],f['identityAngleMeshAreaRatios'],f['projectionAreaFloors'])
            guard_errors.append(float(np.max(abs(expected-f['meshAreaRatios']))))
    groups=[]
    for name in dict.fromkeys(r['stage'] for r in rows):
        idx=[i for i,r in enumerate(rows) if r['stage']==name];selected=[rows[i] for i in idx]
        measured=np.array([ratios[i] for i in idx]);new=all('projectionRequested' in r['patch'] for r in selected)
        group=dict(stage=name,frames=len(idx),actualMin=measured.min(0).tolist(),actualMax=measured.max(0).tolist(),
            wristRMS=float(np.sqrt(np.mean([r['patch']['wristError']**2 for r in selected]))),
            elbowRMS=float(np.sqrt(np.mean([r['patch']['elbowError']**2 for r in selected]))),
            palmRMS=float(np.sqrt(np.mean([r['patch']['palmErrorDeg']**2 for r in selected]))),
            positionHeadingMet=sum(r['patch']['wristError']<=.005 and r['patch']['elbowError']<=.005 and r['patch']['palmErrorDeg']<=4 for r in selected),
            failedFrames=[r['frame'] for r in selected if r['patch']['status']=='failed'],
            minRawArea=min(r['geometry'].get('rawMinTriangleAreaRatio',r['geometry']['minTriangleAreaRatio']) for r in selected),
            minGuardArea=min(r['geometry']['minTriangleAreaRatio'] for r in selected),
            reasons=sorted({reason for r in selected for reason in r['patch']['reasons']}))
        if new:
            requested=np.array([r['patch']['projectionRequested'] for r in selected]);error=measured-requested
            group.update(requestedMin=requested.min(0).tolist(),maxRatioError=np.max(abs(error),axis=0).tolist(),
                projectionMet=int(sum(np.max(abs(x))<=.03 for x in error)))
        groups.append(group)
    return rows,np.array(ratios),dict(root=str(root),stages=groups,guardFormulaMaxError=max(guard_errors,default=None),
        allFailedFrames=[r['frame'] for r in rows if r['patch']['status']=='failed'],
        maxModelResidual=max(r['patch']['metrics']['modelResidual'] for r in rows),
        maxima={k:max(r['patch']['metrics'][k] for r in rows) for k in rows[0]['patch']['metrics']},
        last15Hold=all(r['patch']['positionReached'] and r['patch']['terminalStateMet'] and r['patch'].get('projectionStateMet',True) and r['patch'].get('palmBasisStateMet',True) for r in rows[-15:]))


def compare(before,after,output,zoom=False):
    output.mkdir(parents=True,exist_ok=False)
    a,ar,asummary=read_case(before);b,br,bsummary=read_case(after)
    if len(a)!=len(b):raise ValueError('Different clocks')
    if any((x['time'],x.get('sourceTime'))!=(y['time'],y.get('sourceTime')) for x,y in zip(a,b)):raise ValueError('Retimed samples')
    crop=None
    if zoom:
        points=[]
        for root,rows in [(before,a),(after,b)]:
            for row in rows:
                f=json.loads((root/'capture'/f"{row['frame']:04}.json").read_text(encoding='utf-8-sig'))
                points.extend(np.array(f['patchNodeScreen']).reshape(4,2).tolist())
        points=np.array(points);low=points.min(0);high=points.max(0)
        size=min(1024,int(np.ceil(max(high-low)+160)));center=(low+high)/2
        start=np.clip(center-size/2,0,1024-size).astype(int)
        crop=(int(start[0]),int(start[1]),int(start[0]+size),int(start[1]+size))
    write_spec(output/'comparison.json',dict(before=asummary,after=bsummary,
        scope='same clock and event protocol; projection requests newly enabled. Feedback freeze targets can differ because actual poses differ.',fixedCrop=crop,visualAcceptance=False))
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x696','-r','30','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output/'comparison.mp4')]
    fmt=lambda v:'/'.join(f'{x:.2f}' for x in v)
    with subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE) as proc:
        for i,(old,new) in enumerate(zip(a,b)):
            canvas=Image.new('RGB',(1024,696),'#eeeeee');draw=ImageDraw.Draw(canvas)
            draw.text((8,8),f"FRONTAL SHORTENING | {new['time']:.3f}s | original 30Hz | {new['stage']}",fill='black')
            draw.text((8,28),'Upper / forearm / palm measured lengths relative to same-frame source mesh. Width1, no new wrist or texture.',fill='black')
            for j,(root,row,ratio,title) in enumerate(zip([before,after],[old,new],[ar[i],br[i]],['CHECKPOINT / fixed projection','CANDIDATE / controlled frontal lengths'])):
                draw.text((j*512+8,52),title,fill='black')
                with Image.open(root/'capture'/f"{row['frame']:04}.png") as im:
                    image=im.convert('RGB')
                    if crop is not None:image=image.crop(crop)
                    canvas.paste(image.resize((512,512)),(j*512,80))
                s=row['patch'];draw.text((j*512+8,604),f"ACTUAL length {fmt(ratio)} | {s['status']}",fill='black')
                if 'projectionRequested' in s:draw.text((j*512+8,624),'REQUEST length '+fmt(s['projectionRequested']),fill='black')
                draw.text((j*512+8,644),f"wrist {s['wristError']:.4f} | palm {s['palmErrorDeg']:.1f}deg",fill='black')
                draw.text((j*512+8,666),', '.join(s['reasons'])[:75] or 'within current checks',fill='#9b2424' if s['reasons'] else '#006633')
            proc.stdin.write(canvas.tobytes())
        proc.stdin.close();error=proc.stderr.read();code=proc.wait()
        if code:raise RuntimeError(error.decode(errors='replace'))
    print(output/'comparison.mp4')
    print(json.dumps(bsummary,indent=2))


if __name__=='__main__':compare(*map(Path,sys.argv[1:4]),zoom='--zoom' in sys.argv)
