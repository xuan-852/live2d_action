"""Audit final hand mesh affine, preserve every frame, compare both source ambiguities."""
import json,sys,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image,ImageDraw
import imageio_ffmpeg
from tools.expression.unity import write_spec


def mesh_basis(frame):
    f=frame['wholeArms'];a=np.array(f['palmAffineReadback']).reshape(2,2)
    source=np.array(f['sourcePatchChain'],dtype=float).reshape(4,2)
    u=source[3]-source[2];u/=np.linalg.norm(u);v=np.array([-u[1],u[0]])
    drawn=a@u;length=np.linalg.norm(drawn);d=drawn/length;e=np.array([-d[1],d[0]])
    return np.array([length,e@a@v,d@a@v])


def assess(root):
    rows=[json.loads(x) for x in (root/'track.jsonl').read_text().splitlines()]
    protocol=json.loads((root/'protocol.json').read_text())
    if len(rows)+1!=protocol['frames'] or not(root/'capture/complete.txt').exists():raise ValueError('incomplete capture')
    values=[];command_errors=[]
    for row in rows:
        f=json.loads((root/'capture'/f"{row['frame']:04}.json").read_text(encoding='utf-8-sig'))
        basis=mesh_basis(f);values.append(basis)
        q=f['wholeArms']['patchActuators']
        expected=[1+q[5]/100,1+q[6]/100,q[7]/100]
        command_errors.append(max(abs(basis-expected)))
    groups=[]
    for stage in dict.fromkeys(r['stage'] for r in rows):
        indices=[i for i,r in enumerate(rows) if r['stage']==stage];selected=[rows[i] for i in indices]
        errors=[max(abs(values[i][1:]-rows[i]['patch']['palmBasisRequested'])) for i in indices]
        groups.append(dict(stage=stage,frames=len(indices),meshBasisMet=int(sum(x<=.03 for x in errors)),
            last15BasisMet=all(x<=.03 for x in errors[-15:]),
            wristRMS=float(np.sqrt(np.mean([r['patch']['wristError']**2 for r in selected]))),
            palmRMS=float(np.sqrt(np.mean([r['patch']['palmErrorDeg']**2 for r in selected]))),
            failedFrames=[r['frame'] for r in selected if r['patch']['status']=='failed'],
            finalMeshBasis=values[indices[-1]].tolist(),reasons=sorted({why for r in selected for why in r['patch']['reasons']})))
    return rows,values,dict(stages=groups,maxCommandVsFinalMeshBasisError=float(max(command_errors)),
        maxMeshStretch=float(max(np.linalg.norm([[b[0],b[2]],[0,b[1]]],2) for b in values)),
        minRaw=min(r['geometry']['rawMinTriangleAreaRatio'] for r in rows),
        minGuard=min(r['geometry']['minTriangleAreaRatio'] for r in rows),
        flipped=max(r['geometry']['flippedTriangles'] for r in rows),
        failedFrames=[r['frame'] for r in rows if r['patch']['status']=='failed'],
        maxima={k:max(r['patch']['metrics'][k] for r in rows) for k in rows[0]['patch']['metrics']},visualAcceptance=False)


def compare(old,positive,negative,out):
    out.mkdir(parents=True,exist_ok=False)
    a=[json.loads(x) for x in (old/'track.jsonl').read_text().splitlines()]
    b,basis_b,summary_b=assess(positive);c,basis_c,summary_c=assess(negative)
    if not len(a)==len(b)==len(c):raise ValueError('different frame counts')
    if any((x['time'],x.get('sourceTime'))!=(y['time'],y.get('sourceTime')) or (x['time'],x.get('sourceTime'))!=(z['time'],z.get('sourceTime')) for x,y,z in zip(a,b,c)):raise ValueError('retimed frames')
    write_spec(out/'assessment.json',dict(positive=summary_b,negative=summary_c,
        meaning='same clock and wrist/elbow route; palm transverse basis newly enabled. Shear sign is not observable from sidecar.',visualAcceptance=False))
    # One fixed crop for the entire movie; includes shoulder, elbow, wrist and fingers.
    points=[]
    for root,rows in [(old,a),(positive,b),(negative,c)]:
        for row in rows:
            f=json.loads((root/'capture'/f"{row['frame']:04}.json").read_text(encoding='utf-8-sig'))
            points.extend(np.array(f['patchNodeScreen']).reshape(4,2))
    points=np.array(points);size=min(1024,int(np.ceil(max(points.max(0)-points.min(0))+180)))
    start=np.clip((points.max(0)+points.min(0))/2-size/2,0,1024-size).astype(int)
    crop=(int(start[0]),int(start[1]),int(start[0]+size),int(start[1]+size))
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-f','rawvideo','-pixel_format','rgb24','-video_size','1536x650','-framerate','30','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'comparison.mp4')]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE) as proc:
        for i,(x,y,z) in enumerate(zip(a,b,c)):
            im=Image.new('RGB',(1536,650),'#eeeeee');draw=ImageDraw.Draw(im)
            draw.text((8,8),f"PALM PLANE | {x['time']:.3f}s | {x['stage']} | original 30Hz | no retiming",fill='black')
            draw.text((8,28),'Two explicit shear branches: source sidecar lacks the sign. Neither is certified as full source hand orientation.',fill='black')
            for col,(root,row,title) in enumerate(zip([old,positive,negative],[x,y,z],['OLD / length only','PLANE / positive shear branch','PLANE / negative shear branch'])):
                draw.text((col*512+8,52),title,fill='black')
                with Image.open(root/'capture'/f"{row['frame']:04}.png") as raw:im.paste(raw.convert('RGB').crop(crop).resize((512,512)),(col*512,76))
                if col:
                    vals=(basis_b if col==1 else basis_c)[i]
                    draw.text((col*512+8,596),'MESH length/width/shear '+' / '.join(f'{v:.2f}' for v in vals),fill='black')
                draw.text((col*512+8,616),row['patch']['status']+' | '+', '.join(row['patch']['reasons'])[:66],fill='#992222')
            proc.stdin.write(im.tobytes())
        proc.stdin.close();error=proc.stderr.read();code=proc.wait()
        if code:raise RuntimeError(error.decode(errors='replace'))
    print(out/'comparison.mp4')


def compare_pair(old,new,out):
    out.mkdir(parents=True,exist_ok=False)
    rows,bases,summary=assess(new)
    before=[json.loads(x) for x in (old/'track.jsonl').read_text().splitlines()]
    if len(before)!=len(rows) or any((x['time'],x.get('sourceTime'))!=(y['time'],y.get('sourceTime')) for x,y in zip(before,rows)):
        raise ValueError('different clocks')
    write_spec(out/'assessment.json',summary)
    points=[]
    for root,track in [(old,before),(new,rows)]:
        for row in track:
            frame=json.loads((root/'capture'/f"{row['frame']:04}.json").read_text(encoding='utf-8-sig'))
            points.extend(np.array(frame['patchNodeScreen']).reshape(4,2))
    points=np.array(points);size=min(1024,int(np.ceil(max(points.max(0)-points.min(0))+180)))
    start=np.clip((points.max(0)+points.min(0))/2-size/2,0,1024-size).astype(int)
    crop=tuple(map(int,[*start,*(start+size)]))
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-f','rawvideo','-pixel_format','rgb24','-video_size','1024x680','-framerate','30','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'comparison.mp4')]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE) as proc:
        for i,(a,b) in enumerate(zip(before,rows)):
            canvas=Image.new('RGB',(1024,680),'#eeeeee');draw=ImageDraw.Draw(canvas)
            draw.text((8,8),f"PALM PLANE | {b['time']:.3f}s | {b['stage']} | original clock / wrist route",fill='black')
            draw.text((8,28),'Source basis from verified 3D FK. Single-face near-edge limits remain visible. Fixed crop throughout.',fill='black')
            for col,(root,row,title) in enumerate([(old,a,'BEFORE / longitudinal compression'),(new,b,'CANDIDATE / signed palm plane, independent cuff')]):
                draw.text((col*512+8,52),title,fill='black')
                with Image.open(root/'capture'/f"{row['frame']:04}.png") as raw:
                    canvas.paste(raw.convert('RGB').crop(crop).resize((512,512)),(col*512,76))
                p=row['patch'];draw.text((col*512+8,596),f"{p['status']} | wrist {p['wristError']:.4f} | palm {p['palmErrorDeg']:.1f}deg",fill='black')
                draw.text((col*512+8,646),', '.join(p['reasons'])[:78],fill='#992222')
            draw.text((520,616),'MESH length / width / shear '+' / '.join(f'{v:.2f}' for v in bases[i]),fill='black')
            draw.text((520,631),'REQUEST width / shear '+' / '.join(f'{v:.2f}' for v in b['patch']['palmBasisRequested']),fill='black')
            proc.stdin.write(canvas.tobytes())
        proc.stdin.close();error=proc.stderr.read();code=proc.wait()
        if code:raise RuntimeError(error.decode(errors='replace'))
    print(out/'comparison.mp4')


if __name__=='__main__':
    if '--pair' in sys.argv:compare_pair(*map(Path,sys.argv[1:4]))
    else:compare(*map(Path,sys.argv[1:5]))
