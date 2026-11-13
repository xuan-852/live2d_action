"""Full gesture candidate footage plus independent applied-context mesh audit."""
import json,sys,subprocess,ctypes as C
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image,ImageDraw
import imageio_ffmpeg
from tools.control.pilot import Plant
from tools.control.finger_control import FINGER_IDS
from tools.control.finger_capability_probe import AFFECTED
from tools.expression.unity import write_spec


def assess(root):
    sdk=root/'sdk';rows=[json.loads(x) for x in (sdk/'track.jsonl').read_text(encoding='utf-8').splitlines()]
    protocol=json.loads((root/'acceptance-protocol.json').read_text(encoding='utf-8'))
    expected=json.loads((sdk/'protocol.json').read_text(encoding='utf-8'))['frames']
    if not(sdk/'capture/complete.txt').exists() or len(rows)+1!=expected:raise ValueError('incomplete SDK capture')
    stream=json.loads((sdk/'request-stream.json').read_text(encoding='utf-8'))['frames']
    groups=[]
    for stage in protocol['stages']:
        selected=[r for r in rows if r['stage']==stage['name']];tail=selected[-15:]
        def arm_met(r):
            a=r['native'];return a['wristError']<=.005 and a['elbowError']<=.005 and a['palmErrorDeg']<=4
        groups.append(dict(stage=stage['name'],holdWindowApplicable=stage['frames']>=45,
            last15FingerSettled=all(r['nativeFingers']['settled'] for r in tail),
            last15ArmMet=all(arm_met(r) for r in tail),
            fingerFailures=[r['frame'] for r in selected if r['nativeFingers']['status']=='failed'],
            armFailures=[r['frame'] for r in selected if r['native']['status']=='failed'],
            limitedFrames=sum(r['nativeFingers']['status']=='limited' for r in selected),
            deadlineMisses=sum('finger-deadline-missed' in r['nativeFingers']['reasons'] for r in selected),
            maxFinalFingerError=max(max(abs(x) for x in r['nativeFingers']['residual']) for r in tail)))
    p=Plant();m=p.model
    counts=np.ctypeslib.as_array(m.api('GetDrawableIndexCounts',C.POINTER(C.c_int),[C.c_void_p])(m.ptr),shape=(m.n,))
    ptr=m.api('GetDrawableIndices',C.POINTER(C.POINTER(C.c_ushort)),[C.c_void_p])(m.ptr)
    triangles={name:np.ctypeslib.as_array(ptr[m.names.index(name)],shape=(int(counts[m.names.index(name)]),)).reshape(-1,3).copy() for name in AFFECTED}
    def areas(v,tri):
        a=v[tri[:,1]]-v[tri[:,0]];b=v[tri[:,2]]-v[tri[:,0]];return a[:,0]*b[:,1]-a[:,1]*b[:,0]
    audit=[]
    for r in rows:
        f=json.loads((sdk/'capture'/f"{r['frame']:04}.json").read_text(encoding='utf-8-sig'))
        applied={q['id']:q['value'] for q in f['applied']}
        m.set(applied|{k:0. for k in FINGER_IDS});reference={n:m.mesh[m.names.index(n)].copy() for n in AFFECTED}
        m.set(applied);minimum=1.;flips=0;centroid_error=0.
        for name,tri in triangles.items():
            v=m.mesh[m.names.index(name)];base=areas(reference[name],tri);valid=abs(base)>1e-9
            ratio=areas(v,tri)[valid]/base[valid];minimum=min(minimum,float(min(ratio)));flips+=int(sum(ratio<0))
            actual=next(g for g in r['nativeFingerMeshes'] if g['id']==name)
            centroid_error=max(centroid_error,float(np.linalg.norm(v.mean(0)-[actual['cx'],actual['cy']])))
        audit.append(dict(frame=r['frame'],minFingerAreaAtSameArmContext=minimum,flips=flips,coreVsFinalSDKCentroid=centroid_error))
    summary=dict(stages=groups,frames=expected,
        maxSpeed=max(r['nativeFingers']['metrics']['maxSpeed'] for r in rows),
        maxAcceleration=max(r['nativeFingers']['metrics']['maxAcceleration'] for r in rows),
        maxAppliedCommandResidual=max(r['nativeFingers']['metrics']['commandResidual'] for r in rows),
        allFingerGoalsMatchCurrentRequest=all(r['nativeFingers']['requested']==stream[r['frame']-1]['nativeFingerGoal']['values'] for r in rows),
        finalSettled=all(r['nativeFingers']['settled'] for r in rows[-15:]),
        minFingerArea=min(a['minFingerAreaAtSameArmContext'] for a in audit),
        maxFlippedTriangles=max(a['flips'] for a in audit),
        maxCoreVsFinalSDKCentroid=max(a['coreVsFinalSDKCentroid'] for a in audit),
        meshScope='Core replay at full actual SDK context vs final SDK geometry centroids; not final mesh triangle export',
        patchFingerExecution=False,visualAcceptance=False)
    write_spec(root/'gesture-assessment.json',summary);write_spec(root/'applied-mesh-audit.json',dict(frames=audit))
    return rows,summary


def crop(frame,side,span=160):
    nodes=np.array(frame[side+'NodeScreen']).reshape(4,2);center=(nodes[2]+nodes[3])/2
    return tuple(int(x) for x in [center[0]-span/2,center[1]-span/2,center[0]+span/2,center[1]+span/2])


def review(root):
    rows,summary=assess(root);sdk=root/'sdk'
    output=root/'gesture-review.mp4'
    if output.exists():raise FileExistsError(output)
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1536x672','-r','30','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    contacts=[]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE) as proc:
        for row in rows:
            frame=json.loads((sdk/'capture'/f"{row['frame']:04}.json").read_text(encoding='utf-8-sig'))
            canvas=Image.new('RGB',(1536,672),'#eeeeee');d=ImageDraw.Draw(canvas)
            d.text((10,7),f"GESTURE CANDIDATE | {row['stage']} | {row['time']:.2f}s | actual SDK feedback / physics OFF",fill='black')
            d.text((10,27),'Candidate labels are unverified meanings. Hand close-ups follow wrist by translation only; fixed 160px span. No palm scaling.',fill='black')
            for col,title in enumerate(['FULL RAW FRAME / ARM ROUTE','NATIVE / ACTUAL FINGER EXECUTION','PATCH / STATIC FINGER REFERENCE']):d.text((col*512+8,52),title,fill='black')
            with Image.open(sdk/'capture'/f"{row['frame']:04}.png") as source:
                raw=source.convert('RGB');canvas.paste(raw.resize((512,512)),(0,76))
                for col,side in [(1,'native'),(2,'patch')]:canvas.paste(raw.crop(crop(frame,side)).resize((512,512)),(col*512,76))
                if row['stage'].startswith('shape/') and (not contacts or contacts[-1][0]!=row['stage']):contacts.append([row['stage'],None])
                if contacts and contacts[-1][0]==row['stage']:contacts[-1][1]=raw.crop(crop(frame,'native')).resize((256,256))
            f=row['nativeFingers'];fmt=lambda v:' / '.join(f'{x:.1f}' for x in v)
            d.text((8,603),'FINGER request 102/103/105/106/107: '+fmt(f['requested']),fill='black')
            d.text((8,623),'SDK applied: '+fmt(f['applied'])+f" | {f['status']} | max error {max(abs(x) for x in f['residual']):.3f}",fill='black')
            d.text((8,645),', '.join(f['reasons']) or 'within current finger checks; gesture naturalness pending',fill='#992222' if f['reasons'] else '#006633')
            proc.stdin.write(canvas.tobytes())
        proc.stdin.close();error=proc.stderr.read();code=proc.wait()
        if code:raise RuntimeError(error.decode(errors='replace'))
    sheet=Image.new('RGB',(768,580),'#eeeeee');draw=ImageDraw.Draw(sheet)
    for i,(name,im) in enumerate(contacts[:6]):
        x=i%3*256;y=i//3*290;draw.text((x+6,y+7),name.replace('shape/','')+' candidate',fill='black');sheet.paste(im,(x,y+30))
    sheet.save(root/'candidate-contact.png')
    print(output);print(json.dumps(summary,indent=2))


if __name__=='__main__':review(Path(sys.argv[1]))
