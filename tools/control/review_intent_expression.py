"""Separate task contracts, final SDK geometry, and human expression review."""
import sys,json,subprocess,ctypes as C
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
from tools.control.pilot import Plant
from tools.control.finger_capability_probe import AFFECTED
from tools.control.finger_control import FINGER_IDS
from tools.control.intent_expression_trial import read_markers
from tools.expression.unity import write_spec


def angle_to_target(row,target):
    p=row['taskMarkers'];axis=np.array(p['indexTip'])-p['indexBase'];toward=np.array(target)-p['indexBase']
    return float(np.degrees(np.arccos(np.clip(axis@toward/max(np.linalg.norm(axis)*np.linalg.norm(toward),1e-12),-1,1))))


def assess(root):
    sdk=root/'sdk';rows=[json.loads(x) for x in (sdk/'track.jsonl').read_text(encoding='utf-8').splitlines()]
    protocol=json.loads((root/'acceptance-protocol.json').read_text(encoding='utf-8'))
    frames=json.loads((sdk/'protocol.json').read_text(encoding='utf-8'))['frames']
    if not(sdk/'capture/complete.txt').exists() or len(rows)+1!=frames:raise ValueError('incomplete SDK capture')
    fixed=protocol['fixed'];markers=json.loads((root/'task-markers.json').read_text(encoding='utf-8'))
    model=Plant().model;maximum_marker_error=0.;eye_offsets={}
    counts=np.ctypeslib.as_array(model.api('GetDrawableIndexCounts',C.POINTER(C.c_int),[C.c_void_p])(model.ptr),shape=(model.n,))
    ptr=model.api('GetDrawableIndices',C.POINTER(C.POINTER(C.c_ushort)),[C.c_void_p])(model.ptr)
    triangles={name:np.ctypeslib.as_array(ptr[model.names.index(name)],shape=(int(counts[model.names.index(name)]),)).reshape(-1,3).copy() for name in AFFECTED}
    def areas(v,tri):
        a=v[tri[:,1]]-v[tri[:,0]];b=v[tri[:,2]]-v[tri[:,0]];return a[:,0]*b[:,1]-a[:,1]*b[:,0]
    min_finger_area=1.;finger_flips=0;finger_centroid_error=0.
    for row in rows:
        frame=json.loads((sdk/'capture'/f"{row['frame']:04}.json").read_text(encoding='utf-8-sig'))
        applied={x['id']:x['value'] for x in frame['applied']};model.set(applied)
        replay=read_markers(model,markers)
        maximum_marker_error=max(maximum_marker_error,max(float(np.linalg.norm(replay[k]-row['taskMarkers'][k])) for k in replay))
        actual_fingers={name:model.mesh[model.names.index(name)].copy() for name in AFFECTED}
        for name,v in actual_fingers.items():
            g=next(g for g in row['nativeFingerMeshes'] if g['id']==name)
            finger_centroid_error=max(finger_centroid_error,float(np.linalg.norm(v.mean(0)-[g['cx'],g['cy']])))
        model.set(applied|{key:0. for key in FINGER_IDS})
        for name,tri in triangles.items():
            neutral_area=areas(model.mesh[model.names.index(name)],tri);valid=abs(neutral_area)>1e-9
            ratio=areas(actual_fingers[name],tri)[valid]/neutral_area[valid]
            min_finger_area=min(min_finger_area,float(min(ratio)));finger_flips=max(finger_flips,int(sum(ratio<0)))
        model.set(applied|dict(ParamEyeBallX=0.,ParamEyeBallY=0.));neutral=read_markers(model,markers)
        eye_offsets[row['frame']]=np.mean([np.array(row['taskMarkers'][k])-neutral[k] for k in ('irisLeft','irisRight')],axis=0)
    holds=[]
    for phase in protocol['phases']:
        if phase['phase'] not in ('express','rest'):continue
        selected=[r for r in rows if r['stage']==phase['task']+'/'+phase['phase']][-15:]
        holds.append(dict(stage=phase['task']+'/'+phase['phase'],
            arm=all(r['native']['positionReached'] and r['native']['terminalStateMet'] for r in selected),
            finger=all(r['nativeFingers']['settled'] for r in selected),gaze=all(r['gaze']['settled'] for r in selected),
            maxWristError=max(r['native']['wristError'] for r in selected)))
    point=[r for r in rows if r['stage']=='point/express'][-15:]
    near=[r for r in rows if r['stage']=='near-head/express'][-15:]
    angles=[angle_to_target(r,fixed['pointTarget']) for r in point]
    gaps=[float(np.linalg.norm(np.array(r['native']['actual'])[3]-r['taskMarkers']['headSide'])) for r in near]
    max_gap_error=max(float(np.linalg.norm(np.array(r['native']['actual'])[3]-r['taskMarkers']['headSide']-fixed['nearHeadPalmOffset'])) for r in near)
    stream=json.loads((sdk/'request-stream.json').read_text(encoding='utf-8'))['frames']
    ownership=all(r['nativeFingers']['requested']==stream[r['frame']-1]['nativeFingerGoal']['values'] and r['gaze']['requested']==stream[r['frame']-1]['gazeGoal']['values'] and r['intent']==stream[r['frame']-1]['intent'] for r in rows)
    out=dict(frames=frames,variant=protocol['variant'],holds=holds,
        failedFrames={side:[r['frame'] for r in rows if r[side]['status']=='failed'] for side in ('native','patch','nativeFingers','gaze')},
        maxPointingProxyAngle=max(angles),pointingProxyMet=all(x<=6 for x in angles),
        nearHeadGapRange=[min(gaps),max(gaps)],nearHeadGapMet=all(.02<=x<=.09 for x in gaps),maxNearHeadOffsetError=max_gap_error,
        eyeOffsetDuringPoint=np.mean([eye_offsets[r['frame']] for r in point],axis=0).tolist(),
        pointEyeScreenDirectionMet=all(eye_offsets[r['frame']][0]>0 and eye_offsets[r['frame']][1]<0 for r in point),
        maxCoreVsFinalSDKMarkerError=maximum_marker_error,currentIntentAndGoalOwnership=ownership,
        minFingerAreaAtActualArmContext=min_finger_area,maxNativeFingerFlips=finger_flips,maxFingerCoreVsSDKCentroid=finger_centroid_error,
        fingerSpeed=max(r['nativeFingers']['metrics']['maxSpeed'] for r in rows),
        fingerAcceleration=max(r['nativeFingers']['metrics']['maxAcceleration'] for r in rows),
        gazeSpeed=max(r['gaze']['metrics']['maxSpeed'] for r in rows),gazeAcceleration=max(r['gaze']['metrics']['maxAcceleration'] for r in rows),
        nativeNodeSpeed=max(r['native']['metrics']['maxNodeSpeed'] for r in rows),
        nativeNodeAcceleration=max(r['native']['metrics']['maxNodeAcceleration'] for r in rows),
        minimumPatchArea=min(r['geometry']['minTriangleAreaRatio'] for r in rows),
        maxPairing=max(r['pairing'] for r in rows),maxRestore=max(r['geometry']['restoredMeshError'] for r in rows),
        scope='controlled 2D point/phase proxies; not 3D gaze, collision/contact, semantic recognition or learned policy',visualAcceptance=False)
    write_spec(root/'expression-assessment.json',out)
    return rows,out


def affine(frame):
    nodes=np.array(frame['wholeArms']['nativeChain']).reshape(4,2);screen=np.array(frame['nativeNodeScreen']).reshape(4,2)
    return np.linalg.lstsq(np.c_[nodes,np.ones(4)],screen,rcond=None)[0]


def point_screen(p,matrix,scale=1.):return tuple(map(float,np.r_[p,1.]@matrix*scale))


def local_crop(center,span):return tuple(map(int,[center[0]-span/2,center[1]-span/2,center[0]+span/2,center[1]+span/2]))


def target_overlay(draw,row,matrix,offset=(0,0),scale=1.):
    if row['intent']['task']=='point':
        x,y=point_screen(row['intent']['pointTarget'],matrix,scale);x+=offset[0];y+=offset[1]
        draw.ellipse((x-8,y-8,x+8,y+8),outline='#ffcc44',width=2);draw.text((x-3,y-5),'A',fill='#ffcc44')


def review(root):
    rows,summary=assess(root);sdk=root/'sdk';output=root/'expression-review.mp4'
    if output.exists():raise FileExistsError(output)
    first=json.loads((sdk/'capture/0000.json').read_text(encoding='utf-8-sig'));matrix=affine(first)
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1536x704','-r','30','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    snapshots={}
    with subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE) as proc:
        for row in rows:
            frame=json.loads((sdk/'capture'/f"{row['frame']:04}.json").read_text(encoding='utf-8-sig'))
            canvas=Image.new('RGB',(1536,704),'#eeeeee');d=ImageDraw.Draw(canvas)
            d.text((10,7),f"EXPRESSION SCENARIO | {row['stage']} | {row['time']:.2f}s | {summary['variant']} | physics OFF",fill='black')
            d.text((10,28),'New authored tasks, not learned intent generation. Fixed-size translated close-ups. A = frozen scene target; head-side pause; contact/collision not modeled.',fill='black')
            for col,title in enumerate(['ACTUAL BODY / SCENE TARGET A OVERLAY','ACTUAL FACE / HEAD AND EYES','ACTUAL NATIVE HAND / FINGER POSE']):d.text((col*512+8,53),title,fill='black')
            p=row['taskMarkers'];face=np.mean([p['irisLeft'],p['irisRight']],axis=0)+[0,.025]
            hand=np.mean(np.array(row['native']['actual'])[2:],axis=0)
            with Image.open(sdk/'capture'/f"{row['frame']:04}.png") as source:
                raw=source.convert('RGB');canvas.paste(raw.resize((512,512)),(0,76))
                for col,center,span in [(1,face,170),(2,hand,160)]:
                    screen=np.r_[center,1.]@matrix;canvas.paste(raw.crop(local_crop(screen,span)).resize((512,512)),(col*512,76))
            target_overlay(d,row,matrix,offset=(0,76),scale=.5)
            if row['intent']['phase']=='express':snapshots[row['intent']['task']]=canvas.copy()
            d.text((8,603),f"HAND {row['intent']['hand']} | {row['nativeFingers']['status']} | GAZE {row['gaze']['status']} | ARM {row['native']['status']}",fill='black')
            detail='authored eye input '+str([round(x,2) for x in row['gaze']['applied']])+'; not a calibrated gaze ray'
            if row['intent']['task']=='point':detail+=f" | pointing proxy {angle_to_target(row,row['intent']['pointTarget']):.2f}deg"
            if row['intent']['task']=='near-head':detail+=f" | palm/head-side gap {np.linalg.norm(np.array(row['native']['actual'])[3]-p['headSide']):.4f}"
            d.text((8,625),detail,fill='black')
            d.text((8,647),'REQUEST fingers '+' / '.join(f'{x:.0f}' for x in row['nativeFingers']['requested'])+' | APPLIED '+' / '.join(f'{x:.1f}' for x in row['nativeFingers']['applied']),fill='black')
            reasons=row['native']['reasons']+row['nativeFingers']['reasons']+row['gaze']['reasons']
            d.text((8,673),', '.join(reasons)[:200] or 'within current execution checks; expression meaning/naturalness needs visual review',fill='#992222' if reasons else '#006633')
            proc.stdin.write(canvas.tobytes())
        proc.stdin.close();error=proc.stderr.read();code=proc.wait()
        if code:raise RuntimeError(error.decode(errors='replace'))
    for task,im in snapshots.items():im.save(root/(task+'-review.png'))
    print(output)


def compare(before,after,out):
    out.mkdir(parents=True,exist_ok=False);a,sa=assess(before);b,sb=assess(after)
    if len(a)!=len(b) or any((x['time'],x['stage'])!=(y['time'],y['stage']) for x,y in zip(a,b)):raise ValueError('different task clocks')
    for x,y in zip(a,b):
        if x['native']['requested']!=y['native']['requested']:raise ValueError('different arm requests')
    first=json.loads((after/'sdk/capture/0000.json').read_text(encoding='utf-8-sig'));matrix=affine(first)
    write_spec(out/'comparison.json',dict(neutralHand=sa,coordinated=sb,scope='same node route, head/body, gaze, task clocks; hand pose ablation only',visualAcceptance=False))
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1536x632','-r','30','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out/'comparison.mp4')]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE) as proc:
        for x,y in zip(a,b):
            canvas=Image.new('RGB',(1536,632),'#eeeeee');d=ImageDraw.Draw(canvas)
            d.text((10,8),f"HAND POSE ABLATION | {y['stage']} | {y['time']:.2f}s | same route / gaze / clock | no palm scaling",fill='black')
            for col,title in enumerate(['NEUTRAL HAND / BODY','COORDINATED HAND / BODY','NEUTRAL HAND CLOSE-UP','COORDINATED HAND CLOSE-UP']):d.text((col*384+7,47),title,fill='black')
            for j,(root,row) in enumerate([(before,x),(after,y)]):
                with Image.open(root/'sdk/capture'/f"{row['frame']:04}.png") as source:
                    raw=source.convert('RGB');canvas.paste(raw.resize((384,384)),(j*384,85))
                    center=np.mean(np.array(row['native']['actual'])[2:],axis=0);screen=np.r_[center,1.]@matrix
                    canvas.paste(raw.crop(local_crop(screen,160)).resize((384,384)),((j+2)*384,85))
                target_overlay(d,row,matrix,offset=(j*384,85),scale=.375)
            for j,row in enumerate([x,y]):
                d.text((j*768+8,494),f"HAND {row['nativeFingers']['status']} | ARM {row['native']['status']}",fill='black')
                if row['intent']['task']=='point':d.text((j*768+8,520),f"2D index-axis error to A: {angle_to_target(row,row['intent']['pointTarget']):.1f}deg",fill='black')
            d.text((8,564),'Task labels are authoring intentions, not semantic scores. Complete transitions and failures retained.',fill='black')
            d.text((8,588),'Close-ups follow actual wrist by translation at a fixed scale; full body panels retain original camera.',fill='black')
            proc.stdin.write(canvas.tobytes())
        proc.stdin.close();error=proc.stderr.read();code=proc.wait()
        if code:raise RuntimeError(error.decode(errors='replace'))
    print(out/'comparison.mp4')


def preview(root):
    sdk=root/'sdk';rows=[json.loads(x) for x in (sdk/'track.jsonl').read_text(encoding='utf-8').splitlines()]
    if not(sdk/'capture/complete.txt').exists():raise ValueError('incomplete capture')
    output=root/'expression-preview.mp4'
    if output.exists():raise FileExistsError(output)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',25)
    first=json.loads((sdk/'capture/0000.json').read_text(encoding='utf-8-sig'));matrix=affine(first)
    titles={'idle':'准备','point':'指向目标 A','explain':'解释与强调','near-head':'靠近头侧后停留'}
    phases={'baseline':'中性','prepare':'准备','reach':'伸臂','offer':'展开呈示','emphasize':'轻收拢强调','reopen':'再次展开','approach':'接近','express':'保持','withdraw':'收回','rest':'回到中性'}
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1024x1072','-r','30','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE) as proc:
        for row in rows:
            image=Image.new('RGB',(1024,1072),'#eeeeee');draw=ImageDraw.Draw(image)
            task=row['intent']['task'];phase=row['intent']['phase']
            draw.text((18,7),f"整体表达候选 · {titles[task]} · {phases[phase]}",font=font,fill='#202733')
            with Image.open(sdk/'capture'/f"{row['frame']:04}.png") as raw:image.paste(raw.convert('RGB'),(0,48))
            target_overlay(draw,row,matrix,offset=(0,48))
            proc.stdin.write(image.tobytes())
        proc.stdin.close();error=proc.stderr.read();code=proc.wait()
        if code:raise RuntimeError(error.decode(errors='replace'))
    print(output)


if __name__=='__main__':
    if '--compare' in sys.argv:compare(*map(Path,sys.argv[1:4]))
    elif '--preview' in sys.argv:preview(Path(sys.argv[1]))
    else:review(Path(sys.argv[1]))
