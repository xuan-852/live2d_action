"""Independent model responses and actual SDK body execution, including failures."""
import sys,json,subprocess,collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
from tools.control.review_intent_expression import affine
from tools.control.embodied_plan import POSTURE_KEYS
from tools.control.body_session import HEAD_BODY_IDS
from tools.control.face_expression import EXTENDED_FACE_IDS
from tools.control.finger_control import FINGER_IDS
from tools.control.gaze_control import GAZE_IDS
from tools.expression.unity import write_spec


def assess(root):
    rows=[json.loads(x) for x in (root/'sdk/track.jsonl').read_text(encoding='utf-8').splitlines()]
    protocol=json.loads((root/'acceptance-protocol.json').read_text(encoding='utf-8'))
    if len(rows)+1!=protocol['frames'] or not (root/'sdk/capture/complete.txt').exists():raise ValueError('incomplete capture')
    history=json.loads((root/'model-history.json').read_text(encoding='utf-8'))
    service=json.loads((root/'model-service.json').read_text(encoding='utf-8'))
    scene=json.loads((root/'scene.json').read_text(encoding='utf-8'))
    decisions=history['decisions'];per_plan=[];violations=collections.Counter();limits=collections.Counter()
    for entry in decisions:
        plan=entry['plan'];rr=[r for r in rows if r['intent']['planId']==plan['id']]
        per_plan.append(dict(id=plan['id'],kind=plan['kind'],target=plan['target'],reason=entry['reason'],
            acceptedAt=entry['acceptedAt'],maxStableFrames=max((r['session']['upperExecution']['stableFrames'] for r in rr),default=0),
            held=any(r['session']['upperExecution']['held'] for r in rr),
            failures=sorted({r['session']['upperExecution']['constraintFailure'] for r in rr if r['session']['upperExecution']['constraintFailure']}),
            inferenceWallSeconds=entry['inferenceWallSeconds']))
    readback_error=0.;cropped=[]
    matrix=affine(json.loads((root/'sdk/capture/0000.json').read_text(encoding='utf-8-sig')))
    for r in rows:
        for key in ('native','patch','nativeFingers','gaze','face'):
            for reason in r[key]['reasons']:
                if reason.startswith('actual-') or reason in ('gaze-stop-conflict','finger-stop-conflict'):violations[key+':'+reason]+=1
                else:limits[key+':'+reason]+=1
        f=json.loads((root/'sdk/capture'/f"{r['frame']:04}.json").read_text(encoding='utf-8-sig'))
        applied={v['id']:v['value'] for v in f['applied']}
        for side in ('native','patch'):
            readback_error=max(readback_error,float(max(abs(np.array(r[side]['actual']).ravel()-f['wholeArms'][side+'Chain']))))
        for name,ids in [('nativeFingers',FINGER_IDS),('gaze',GAZE_IDS),('face',EXTENDED_FACE_IDS[:len(r['face']['applied'])])]:
            readback_error=max(readback_error,float(max(abs(np.array(r[name]['applied'])-[applied[k] for k in ids]))))
        readback_error=max(readback_error,float(max(abs(np.array(r['session']['headBodyActual'])-[applied[k] for k in HEAD_BODY_IDS]))))
        hand_ids={'ArtMesh'+str(i) for i in range(230,237)}
        corners=np.array([[g[x],g[y],1.] for g in f['drawables'] if g['id'] in hand_ids for x in ('minx','maxx') for y in ('miny','maxy')])@matrix
        if np.min(corners)<0 or np.max(corners)>1024:cropped.append(r['frame'])
    revision=[]
    for i,d in enumerate(decisions):
        if d['reason']=='actual-execution-limited' and i:
            revision.append(dict(previous=decisions[i-1]['plan']['id'],next=d['plan']['id'],
                previousTarget=decisions[i-1]['plan']['target'],newTarget=d['plan']['target'],
                previousPosture=decisions[i-1]['plan']['posture'],newPosture=d['plan']['posture']))
    interrupts=[]
    for d in decisions:
        if d['reason']=='user-interruption-during-motion':
            q=json.loads((root/'model-mailbox'/f"{d['requestId']:03}.request.json").read_text(encoding='utf-8'))
            frame=round(q['observationTime']*30);previous=rows[frame-1]
            interrupts.append(dict(observationFrame=frame,maxActualNodeSpeed=previous['native']['metrics']['maxNodeSpeed'],responseKind=d['plan']['kind']))
    posture=np.array([r['session']['headBodyActual'] for r in rows]);final=rows[-15:]
    held_c_after_u=False
    for i,entry in enumerate(decisions):
        if entry['reason']=='actual-execution-limited' and i and decisions[i-1]['plan']['target']=='U':
            # A model may acknowledge a limit in speech before executing its
            # revised motor plan. Intermediate speech is not a new user request.
            failed_request=json.loads((root/'model-mailbox'/f"{entry['requestId']:03}.request.json").read_text(encoding='utf-8'))
            for later in decisions[i:]:
                if later['plan']['kind']!='reference' or later['plan']['target']!='C':continue
                later_request=json.loads((root/'model-mailbox'/f"{later['requestId']:03}.request.json").read_text(encoding='utf-8'))
                if later_request.get('activeUserRequest')==failed_request.get('activeUserRequest'):
                    held_c_after_u=held_c_after_u or any(p['id']==later['plan']['id'] and p['held'] for p in per_plan)
    out=dict(frames=len(rows)+1,actionSeconds=len(rows)/30,physics=protocol['physics'],service=service,
        decisions=len(decisions),apiRequests=len(list((root/'model-mailbox').glob('*.api-*.json')))-len(list((root/'model-mailbox').glob('*.api-*.rejected.json'))),
        perPlan=per_plan,modelRevisionsAfterActualLimits=revision,interruptions=interrupts,
        driverFinished=history['finished'],modelFinished=history.get('modelFinished',history['finished']),
        budgetExhausted=history.get('budgetExhausted',False),interrupted=history['interrupted'],actualUFailure=history['probeFailed'],
        heldCChosenByModelAfterUFailure=held_c_after_u,
        bilateralHoldPassed=any(p['target']=='E' and p['kind']=='reference' and p['held'] for p in per_plan),
        allScenarioStagesExercised=bool(history['finished'] and not history.get('budgetExhausted',False) and interrupts and history['probeFailed'] and held_c_after_u and any(p['target']=='E' and p['kind']=='reference' and p['held'] for p in per_plan)),
        actualPostureMaxAbs=dict(zip(POSTURE_KEYS,np.max(abs(posture),axis=0).tolist())),
        actualHeadBodyCommandResidual=max(r['session']['headBodyInputResidual'] for r in rows),
        independentSDKFeedbackMaxError=readback_error,nativeHandOutOfFrame=cropped,
        finalAllChannelsSettled=all(r['session']['settled'] and r['nativeFingers']['settled'] and r['gaze']['settled'] and r['face']['settled'] for r in final),
        actualContinuityViolations=dict(violations),solverAndTerminalLimits=dict(limits),
        minPatchArea=min(r['geometry']['minTriangleAreaRatio'] for r in rows),maxFlips=max(r['geometry']['flippedTriangles'] for r in rows),
        maxPairing=max(r['pairing'] for r in rows),maxRestore=max(r['geometry']['restoredMeshError'] for r in rows),
        wallModelSeconds=sum(x['inferenceWallSeconds'] for x in decisions),
        realTime=False,visualAcceptance=False,
        scope='independent model generates semantic plans using numeric SDK feedback; fixed scene affordances; not free joint synthesis, semantic visual certification or audio lip sync')
    write_spec(root/'model-loop-assessment.json',out)
    return rows,scene,out


def review(root):
    rows,scene,summary=assess(root);output=root/'model-body-review.mp4';raw_path=root/'model-body-raw.mp4'
    if output.exists() or raw_path.exists():raise FileExistsError(output)
    matrix=affine(json.loads((root/'sdk/capture/0000.json').read_text(encoding='utf-8-sig')))
    pixels=float(np.linalg.norm(matrix[0]));font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',23);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r','30','-i','-',
         '-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    snapshot_frames={round(d['acceptedAt']*30) for d in summary['perPlan']}|{len(rows)}
    def plot(p):return 660+(float(p[0])+.7)/1.4*600,748-(float(p[1])+.15)/1.3*600
    with subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE) as proc:
        for r in rows:
            intent=r['intent'];plan=intent['upperPlan'];report=r['session']['upperExecution']
            canvas=Image.new('RGB',(1920,1080),'#20242d');d=ImageDraw.Draw(canvas)
            d.text((15,8),f"独立模型闭环 | {summary['service']['modelRequested']} | 动作时间 {r['time']:.2f}s | 计划 {plan['id']}",font=font,fill='white')
            d.text((15,45),'SDK物理开启；模型独立生成语言＋头身＋双臂意图；推理时动作时钟暂停，墙钟耗时另记；无语音同步',font=small,fill='#cccccc')
            for x,title in [(10,'实际身体原图'),(650,'请求蓝／可执行理想绿／实际红；材料参考'),(1290,'实际脸部与头身朝向近景')]:d.text((x,83),title,font=small,fill='white')
            with Image.open(root/'sdk/capture'/f"{r['frame']:04}.png") as im:
                raw=im.convert('RGB');canvas.paste(raw.resize((640,640)),(0,120))
                face=np.mean([r['taskMarkers'][k] for k in ('irisLeft','irisRight')],axis=0)+[0,.025]
                x,y=np.r_[face,1.]@matrix;span=.27*pixels
                canvas.paste(raw.crop(tuple(map(int,[x-span/2,y-span/2,x+span/2,y+span/2]))).resize((640,640)),(1280,120))
            d.rectangle((640,120,1279,759),outline='#71808b')
            for name,obj in scene.items():
                x,y=plot(obj['xy']);d.ellipse((x-6,y-6,x+6,y+6),outline='#ffcf50',width=2);d.text((x+10,y-14),name,font=font,fill='#ffcf50')
            for side in ('native','patch'):
                for key,color in [('requested','#59bfff'),('ideal','#70cf88'),('actual','#ff6868')]:d.line([plot(v) for v in r[side][key]],fill=color,width=2)
            shoulders=np.mean([r[s]['actual'][0] for s in ('native','patch')],axis=0)
            eyes=np.mean([r['taskMarkers'][k] for k in ('irisLeft','irisRight')],axis=0)
            d.line([plot(shoulders),plot(eyes)],fill='#77e2dd',width=3)
            d.text((15,782),plan['text'][:72],font=font,fill='white')
            if len(plan['text'])>72:d.text((15,812),plan['text'][72:145],font=font,fill='white')
            d.text((15,852),f"{plan['kind']} / {plan['target']} / {plan['affect']} | 实际保持 {report['stableFrames']} 帧 | 原生 {r['native']['status']} | 补丁 {r['patch']['status']}",font=small,fill='white')
            desired=[plan['posture'][k] for k in POSTURE_KEYS];actual=r['session']['headBodyActual']
            fmt=lambda a:' / '.join(f'{v:.1f}' for v in a)
            d.text((15,887),'头转/点头/侧倾/身转/身倾/身侧：目标 '+fmt(desired)+' | 应用 '+fmt(actual),font=small,fill='#b8d9ff')
            d.text((15,922),'模型决策说明：'+plan['explanation'][:95],font=small,fill='#cccccc')
            reason=report['constraintFailure'] or '; '.join(r['session']['reasons']+r['session']['expressionReasons']) or '当前未报告执行限制；含义与自然度仍需人工审看'
            d.text((15,957),reason[:145],font=small,fill='#ffc287')
            d.text((15,994),'数值反馈不是自动视觉评分；蓝绿色链不代表已做到。原生与补丁能力不同，所有失败和未完成目标保留。',font=small,fill='#cccccc')
            d.text((15,1030),f"独立请求决策 #{intent['modelLoop']['requestId']} | finish={plan['finish']} | 不是训练结果，也不是生产实时集成",font=small,fill='#cccccc')
            if r['frame'] in snapshot_frames or r['frame']%120==0:canvas.save(root/f"review-{r['frame']:04}.png")
            proc.stdin.write(canvas.tobytes())
        proc.stdin.close();error=proc.stderr.read();code=proc.wait()
        if code:raise RuntimeError(error.decode(errors='replace'))
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-framerate','30','-start_number','1','-i',str(root/'sdk/capture/%04d.png'),
        '-frames:v',str(len(rows)),'-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(raw_path)],check=True)
    print(output)


if __name__=='__main__':review(Path(sys.argv[1]))
