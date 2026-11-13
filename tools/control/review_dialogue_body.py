"""Preserve whole-session images, semantic decisions and actual execution limits."""
import sys,json,subprocess,collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
from tools.control.review_intent_expression import affine,angle_to_target
from tools.expression.unity import write_spec


def assess(root):
    rows=[json.loads(x) for x in (root/'sdk/track.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(rows)!=780 or not (root/'sdk/capture/complete.txt').exists():raise ValueError('incomplete recording')
    events=json.loads((root/'dialogue-events.json').read_text(encoding='utf-8'))
    history=json.loads((root/'planner-history.json').read_text(encoding='utf-8'))
    # The explicit window checks below belong to the shipped acceptance fixture.
    # Custom event orders still run, but must not inherit its test certification.
    from tools.control.dialogue_body_trial import fixture
    built_in=events['events']==[[frame,p] for frame,p in fixture()]
    mapping={int(frame):p['id'] for frame,p in events['events']}
    applied=all(rows[f-1]['intent']['planId']==identifier for f,identifier in mapping.items())
    cancelled=[r for r in rows if 391<=r['frame']<436]
    unavailable=[r for r in rows if 436<=r['frame']<513]
    holds=[r for r in rows if 120<=r['frame']<=141]
    point_angles=[angle_to_target(r,events['scene']['A']['xy']) for r in holds]
    safety_reasons=collections.Counter()
    for r in rows:
        for key in ('native','patch','nativeFingers','gaze','face'):
            for reason in r[key]['reasons']:
                if 'actual-' in reason or 'conflict' in reason:safety_reasons[key+':'+reason]+=1
    per_plan=[]
    for identifier in mapping.values():
        rr=[r for r in rows if r['intent']['planId']==identifier]
        per_plan.append(dict(id=identifier,frames=len(rr),limitedFrames=sum(r['session']['executionStatus']=='limited' for r in rr),
            maxWristError=max(r['native']['wristError'] for r in rr),
            reasons=dict(collections.Counter(x for r in rr for key in ('native','patch','nativeFingers','gaze','face') for x in r[key]['reasons']))))
    final=rows[-15:]
    feedback_error=0.
    from tools.control.face_control import FACE_IDS
    from tools.control.finger_control import FINGER_IDS
    from tools.control.gaze_control import GAZE_IDS
    for r in rows:
        f=json.loads((root/'sdk/capture'/f"{r['frame']:04}.json").read_text(encoding='utf-8-sig'))
        actual={v['id']:v['value'] for v in f['applied']}
        for side in ('native','patch'):
            feedback_error=max(feedback_error,float(np.max(abs(np.array(r[side]['actual']).ravel()-f['wholeArms'][side+'Chain']))))
        for key,ids in [('face',FACE_IDS),('nativeFingers',FINGER_IDS),('gaze',GAZE_IDS)]:
            feedback_error=max(feedback_error,float(max(abs(np.array(r[key]['applied'])-[actual[k] for k in ids]))))
    out=dict(frames=len(rows)+1,seconds=26,eventOwnershipPassed=applied,independentSDKFeedbackMaxError=feedback_error,
        shippedFixture=built_in,
        cancellationPassed=all(r['intent']['gesture']=='neutral' and r['session']['requestId']=='interrupt' for r in cancelled),
        unavailableTargetFallbackPassed=all(r['intent']['gesture']=='neutral' and r['intent']['fallback']=='referent-outside-tested-pointing-capability' for r in unavailable),
        finalAllChannelsSettled=all(r['session']['settled'] and r['nativeFingers']['settled'] and r['gaze']['settled'] and r['face']['settled'] for r in final),
        pointHold=dict(maxMaterialAxisAngleDegrees=max(point_angles,default=None),
                       armSettled=all(r['native']['positionReached'] and r['native']['terminalStateMet'] for r in holds),
                       handSettled=all(r['nativeFingers']['settled'] for r in holds)),
        safetyReasons=dict(safety_reasons),
        maxPairing=max(r['pairing'] for r in rows),minimumPatchArea=min(r['geometry']['minTriangleAreaRatio'] for r in rows),
        maxFlippedTriangles=max(r['geometry']['flippedTriangles'] for r in rows),
        maxRestore=max(r['geometry']['restoredMeshError'] for r in rows),
        continuity={s:{k:max(r[s]['metrics'][k] for r in rows) for k in ('maxNodeStep','maxNodeSpeed','maxNodeAcceleration','maxParameterStep','maxParameterAcceleration')} for s in ('native','patch')},
        perPlan=per_plan,plannerEvents=history['events'],
        visualAcceptance=False,modelGenerated=False,physics=False,
        scope='offline semantic rule baseline, A registered anchor; no arbitrary point IK, 3D gaze, audio sync or production integration')
    if not built_in:
        for key in ('cancellationPassed','unavailableTargetFallbackPassed','pointHold'):out[key]=None
    write_spec(root/'dialogue-assessment.json',out)
    return rows,events,out


def review(root):
    rows,events,summary=assess(root)
    output=root/'dialogue-review.mp4'
    if output.exists() or (root/'dialogue-raw.mp4').exists():raise FileExistsError(output)
    matrix=affine(json.loads((root/'sdk/capture/0000.json').read_text(encoding='utf-8-sig')))
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',23)
    small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1536x900','-r','30','-i','-',
         '-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE) as proc:
        for r in rows:
            canvas=Image.new('RGB',(1536,900),'#20242d');d=ImageDraw.Draw(canvas)
            d.text((12,8),f"离线对话身体原型 | {r['time']:.2f}s | {r['intent']['kind']} / {r['intent']['affect']} | 关注：{r['intent']['target']}",font=font,fill='white')
            d.text((12,43),'规则基线产生表达计划；固定30Hz实际SDK反馈；物理关闭；文字时钟嘴部提示，无语音同步',font=small,fill='#cccccc')
            with Image.open(root/'sdk/capture'/f"{r['frame']:04}.png") as image:
                raw=image.convert('RGB');canvas.paste(raw.resize((512,512)),(0,110));canvas.paste(raw.resize((512,512)),(512,110))
                center=np.mean([r['taskMarkers']['irisLeft'],r['taskMarkers']['irisRight']],axis=0)+[0,.025]
                screen=np.r_[center,1.]@matrix;span=180
                canvas.paste(raw.crop(tuple(map(int,[screen[0]-span/2,screen[1]-span/2,screen[0]+span/2,screen[1]+span/2]))).resize((512,512)),(1024,110))
            for x,title in [(10,'原图'),(522,'请求节点（蓝）／实际节点（红）／场景目标'),(1034,'实际脸部近景')]:d.text((x,79),title,font=small,fill='white')
            def screenpoint(p):
                x,y=np.r_[p,1.]@matrix;return float(x/2+512),float(y/2+110)
            for name,obj in events['scene'].items():
                x,y=screenpoint(obj['xy']);d.ellipse((x-8,y-8,x+8,y+8),outline='#ffcf50',width=2)
                d.text((x+10,y-15),name,font=font,fill='#ffcf50')
            for side in ('native','patch'):
                for key,color in [('requested','#59bfff'),('actual','#ff6868')]:
                    p=r[side][key]
                    if key=='requested':
                        d.line([screenpoint(v) for v in p],fill=color,width=3)
                    else:d.line([screenpoint(v) for v in p],fill=color,width=2)
            tip=screenpoint(r['taskMarkers']['indexTip']);base=screenpoint(r['taskMarkers']['indexBase'])
            d.line([base,tip],fill='#ffe95b',width=4)
            text=r['intent']['text'];d.text((15,642),text[:62],font=font,fill='white')
            if len(text)>62:d.text((15,673),text[62:124],font=font,fill='white')
            d.text((15,710),f"计划 {r['intent']['planId']} | 手形 {r['intent']['gesture']} | {'文字发言期' if r['intent']['speaking'] else '停顿/倾听'} | 身体执行 {r['session']['executionStatus']}",font=small,fill='white')
            d.text((15,744),f"原生 {r['native']['status']} 腕误差 {r['native']['wristError']:.4f} | 补丁 {r['patch']['status']} | 手指 {r['nativeFingers']['status']} | 视线 {r['gaze']['status']} | 脸 {r['face']['status']}",font=small,fill='white')
            reason=r['intent']['fallback'] or '; '.join(r['session']['reasons']+r['session']['expressionReasons']) or '当前无执行限制；表达含义和自然度仍需人工审看'
            d.text((15,779),reason[:125],font=small,fill='#ffcc77')
            d.text((15,819),'A：已有指向能力登记；B：未登记可用指向，降级为视线＋文字。所有阶段连续录制，不重置控制器。',font=small,fill='#cccccc')
            d.text((15,852),'蓝/红点是模型二维材料参考；眼球输入不是三维视线；嘴形变化不是音素口型。',font=small,fill='#cccccc')
            if r['frame'] in (60,130,240,360,385,410,480,560,660,780):canvas.save(root/f"review-{r['frame']:04}.png")
            proc.stdin.write(canvas.tobytes())
        proc.stdin.close();error=proc.stderr.read();code=proc.wait()
        if code:raise RuntimeError(error.decode(errors='replace'))
    raw=root/'dialogue-raw.mp4'
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-framerate','30','-start_number','1','-i',str(root/'sdk/capture/%04d.png'),
                    '-frames:v','780','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(raw)],check=True)
    print(output)


if __name__=='__main__':review(Path(sys.argv[1]))
