"""Task decisions, material-node requests and SDK execution on one action clock."""
import sys,json,subprocess,collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
from tools.control.review_intent_expression import affine
from tools.control.face_control import FACE_IDS
from tools.control.finger_control import FINGER_IDS
from tools.control.gaze_control import GAZE_IDS
from tools.expression.unity import write_spec


def assess(root):
    rows=[json.loads(x) for x in (root/'sdk/track.jsonl').read_text(encoding='utf-8').splitlines()]
    protocol=json.loads((root/'acceptance-protocol.json').read_text(encoding='utf-8'))
    if len(rows)+1!=protocol['frames'] or not (root/'sdk/capture/complete.txt').exists():raise ValueError('incomplete SDK session')
    source=json.loads((root/'task-inputs.json').read_text(encoding='utf-8'))
    history=json.loads((root/'director-history.json').read_text(encoding='utf-8'))
    actual_violations=collections.Counter();solver_limits=collections.Counter();readback_error=0.;cropped=[]
    matrix=affine(json.loads((root/'sdk/capture/0000.json').read_text(encoding='utf-8-sig')))
    for r in rows:
        for key in ('native','patch','nativeFingers','gaze','face'):
            for reason in r[key]['reasons']:
                if reason.startswith('actual-') or reason in ('gaze-stop-conflict','finger-stop-conflict'):actual_violations[key+':'+reason]+=1
                else:solver_limits[key+':'+reason]+=1
        frame=json.loads((root/'sdk/capture'/f"{r['frame']:04}.json").read_text(encoding='utf-8-sig'))
        values={v['id']:v['value'] for v in frame['applied']}
        hand_ids={'ArtMesh'+str(i) for i in range(230,237)}
        corners=np.array([[g[x],g[y],1.] for g in frame['drawables'] if g['id'] in hand_ids for x in ('minx','maxx') for y in ('miny','maxy')])@matrix
        if np.min(corners)<0 or np.max(corners)>1024:cropped.append(r['frame'])
        for side in ('native','patch'):
            readback_error=max(readback_error,float(max(abs(np.array(r[side]['actual']).ravel()-frame['wholeArms'][side+'Chain']))))
        for name,ids in [('face',FACE_IDS),('nativeFingers',FINGER_IDS),('gaze',GAZE_IDS)]:
            readback_error=max(readback_error,float(max(abs(np.array(r[name]['applied'])-[values[k] for k in ids]))))
    outcomes=history['outcomes'];events=history['events']
    outside=[o for o in outcomes if o['taskId']=='outside']
    failed=next((o for o in outside if o['target']=='U' and o['result']=='limited'),None)
    recovered=next((o for o in outside if o['target']=='C' and o['result']=='execution-goal-met'),None)
    final=rows[-15:]
    main=[o for o in outcomes if o['taskId']=='compare-main']
    patch_motion=max(float(np.linalg.norm(np.array(r['patch']['actual'])[2]-rows[0]['patch']['actual'][2])) for r in rows)
    limits={}
    branches={}
    for side in ('native','patch'):
        longest=0;current=0
        for r in rows:
            current=current+1 if r[side]['status']=='limited' else 0;longest=max(longest,current)
        limits[side]=dict(frames=sum(r[side]['status']=='limited' for r in rows),longestContinuousSeconds=longest/30)
        branches[side]=[r['frame'] for a,r in zip(rows,rows[1:]) if a[side]['actualBranch']!=r[side]['actualBranch']]
    out=dict(frames=len(rows)+1,seconds=len(rows)/30,physics=protocol['physics'],
        taskOwnershipPassed=all(rows[f-1]['intent']['task']['id']==t['id'] for f,t in source['interventions']),
        mainComparisonActualGoalsMet=[o['target'] for o in main if o['result']=='execution-goal-met'],
        sameTaskFailureRecovery=bool(failed and recovered and recovered['requestedAt']>failed['endedAt']),
        outsideTaskCorrectlyLimited=any(e['event']=='task-finished' and e['taskId']=='outside' and e['result']=='limited' for e in events),
        noOldGoalAfterCancel=all(r['intent']['task']['id']=='interrupt' and r['intent']['gesture']=='neutral' for r in rows if 409<=r['frame']<481),
        finalAllChannelsSettled=all(r['session']['settled'] and r['nativeFingers']['settled'] and r['gaze']['settled'] and r['face']['settled'] for r in final),
        actualContinuityViolations=dict(actual_violations),solverAndTerminalLimits=dict(solver_limits),
        limitationDuration=limits,actualBranchTransitionFrames=branches,
        nativeHandOutOfFrame=cropped,
        independentSDKFeedbackMaxError=readback_error,maxPairing=max(r['pairing'] for r in rows),
        minPatchArea=min(r['geometry']['minTriangleAreaRatio'] for r in rows),maxFlips=max(r['geometry']['flippedTriangles'] for r in rows),
        maxRestore=max(r['geometry']['restoredMeshError'] for r in rows),patchWristExcursion=patch_motion,
        nativeWristYRange=[min(r['native']['actual'][2][1] for r in rows),max(r['native']['actual'][2][1] for r in rows)],
        continuity={s:{k:max(r[s]['metrics'][k] for r in rows) for k in ('maxNodeStep','maxNodeSpeed','maxNodeAcceleration','maxParameterStep','maxParameterAcceleration')} for s in ('native','patch')},
        outcomes=outcomes,visualAcceptance=False,learnedPolicy=False,
        scope='task-directed rule baseline; fixed scene goals; material-axis proxy, not 3D hand normals, semantic recognition or free-text understanding')
    write_spec(root/'task-assessment.json',out)
    return rows,source,out


def review(root):
    rows,source,summary=assess(root);output=root/'task-expression-review.mp4';raw_path=root/'task-expression-raw.mp4'
    if output.exists() or raw_path.exists():raise FileExistsError(output)
    matrix=affine(json.loads((root/'sdk/capture/0000.json').read_text(encoding='utf-8-sig')))
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',22);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',17)
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1536x960','-r','30','-i','-',
         '-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    snapshots={int(round(o['endedAt']*30)) for o in summary['outcomes'] if 'endedAt' in o}|{420,530,1080}
    def plot(p):return 525+(float(p[0])+.7)/1.4*486,599-(float(p[1])+.15)/1.3*482
    with subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE) as proc:
        for r in rows:
            intent=r['intent'];task=intent['task'];report=r['session']['task']
            canvas=Image.new('RGB',(1536,960),'#20242d');d=ImageDraw.Draw(canvas)
            d.text((12,8),f"任务驱动身体测试 | {r['time']:.2f}s | {task['id']} | {task['purpose']} | 规则状态：{report['phase']}",font=font,fill='white')
            d.text((12,42),'实际SDK物理开启；目标切换由实际到位/失败反馈推进；离线规则基线，无自由文本理解或学习声明',font=small,fill='#cccccc')
            for x,title in [(10,'实际原图'),(522,'模型坐标：请求蓝／理想绿／实际红'),(1034,'实际脸部；下方为原生／补丁手近景')]:d.text((x,77),title,font=small,fill='white')
            with Image.open(root/'sdk/capture'/f"{r['frame']:04}.png") as im:
                raw=im.convert('RGB');canvas.paste(raw.resize((512,512)),(0,100))
                def crop(center,span,size):
                    x,y=np.r_[center,1.]@matrix
                    return raw.crop(tuple(map(int,[x-span/2,y-span/2,x+span/2,y+span/2]))).resize((size,size))
                face=np.mean([r['taskMarkers']['irisLeft'],r['taskMarkers']['irisRight']],axis=0)+[0,.025]
                pixels_per_unit=float(np.linalg.norm(matrix[0]))
                canvas.paste(crop(face,.254*pixels_per_unit,512),(1024,100))
                for j,side in enumerate(('native','patch')):
                    center=np.mean(np.array(r[side]['actual'])[2:],axis=0)
                    canvas.paste(crop(center,.198*pixels_per_unit,256),(1024+j*256,612))
            d.rectangle((512,100,1023,611),fill='#28313b',outline='#71808b')
            for name,obj in source['scene'].items():
                x,y=plot(obj['xy']);d.ellipse((x-6,y-6,x+6,y+6),outline='#ffcf50',width=2);d.text((x+9,y-15),name,font=font,fill='#ffcf50')
            for side in ('native','patch'):
                for key,color in [('requested','#59bfff'),('ideal','#70cf88'),('actual','#ff6868')]:
                    p=r[side][key];points=[plot(v) for v in p];d.line(points,fill=color,width=2)
                    for x,y in points:d.ellipse((x-3,y-3,x+3,y+3),outline=color,width=2)
            d.line([plot(r['taskMarkers'][k]) for k in ('indexBase','indexTip')],fill='#ffed6a',width=3)
            d.text((15,631),task['request'][:45],font=font,fill='white')
            d.text((15,672),intent['text'][:50],font=font,fill='#ffdc86')
            d.text((15,714),f"目标 {intent['target']} | 连续达标 {report['stableFrames']}/15 帧 | 任务置信状态 {report['confidence']:.2f}",font=small,fill='white')
            angle=report['axisErrorDegrees'];angle='不适用' if angle is None else f'{angle:.2f}°（材料代理）'
            d.text((15,747),f"指向误差 {angle} | 原生 {r['native']['status']} | 补丁 {r['patch']['status']}",font=small,fill='white')
            d.text((15,780),f"腕残差 原生 {r['native']['wristError']:.4f} / 补丁 {r['patch']['wristError']:.4f} | 已受限目标 {report['failedTargets']}",font=small,fill='white')
            reason=r['session']['expressionFallback'] or '; '.join(r['session']['reasons']+r['session']['expressionReasons']) or '无当前执行限制；不代表视觉含义或自然度已通过'
            d.text((15,815),reason[:100],font=small,fill='#ffb87a')
            d.text((15,854),'A/C：固定场景指向；E：双臂展开；U：故意超范围，实际失败后转向C。',font=small,fill='#cccccc')
            d.text((15,887),'空间节点与掌/指轴是二维材料参考；未提供三维掌向或语音同步。',font=small,fill='#cccccc')
            d.text((1034,883),'原生手／补丁手：相同源像素裁剪跨度',font=small,fill='white')
            d.text((15,921),'全过程保留：不裁掉失败，不重置控制器，不缩小全部动作，不把未完成目标算成功。',font=small,fill='#cccccc')
            if r['frame'] in snapshots:canvas.save(root/f"review-{r['frame']:04}.png")
            proc.stdin.write(canvas.tobytes())
        proc.stdin.close();error=proc.stderr.read();code=proc.wait()
        if code:raise RuntimeError(error.decode(errors='replace'))
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-framerate','30','-start_number','1','-i',str(root/'sdk/capture/%04d.png'),
                    '-frames:v',str(len(rows)),'-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(raw_path)],check=True)
    print(output)


if __name__=='__main__':review(Path(sys.argv[1]))
