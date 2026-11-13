"""Paired face/body replay review, with independent SDK parameter readback."""
import sys,json,subprocess,collections
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
from tools.control.face_expression import EXTENDED_FACE_IDS
from tools.control.review_intent_expression import affine
from tools.expression.unity import write_spec


def review(root):
    protocol=json.loads((root/'protocol.json').read_text(encoding='utf-8'));source=Path(protocol['source'])
    rows=[json.loads(s) for s in (root/'sdk/track.jsonl').read_text(encoding='utf-8').splitlines()]
    old=[json.loads(s) for s in (source/'sdk/track.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(rows)!=720 or not (root/'sdk/capture/complete.txt').exists():raise ValueError('incomplete replay')
    matrix=affine(json.loads((root/'sdk/capture/0000.json').read_text(encoding='utf-8-sig')))
    residual=0.;body_delta=0.;pairing=0.;reasons=collections.Counter()
    for r,b in zip(rows,old):
        f=json.loads((root/'sdk/capture'/f"{r['frame']:04}.json").read_text(encoding='utf-8-sig'))
        applied={v['id']:v['value'] for v in f['applied']}
        residual=max(residual,max(abs(np.array(r['face']['applied'])-[applied[k] for k in EXTENDED_FACE_IDS])))
        for s in ('native','patch'):body_delta=max(body_delta,float(np.max(abs(np.array(r[s]['actual'])-b[s]['actual']))))
        pairing=max(pairing,r['pairing']);reasons.update(r['face']['reasons'])
    summary=dict(frames=len(rows),newModelCalls=0,independentFaceReadbackError=float(residual),
        maximumBodyNodeDifference=body_delta,pairing=pairing,faceReasons=dict(reasons),
        minimumEyeOpen=min(min(r['face']['applied'][:2]) for r in rows),
        final15FaceSettled=all(r['face']['settled'] for r in rows[-15:]),
        browsRange=np.ptp([r['face']['applied'][6:] for r in rows],axis=0).tolist(),
        semanticAcceptance=False,source=str(source),scope='same recorded body requests; face candidate pending user review')
    summary['engineeringPassed']=bool(residual<=1e-6 and pairing<=1e-6 and not reasons and summary['final15FaceSettled'] and summary['minimumEyeOpen']<.1 and body_delta<1e-5)
    write_spec(root/'face-assessment.json',summary)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',23);small=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    output=root/'face-body-review.mp4'
    if output.exists():raise FileExistsError(output)
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1440x900','-r','30','-i','-',
         '-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE) as proc:
        for r in rows:
            i=r['frame'];c=Image.new('RGB',(1440,900),'#20242d');d=ImageDraw.Draw(c)
            d.text((15,10),f"表情候选 | 同一段 DeepSeek 身体计划复放 | {r['time']:.2f}s",font=font,fill='white')
            d.text((15,48),'无新增模型决策；眼／眉／嘴组合＋眨眼；嘴部只是文字时钟提示，无音频同步',font=small,fill='#cccccc')
            center=np.mean([r['taskMarkers'][k] for k in ('irisLeft','irisRight')],axis=0)+[0,.012]
            x,y=np.r_[center,1.]@matrix;span=.23*np.linalg.norm(matrix[0])
            images=[]
            for base in (source,root):
                with Image.open(base/'sdk/capture'/f'{i:04}.png') as im:images.append(im.convert('RGB'))
            for im,left,label in [(images[0],0,'基线脸部'),(images[1],480,'表情候选脸部')]:
                c.paste(im.crop(tuple(map(int,[x-span/2,y-span/2,x+span/2,y+span/2]))).resize((480,480)),(left,120))
                d.text((left+12,85),label,font=font,fill='white')
            c.paste(images[1].resize((480,480)),(960,120));d.text((972,85),'候选全身原图',font=font,fill='white')
            intent=r['intent'];d.text((15,625),intent['text'][:85],font=font,fill='white')
            d.text((15,665),f"计划 {intent['planId']} | {intent['kind']} / {intent['affect']} | 眼／嘴／微笑／眉 应用值",font=small,fill='#b8d9ff')
            d.text((15,705),' / '.join(f'{v:.2f}' for v in r['face']['applied']),font=font,fill='white')
            reasons_now=r['face']['reasons']+r['native']['reasons']+r['patch']['reasons']
            d.text((15,750),'实际限制：'+('; '.join(reasons_now)[:125] or '当前无执行限制报告'),font=small,fill='#ffc287')
            d.text((15,795),'保留原动作请求与节奏；到位不等于表情含义／自然度通过；本轮未接生产桌宠。',font=small,fill='#cccccc')
            if i in (60,111,150,300,450,600,720):c.save(root/f'face-review-{i:04}.png')
            proc.stdin.write(c.tobytes())
        proc.stdin.close();error=proc.stderr.read();code=proc.wait()
        if code:raise RuntimeError(error.decode(errors='replace'))
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-framerate','30','-start_number','1','-i',str(root/'sdk/capture/%04d.png'),
        '-frames:v','720','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(root/'face-body-raw.mp4')],check=True)
    print(json.dumps(summary,ensure_ascii=False));print(output)


if __name__=='__main__':review(Path(sys.argv[1]))
