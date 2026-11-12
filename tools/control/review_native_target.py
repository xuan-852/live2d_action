"""Same source samples, full native failure comparison and static-goal metrics."""
import sys,json,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image,ImageDraw
import imageio_ffmpeg
from tools.expression.unity import write_spec


def run(root):
    old=Path('E:/FuXuanLab/runs/20261112-mmd-timed-controller-01/revision-02/sdk')
    cases=[old,root/'mmd-five-same-budgets/sdk',root/'mmd-five-final/sdk']
    tracks=[[json.loads(x) for x in (p/'track.jsonl').read_text().splitlines() if json.loads(x)['stage']=='mmd-source'] for p in cases]
    if len({len(x) for x in tracks})!=1:raise ValueError('source sample count mismatch')
    maxdiff=0.;maxangle=0.;summaries=[]
    for rows in zip(*tracks):
        if len({r['sourceTime'] for r in rows})!=1:raise ValueError('source time mismatch')
        for r in rows[1:]:
            a=np.array(r['native']['requested']);b=np.array(rows[0]['native']['requested'])
            maxdiff=max(maxdiff,float(np.max(abs(a[1:3]-b[1:3]))))
            angle=lambda p:np.degrees(np.arctan2(*(p[3]-p[2])[::-1]))
            maxangle=max(maxangle,float(abs((angle(a)-angle(b)+180)%360-180)))
    if maxdiff>1e-6 or maxangle>1e-6:raise ValueError('modified source target')
    for p,track in zip(cases,tracks):
        s=[r['native'] for r in track]
        met=lambda x:x['wristError']<=.005 and x['elbowError']<=.005 and x['palmErrorDeg']<=4
        summaries.append(dict(case=str(p),wristRMS=float(np.sqrt(np.mean([x['wristError']**2 for x in s]))),
            spatialTrackingMet=sum(met(x) for x in s),spatialMissSourceFrames=[i for i,x in enumerate(s) if not met(x)],
            overheadFrame28={k:s[28][k] for k in ['wristError','elbowError','palmErrorDeg','status','reasons']},
            failedFrames=[i for i,x in enumerate(s) if x['status']=='failed'],
            maxima={k:max(x['metrics'][k] for x in s) for k in s[0]['metrics']},
            actualWristTravel=float(np.linalg.norm(np.diff(np.array([x['actual'][2] for x in s]),axis=0),axis=1).sum()),
            reasons=sorted({k for x in s for k in x['reasons']})))
    static=[json.loads(x) for x in (root/'static-five-final/track.jsonl').read_text().splitlines()]
    hold={}
    for stage in ['overhead-hold','neutral-return']:
        selected=[x for x in static if x['stage']==stage][-15:]
        hold[stage]=dict(final15HoldMet=all(x['native']['positionReached'] and x['native']['terminalStateMet'] for x in selected),
            final=selected[-1]['native'],maxWristError=max(x['native']['wristError'] for x in selected),maxElbowError=max(x['native']['elbowError'] for x in selected))
    write_spec(root/'native-comparison.json',dict(sameRequestsMaxDifference=maxdiff,samePalmHeadingMaxDifference=maxangle,summaries=summaries,static=hold,
        scope='same original source samples; columns show combined backend/solver/response changes, not one-factor ablation',visualAcceptance=False))
    output=root/'native-source-comparison.mp4'
    if output.exists():raise FileExistsError(output)
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1536x656','-r','30','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE) as proc:
        for rows in zip(*tracks):
            canvas=Image.new('RGB',(1536,656),'#eeeeee');draw=ImageDraw.Draw(canvas)
            draw.text((10,8),f"NATIVE TARGET CONTROL | same source {rows[0]['sourceTime']:.3f}s | original 30Hz; no retiming/scaling",fill='black')
            draw.text((10,28),'Green=request wrist/elbow, orange=actual. All source failures retained. Changed budgets are explicit, not naturalness certification.',fill='black')
            for j,(r,p,title) in enumerate(zip(rows,cases,['OLD THREE AXES / old limits','FIVE AXES / old limits, earlier solver','FIVE AXES / finite IK + native response'])):
                draw.text((j*512+8,52),title,fill='black')
                with Image.open(p/'capture'/f"{r['frame']:04}.png") as im:canvas.paste(im.convert('RGB').resize((512,512)),(j*512,80))
                frame=json.loads((p/'capture'/f"{r['frame']:04}.json").read_text(encoding='utf-8-sig'))
                actual=np.array(r['native']['actual']);screen=np.array(frame['nativeNodeScreen']).reshape(4,2)
                affine=np.linalg.lstsq(np.c_[actual,np.ones(4)],screen,rcond=None)[0]
                req=np.c_[r['native']['requested'],np.ones(4)]@affine
                for n in (1,2):
                    for point,color in [(req[n],'#00f080'),(screen[n],'#ff9200')]:
                        x,y=point/2; x+=j*512;y+=80
                        draw.ellipse((x-4,y-4,x+4,y+4),outline=color,width=2)
                s=r['native'];draw.text((j*512+8,604),f"{s['status']} | wrist {s['wristError']:.4f} | palm {s['palmErrorDeg']:.1f}deg",fill='black')
                draw.text((j*512+8,626),', '.join(s['reasons'])[:76] or 'within current checks',fill='#9b2424' if s['reasons'] else '#006633')
            proc.stdin.write(canvas.tobytes())
        proc.stdin.close();error=proc.stderr.read();code=proc.wait()
        if code:raise RuntimeError(error.decode(errors='replace'))
    print(output)
    for s in summaries:print(s['case'],'RMS',s['wristRMS'],'met',s['spatialTrackingMet'],'high',s['overheadFrame28']['wristError'])


if __name__=='__main__':run(Path(sys.argv[1]))
