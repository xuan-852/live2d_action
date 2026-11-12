"""Identical MMD requests/clock/budgets; compare actual SDK controller versions."""
import sys,json,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image,ImageDraw
import imageio_ffmpeg
from tools.control.timed_arm import heading
from tools.expression.unity import write_spec


def compare(before,after,output):
    output.mkdir(parents=True,exist_ok=False)
    roots=[before,after]
    protocols=[json.loads((p/'protocol.json').read_text()) for p in roots]
    if protocols[0]['nativeLimits']!=protocols[1]['nativeLimits']:raise ValueError('Different native execution budgets')
    if protocols[0]['limits']!=protocols[1]['limits']:raise ValueError('Different patch execution budgets')
    tracks=[[json.loads(x) for x in (p/'track.jsonl').read_text().splitlines() if json.loads(x)['stage']=='mmd-source'] for p in roots]
    if len(tracks[0])!=len(tracks[1]):raise ValueError('Different sample counts')
    maximum=0.;angle=0.
    for a,b in zip(*tracks):
        if (a['time'],a['sourceTime'])!=(b['time'],b['sourceTime']):raise ValueError('Retimed source')
        if not np.allclose(a['headBodyRequest'],b['headBodyRequest'],atol=1e-10):raise ValueError('Changed head/body')
        for side in ('native','patch'):
            x=np.array(a[side]['requested']);y=np.array(b[side]['requested'])
            maximum=max(maximum,float(np.max(abs(x[1:3]-y[1:3]))))
            angle=max(angle,float(abs((heading(x)-heading(y)+180)%360-180)))
    if maximum>1e-8 or angle>1e-6:raise ValueError('Changed target')
    report=dict(maxTargetDifference=maximum,maxHeadingDifference=angle,sourceFrames=len(tracks[0]),sameBudgets=True,results=[])
    for root,track in zip(roots,tracks):
        for side in ('native','patch'):
            rows=[r[side] for r in track]
            report['results'].append(dict(version=str(root),side=side,
                rms={k:float(np.sqrt(np.mean([r[k]**2 for r in rows]))) for k in ['wristError','elbowError','palmErrorDeg']},
                maxima={k:max(r[k] for r in rows) for k in ['wristError','elbowError','palmErrorDeg']},
                met={k:sum(r[k]<=limit for r in rows) for k,limit in [('wristError',.005),('elbowError',.005),('palmErrorDeg',4.)]},
                jointMet=sum(r['wristError']<=.005 and r['elbowError']<=.005 and r['palmErrorDeg']<=4 for r in rows),
                jointMissFrames=[i for i,r in enumerate(rows) if not(r['wristError']<=.005 and r['elbowError']<=.005 and r['palmErrorDeg']<=4)],
                failedFrames=[i for i,r in enumerate(rows) if r['status']=='failed'],
                reasons=sorted({reason for r in rows for reason in r['reasons']})))
    write_spec(output/'comparison.json',report)
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1024x656','-r','30','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output/'comparison.mp4')]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE) as proc:
        for rows in zip(*tracks):
            canvas=Image.new('RGB',(1024,656),'#eeeeee');draw=ImageDraw.Draw(canvas)
            draw.text((8,8),f"MMD {rows[0]['sourceTime']:.3f}s | SAME requests, timings, budgets, mapping, physics | no dropped failures",fill='black')
            draw.text((8,28),'Green=request wrist/elbow; orange=actual. Remaining elbow errors are reported, never declared reached.',fill='black')
            for j,(root,row,label) in enumerate(zip(roots,rows,['CHECKPOINT 06153c8 / clipped inverse step','BOUNDED inverse step / current'])):
                draw.text((j*512+8,52),label,fill='black')
                with Image.open(root/'capture'/f"{row['frame']:04}.png") as im:canvas.paste(im.convert('RGB').resize((512,512)),(j*512,80))
                f=json.loads((root/'capture'/f"{row['frame']:04}.json").read_text(encoding='utf-8-sig'))
                s=row['native'];actual=np.array(s['actual']);screen=np.array(f['nativeNodeScreen']).reshape(4,2)
                affine=np.linalg.lstsq(np.c_[actual,np.ones(4)],screen,rcond=None)[0]
                request=np.c_[s['requested'],np.ones(4)]@affine
                for n in (1,2):
                    for xy,color in [(request[n],'#00f080'),(screen[n],'#ff9200')]:
                        x,y=xy/2;x+=j*512;y+=80;draw.ellipse((x-4,y-4,x+4,y+4),outline=color,width=2)
                draw.text((j*512+8,604),f"{s['status']} | wrist {s['wristError']:.4f} elbow {s['elbowError']:.4f} palm {s['palmErrorDeg']:.2f}",fill='black')
                draw.text((j*512+8,626),', '.join(s['reasons'])[:77] or 'within current checks',fill='#9c2424' if s['reasons'] else '#006633')
            proc.stdin.write(canvas.tobytes())
        proc.stdin.close();error=proc.stderr.read();code=proc.wait()
        if code:raise RuntimeError(error.decode(errors='replace'))
    print(json.dumps(report,indent=2));print(output/'comparison.mp4')


if __name__=='__main__':compare(*map(Path,sys.argv[1:4]))
