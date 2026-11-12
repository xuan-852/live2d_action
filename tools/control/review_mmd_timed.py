"""Source-clock MMD comparison and measured retention; no visual pass inference."""
import sys, json, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image, ImageDraw
import imageio_ffmpeg
from tools.control.timed_arm import heading
from tools.expression.unity import write_spec


def run(root, old):
    sdk=root/'sdk'
    doc=json.loads((root/'source-map.json').read_text(encoding='utf-8-sig'))
    rows=[json.loads(x) for x in (sdk/'track.jsonl').read_text().splitlines()]
    selected=[x for x in rows if x['stage']=='mmd-source']
    if len(selected)!=len(doc['sides']['anatomicalLeft']['frames']):
        raise ValueError('Source samples missing')
    ids=['ParamAngleX','ParamAngleY','ParamAngleZ','ParamBodyAngleX','ParamBodyAngleY','ParamBodyAngleZ']
    report=dict(sourceTimesExact=all(abs(r['sourceTime']-doc['sides']['anatomicalLeft']['frames'][i]['seconds'])<1e-8 for i,r in enumerate(selected)),
                frames=len(selected),sides={},headBodyRequestedMaxError=0.,visualAcceptance=False,
                oldComparisonScope='historical converter uses different actuator/projection capability and initial physics history; not a controller-only causal ablation')
    for r in rows:
        f=json.loads((sdk/'capture'/f"{r['frame']:04}.json").read_text(encoding='utf-8-sig'))
        req={v['id']:v['value'] for v in f['requested']}
        report['headBodyRequestedMaxError']=max(report['headBodyRequestedMaxError'],max(abs(req[k]-v) for k,v in zip(ids,r['headBodyRequest'])))
    for side in ('native','patch'):
        actual=np.array([r[side]['actual'] for r in selected]); request=np.array([r[side]['requested'] for r in selected])
        angles=lambda p:np.degrees(np.unwrap(np.radians([heading(n) for n in p])))
        metrics={}
        for key in selected[0][side]['metrics']:
            metrics[key]=max(r[side]['metrics'][key] for r in selected)
        final=rows[-15:]
        report['sides'][side]=dict(wristRMS=float(np.sqrt(np.mean(np.sum((actual[:,2]-request[:,2])**2,axis=1)))),
            allSourceTrackingFramesMet=all(r[side]['wristError']<=.005 and r[side]['elbowError']<=.005 and r[side]['palmErrorDeg']<=4 for r in selected),
            sourceTrackingMetFrames=sum(r[side]['wristError']<=.005 and r[side]['elbowError']<=.005 and r[side]['palmErrorDeg']<=4 for r in selected),
            requestWristTravel=float(np.linalg.norm(np.diff(request[:,2],axis=0),axis=1).sum()),
            actualWristTravel=float(np.linalg.norm(np.diff(actual[:,2],axis=0),axis=1).sum()),
            requestedPalmRange=float(np.ptp(angles(request))),actualPalmRange=float(np.ptp(angles(actual))),
            sourceFailedFrames=[r['frame'] for r in selected if r[side]['status']=='failed'],
            sourceLimitedFrames=sum(r[side]['status']=='limited' for r in selected),
            sourceMaxima=metrics,sourceReasons=sorted({reason for r in selected for reason in r[side]['reasons']}),
            finalHoldMet=all(r[side]['positionReached'] and r[side]['terminalStateMet'] for r in final),
            finalWristError=rows[-1][side]['wristError'])
    write_spec(root/'mmd-assessment.json',report)
    # Real source-space arm diagrams, with one fixed framing across the entire clip.
    source=[]
    for side in ('anatomicalLeft','anatomicalRight'):
        source.append(np.array([[f['target']['sourceJointScreen'][k] for k in ('shoulder','elbow','wrist','knuckle')] for f in doc['sides'][side]['frames']]))
    points=np.concatenate([p.reshape(-1,2) for p in source]); center=(points.min(0)+points.max(0))/2
    scale=min(430/max(np.ptp(points[:,0]),.01),450/max(np.ptp(points[:,1]),.01))
    output=root/'mmd-source-old-new.mp4'
    if output.exists(): raise FileExistsError(output)
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s','1536x640','-r','30','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    with subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE) as proc:
        for i,r in enumerate(selected):
            canvas=Image.new('RGB',(1536,640),'#eeeeee');draw=ImageDraw.Draw(canvas)
            draw.text((10,8),f"makoto-hair | original source {r['sourceTime']:.3f}s | sample {i} | 30Hz, no retiming",fill='black')
            draw.text((10,28),'Historical converter comparison is NOT an isolated controller ablation. All failed samples retained.',fill='black')
            titles=['SOURCE ARM PROJECTION (fixed independent scale)','HISTORICAL CONVERTER / original capture','TIMED CONTROLLER / live SDK feedback']
            for j,title in enumerate(titles):draw.text((j*512+8,52),title,fill='black')
            for j,p in enumerate(source):
                xy=(p[i]-center)*[scale,-scale]+[256,330];pts=[tuple(x) for x in xy]
                draw.line(pts,fill='#1262b5' if j==0 else '#c66c12',width=3)
                for x,y in pts:draw.ellipse((x-4,y-4,x+4,y+4),outline='black',width=2)
            for col,path in [(1,old/f'{i:04}.png'),(2,sdk/'capture'/f"{r['frame']:04}.png")]:
                with Image.open(path) as im:canvas.paste(im.convert('RGB').resize((512,512)),(col*512,76))
            for j,side in enumerate(('native','patch')):
                s=r[side]; draw.text((1032,594+j*20),f"{side}: {s['status']} wrist {s['wristError']:.4f} palm {s['palmErrorDeg']:.1f}deg",fill='#932222')
            draw.text((10,604),'native/anatomicalLeft=blue; patch/right=orange. Source length shortening is retained in requests.',fill='black')
            proc.stdin.write(canvas.tobytes())
        proc.stdin.close();error=proc.stderr.read();code=proc.wait()
        if code:raise RuntimeError(error.decode(errors='replace'))
    print(json.dumps(report,indent=2));print(output)


if __name__=='__main__':run(Path(sys.argv[1]),Path(sys.argv[2]))
