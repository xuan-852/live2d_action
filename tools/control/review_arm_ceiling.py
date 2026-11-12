"""Source arms / model target nodes / actual full model, all same source time."""
import sys,json,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import imageio_ffmpeg

def run(root):
    results=json.loads((root/'results.json').read_text());out=root/'arm-ceiling-review.mp4'
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','rawvideo','-pix_fmt','rgb24','-s','1536x720','-r','30','-i','-',
        '-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out)]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE) as proc:
        for result in results:
            case=root/result['source'];doc=json.loads((case/'source-map.json').read_text());targets=json.loads((case/'targets.json').read_text())
            frames=[json.loads((case/'with-arms'/f'{i:04}.json').read_text(encoding='utf-8-sig')) for i in range(result['frames'])]
            x=np.array([f['wholeArms']['nativeChain'] for f in frames]).reshape(-1,2);y=np.array([f['nativeNodeScreen'] for f in frames]).reshape(-1,2)
            matrix=np.linalg.lstsq(np.c_[x,np.ones(len(x))],y,rcond=None)[0]
            if abs(np.c_[x,np.ones(len(x))]@matrix-y).max()>.1:raise ValueError('Projection mismatch')
            allsource=np.array([row['target']['sourceJointScreen'][n] for s in ('anatomicalLeft','anatomicalRight') for row in doc['sides'][s]['frames'] for n in ('shoulder','elbow','wrist','knuckle')])
            centre=(allsource.min(0)+allsource.max(0))/2;sourceScale=min(430/max(np.ptp(allsource[:,0]),1e-6),480/max(np.ptp(allsource[:,1]),1e-6))
            alltarget=np.array(targets['nodes']).reshape(-1,2)
            tc=(alltarget.min(0)+alltarget.max(0))/2;targetScale=min(430/max(np.ptp(alltarget[:,0]),1e-6),480/max(np.ptp(alltarget[:,1]),1e-6))
            for i,f in enumerate(frames):
                im=Image.new('RGB',(1536,720),'#202733');d=ImageDraw.Draw(im)
                d.text((12,7),f'{result["source"]} | source time {i/30:.2f}s | SOURCE CUT BETWEEN CASES | NOT TRAINED',fill='white')
                for col,title in enumerate(('SOURCE ARM JOINTS','MODEL REQUEST NODES','LIVE2D + REQUEST / WHITE ACTUAL')):d.text((col*512+10,30),title,fill='white')
                with Image.open(case/'with-arms'/f'{i:04}.png') as image:im.paste(image.convert('RGB').resize((512,512)),(1024,100))
                for side,(s,color,key) in enumerate([('anatomicalLeft','#00ffff','nativeNodeScreen'),('anatomicalRight','#ff66cc','patchNodeScreen')]):
                    p=doc['sides'][s]['frames'][i]['target']['sourceJointScreen'];source=np.array([p[n] for n in ('shoulder','elbow','wrist','knuckle')])
                    pts=(source-centre)*[sourceScale,-sourceScale]+[256,350]
                    d.line([tuple(p) for p in pts],fill=color,width=3)
                    for a,b in pts:d.ellipse((a-4,b-4,a+4,b+4),outline=color,width=2)
                    target=np.array(targets['nodes'][i][side]);screen=np.c_[target,np.ones(4)]@matrix*.5+[0,100]
                    modelpts=(target-tc)*[targetScale,-targetScale]+[768,350]
                    d.line([tuple(p) for p in modelpts],fill=color,width=3)
                    d.line([(float(a+1024),float(b)) for a,b in screen],fill=color,width=3)
                    actual=np.array(f[key]).reshape(4,2)*.5+[1024,100]
                    for a,b in actual:d.ellipse((a-3,b-3,a+3,b+3),fill='white')
                diag=targets['diagnostics'][i]
                d.text((12,650),f'Patch angle limited: {diag["patchAngleClipped"]}; projection limited: {diag["patchProjectionClipped"]}',fill='white')
                d.text((12,674),f'Flips {f["wholeArms"]["flippedTriangles"]}; min area {f["wholeArms"]["minTriangleAreaRatio"]:.3f} | fixed model shoulder roots lose source translations',fill='white')
                if i in (0,result['frames']//2,result['frames']-1):im.save(case/f'compare-{i:04}.png')
                proc.stdin.write(im.tobytes())
        proc.stdin.close();err=proc.stderr.read();code=proc.wait()
        if code:raise RuntimeError(err.decode(errors='replace'))
    print(out)

if __name__=='__main__':run(Path(sys.argv[1]))
