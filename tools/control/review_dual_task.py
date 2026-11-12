"""Fixed dual-wrist goals, unverified visibility, original action-time playback."""
import sys,json,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image,ImageDraw
import imageio_ffmpeg
from tools.control.task_objective import hands


def run(root):
    spec=json.loads((root/'goals.json').read_text());radius=spec['radius']
    frames=[json.loads((root/'baseline'/f'{i:04}.json').read_text(encoding='utf-8-sig')) for i in range(241)]
    points=np.array([f['wholeArms']['nativeChain'][2:6] for f in frames]).reshape(-1,2)
    screens=np.array([f['actualScreen'] for f in frames]).reshape(-1,2)
    x=np.c_[points,np.ones(len(points))];transform=np.linalg.lstsq(x,screens,rcond=None)[0]
    residual=float(abs(x@transform-screens).max())
    if residual>.1:raise ValueError('Projection check failed')
    (root/'projection-check.json').write_text(json.dumps({'maxPixelResidual':residual}))
    output=root/'dual-task-review.mp4'
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','rawvideo','-pix_fmt','rgb24','-s','768x856','-r','30','-i','-',
        '-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    with subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE) as proc:
        streak=0
        for i,frame in enumerate(frames):
            phase=min(i//80,2);goal=np.array(spec['goals'][phase]);actual=hands(frame)
            if i%80==0:streak=0
            streak=streak+1 if (np.linalg.norm(actual-goal,axis=1)<=radius).all() else 0
            canvas=Image.new('RGB',(768,856),'#eee');d=ImageDraw.Draw(canvas)
            d.text((8,7),f'FIXED DUAL-HAND BASELINE | phase {phase} | action {i/30:.2f}s',fill='black')
            d.text((8,27),f'Joint material hold {streak}/15 frames | VISIBILITY UNVERIFIED | NOT TRAINED POLICY',fill='black')
            d.text((8,47),'Cyan: native wrist goal. Pink: patch wrist goal. White dots: material points.',fill='black')
            with Image.open(root/'baseline'/f'{i:04}.png') as im:canvas.paste(im.convert('RGB').resize((768,768)),(0,88))
            pg=np.c_[goal,np.ones(2)]@transform*.75+[0,88];pa=np.c_[actual,np.ones(2)]@transform*.75+[0,88]
            rx=radius*np.linalg.norm(transform[0])*.75;ry=radius*np.linalg.norm(transform[1])*.75
            for j,color in enumerate(('#00ffff','#ff66cc')):
                gx,gy=pg[j];ax,ay=pa[j]
                d.ellipse((gx-rx,gy-ry,gx+rx,gy+ry),outline=color,width=2)
                d.ellipse((ax-3,ay-3,ax+3,ay+3),fill='white')
            proc.stdin.write(canvas.tobytes())
        proc.stdin.close();error=proc.stderr.read();code=proc.wait()
        if code:raise RuntimeError(error.decode(errors='replace'))
    print(output)


if __name__=='__main__':run(Path(sys.argv[1]))
