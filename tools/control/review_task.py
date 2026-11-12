"""Overlay fixed hand target regions on existing qualification captures."""
import sys,json,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image,ImageDraw
import imageio_ffmpeg
from tools.control.task_objective import hands


def run(source,output):
    protocol=json.loads((output/'protocol.json').read_text())
    fs={m:[json.loads((source/m/f'{i:04}.json').read_text(encoding='utf-8-sig')) for i in range(241)] for m in ('simultaneous','head-leading')}
    points=np.array([f['wholeArms']['nativeChain'][2:6] for f in fs['simultaneous']]).reshape(-1,2)
    screen=np.array([f['actualScreen'] for f in fs['simultaneous']]).reshape(-1,2)
    design=np.c_[points,np.ones(len(points))]
    transform=np.linalg.lstsq(design,screen,rcond=None)[0]
    residual=float(abs(design@transform-screen).max())
    if residual>.1:raise ValueError('Camera projection not sufficiently affine')
    (output/'projection-check.json').write_text(json.dumps({'maxPixelResidual':residual,'scope':'same fixed camera; fitted to native same-frame screen readback'}))
    video=output/'fixed-hand-target-review.mp4'
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','rawvideo','-pix_fmt','rgb24','-s','1024x600','-r','30','-i','-',
         '-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE) as proc:
        streak=[0,0]
        for i in range(241):
            canvas=Image.new('RGB',(1024,600),'#eee');d=ImageDraw.Draw(canvas);phase=min(i//60,3)
            if i%60==0:streak=[0,0]
            d.text((8,6),f'FIXED TWO-HAND REGIONS | phase {phase} | {i/30:.2f}s | QUALIFICATION, NOT TRAINING',fill='black')
            d.text((8,26),'Cyan/pink rings: identified targets. White dots: material wrists, NOT proof of visibility.',fill='black')
            goals=np.array(protocol['goals'][phase]);radius=protocol['radius']
            for side,method in enumerate(fs):
                frame=fs[method][i];actual=hands(frame)
                streak[side]=streak[side]+1 if np.all(np.linalg.norm(actual-goals,axis=1)<=radius) else 0
                d.text((side*512+8,48),f'{method} | joint hold {streak[side]}/15 | VISUAL UNVERIFIED',fill='black')
                with Image.open(source/method/f'{i:04}.png') as im:canvas.paste(im.convert('RGB').resize((512,512)),(side*512,88))
                projected=np.c_[goals,np.ones(2)]@transform/2+[side*512,88]
                current=np.c_[actual,np.ones(2)]@transform/2+[side*512,88]
                rx=radius*np.linalg.norm(transform[0])/2;ry=radius*np.linalg.norm(transform[1])/2
                for j,color in enumerate(('#00ffff','#ff66cc')):
                    x,y=projected[j];a,b=current[j]
                    d.ellipse((x-rx,y-ry,x+rx,y+ry),outline=color,width=2)
                    d.ellipse((a-2,b-2,a+2,b+2),fill='white')
            proc.stdin.write(canvas.tobytes())
        proc.stdin.close();err=proc.stderr.read();code=proc.wait()
        if code:raise RuntimeError(err.decode(errors='replace'))
    print(video)


if __name__=='__main__':run(Path(sys.argv[1]),Path(sys.argv[2]))
