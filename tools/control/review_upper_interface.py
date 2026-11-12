"""Requested node skeleton versus actual SDK capture, common fixed projection."""
import sys,json,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import imageio_ffmpeg

def run(root):
    targets=json.loads((root/'node-targets.json').read_text())['nodes']
    frames=[json.loads((root/'capture'/f'{i:04}.json').read_text(encoding='utf-8-sig')) for i in range(241)]
    x=np.array([f['wholeArms']['nativeChain'] for f in frames]).reshape(-1,2)
    y=np.array([f['nativeNodeScreen'] for f in frames]).reshape(-1,2)
    design=np.c_[x,np.ones(len(x))];matrix=np.linalg.lstsq(design,y,rcond=None)[0]
    if abs(design@matrix-y).max()>.1:raise ValueError('Camera check failed')
    output=root/'upper-node-interface-review.mp4'
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','rawvideo','-pix_fmt','rgb24','-s','1536x856','-r','30','-i','-',
        '-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE) as proc:
        for i,f in enumerate(frames):
            image=Image.new('RGB',(1536,856),'#202733');d=ImageDraw.Draw(image)
            d.text((12,8),f'UPPER NODE INTERFACE | {i/30:.2f}s | PROGRAMMED NODE TEST, NOT TRAINED GENERATION',fill='white')
            d.text((12,32),'LEFT: requested nodes only | RIGHT: actual SDK + requested solid / actual dotted markers',fill='white')
            with Image.open(root/'capture'/f'{i:04}.png') as im:image.paste(im.convert('RGB').resize((768,768)),(768,88))
            for side,color,key in [(0,'#00ffff','nativeNodeScreen'),(1,'#ff66cc','patchNodeScreen')]:
                nodes=np.array(targets[i][side]);screen=np.c_[nodes,np.ones(4)]@matrix*.75+[0,88]
                for col in (0,1):
                    points=[(float(a+col*768),float(b)) for a,b in screen];d.line(points,fill=color,width=3)
                    for a,b in points:d.ellipse((a-4,b-4,a+4,b+4),outline=color,width=2)
                actual=np.array(f[key]).reshape(4,2)*.75+[768,88]
                for a,b in actual:d.ellipse((a-2,b-2,a+2,b+2),fill='white')
            if i in (0,60,120,180,240):image.save(root/f'interface-{i:04}.png')
            proc.stdin.write(image.tobytes())
        proc.stdin.close();err=proc.stderr.read();code=proc.wait()
        if code:raise RuntimeError(err.decode(errors='replace'))
    print(output)

if __name__=='__main__':run(Path(sys.argv[1]))
