"""Whole heldout model-node continuations, fixed camera and sample ordering."""
import sys,json,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import imageio_ffmpeg

def run(root):
    tracks=json.loads((root/'tracks.json').read_text());names=sorted({t['source'] for t in tracks})
    home=np.load(root/'home.npy');centre=home[:,0].mean(0)
    output=root/'upper-sequence-review.mp4'
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','rawvideo','-pix_fmt','rgb24','-s','1600x600','-r','30','-i','-',
        '-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE) as proc:
        for name in names:
            cases={t['method']:t for t in tracks if t['source']==name};ref=cases['trained']['reference']
            for i in range(len(ref)):
                image=Image.new('RGB',(1600,600),'#202733');d=ImageDraw.Draw(image)
                d.text((12,8),f'UPPER ARM-NODE CONTINUATION | {name} | {i/30:.2f}s | '+('OBSERVED PREFIX' if i<30 else 'GENERATED')+' | NOT SDK',fill='white')
                for col,method in enumerate(('reference','hold','untrained','trained')):
                    d.text((col*400+12,40),method,fill='white')
                    nodes=np.array(ref[i] if method=='reference' else cases[method]['nodes'][i])
                    for side,color in enumerate(('#00ffff','#ff66cc')):
                        points=(nodes[side]-centre)*[1000,-1000]+[col*400+200,180]
                        d.line([tuple(p) for p in points],fill=color,width=3)
                        for x,y in points:d.ellipse((x-4,y-4,x+4,y+4),outline=color,width=2)
                proc.stdin.write(image.tobytes())
        proc.stdin.close();err=proc.stderr.read();code=proc.wait()
        if code:raise RuntimeError(err.decode(errors='replace'))
    print(output)

if __name__=='__main__':run(Path(sys.argv[1]))
