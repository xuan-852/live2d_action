"""Node-only review, not rendered Live2D or natural-motion certification."""
import sys,json,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image,ImageDraw
import imageio_ffmpeg
from tools.control.node_policy import demonstration

def run(root):
    data=np.load(root/'tasks.npz');home=data['home'];tasks=data['tasks']
    tracks=json.loads((root/'tracks.json').read_text())
    scale=np.linalg.norm(home[0,0]-home[1,0]);origin=home[:,0].mean(0)
    output=root/'upper-node-review.mp4'
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','rawvideo','-pix_fmt','rgb24','-s','1200x520','-r','30','-i','-',
        '-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE) as proc:
        for ep in (112,113,114,115):
            chosen={t['method']:t for t in tracks if t['episode']==ep}
            for i in range(61):
                canvas=Image.new('RGB',(1200,520),'#202733');d=ImageDraw.Draw(canvas)
                d.text((12,8),f'UPPER NODE MODEL | heldout episode {ep} | {i/30:.2f}s | NODE SIMULATION ONLY, NOT SDK EXECUTION',fill='white')
                for col,method in enumerate(('synthetic teacher','untrained','trained')):
                    d.text((col*400+20,38),method,fill='white')
                    nodes=demonstration(home,tasks[ep],i/60) if col==0 else np.array(chosen[method]['nodes'][i])
                    def point(p):return tuple(((p-origin)/scale*np.array([160,-160])+[col*400+200,210]).tolist())
                    for side,color in enumerate(('#55ddff','#ff88bb')):
                        pts=[point(p) for p in nodes[side]]
                        d.line(pts,fill=color,width=3)
                        for x,y in pts:d.ellipse((x-4,y-4,x+4,y+4),fill=color)
                        gx,gy=point(chosen['trained']['goalWrists'][side])
                        d.ellipse((gx-7,gy-7,gx+7,gy+7),outline='white',width=2)
                proc.stdin.write(canvas.tobytes())
        proc.stdin.close();err=proc.stderr.read();code=proc.wait()
        if code:raise RuntimeError(err.decode(errors='replace'))
    print(output)

if __name__=='__main__':run(Path(sys.argv[1]))
