"""Show actual SDK control rollouts with targets and measured landmarks."""
import sys,json,subprocess
from pathlib import Path
import imageio_ffmpeg
from PIL import Image,ImageDraw


def run(root):
    output=root/'closed-loop-review.mp4'
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24',
         '-s','1024x600','-r','30','-i','-','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE) as proc:
        for ep in range(2):
            for i in range(181):
                canvas=Image.new('RGB',(1024,600),'#eeeeee');draw=ImageDraw.Draw(canvas)
                phase=['TARGET A','TARGET B','RETURN'][min(i//60,2)]
                draw.text((8,5),f'ACTUAL SDK FEEDBACK | episode {ep} | {phase} | {i/30:.2f}s',fill='black')
                draw.text((8,24),'Green rings: goal elbow/wrist. Orange dots: actual. Lines: error (not a skeleton).',fill='black')
                for side,method in enumerate(('numerical','history')):
                    folder=root/f'ep{ep}-{method}'
                    frame=json.loads((folder/f'{i:04}.json').read_text(encoding='utf-8-sig'))
                    with Image.open(folder/f'{i:04}.png') as im:canvas.paste(im.convert('RGB').resize((512,512)),(side*512,88))
                    actual=frame['wholeArms']['nativeChain'][2:6];goal=frame['goal']
                    error=(sum((a-b)**2 for a,b in zip(actual,goal))/4)**.5
                    draw.text((side*512+8,49),f'{method.upper()} | 4-coordinate RMS {error:.5f}',fill='black')
                    for j in range(2):
                        gx,gy=frame['goalScreen'][2*j:2*j+2];ax,ay=frame['actualScreen'][2*j:2*j+2]
                        gx=gx/2+side*512;gy=gy/2+88;ax=ax/2+side*512;ay=ay/2+88
                        draw.line((gx,gy,ax,ay),fill='#ffad33',width=2)
                        draw.ellipse((gx-6,gy-6,gx+6,gy+6),outline='#00ff77',width=2)
                        draw.ellipse((ax-2,ay-2,ax+2,ay+2),fill='#ffad33')
                proc.stdin.write(canvas.tobytes())
        proc.stdin.close();err=proc.stderr.read();code=proc.wait()
        if code:raise RuntimeError(err.decode(errors='replace'))
    print(output)


if __name__=='__main__':run(Path(sys.argv[1]))
