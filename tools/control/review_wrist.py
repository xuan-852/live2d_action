"""Wrist candidate close-up review, fixed crop and scale across both variants."""
import sys,json,subprocess
from pathlib import Path
from PIL import Image,ImageDraw
import imageio_ffmpeg

def run(root):
    frames={k:[json.loads((root/k/f'{i:04}.json').read_text(encoding='utf-8-sig')) for i in range(181)] for k in ('old-wrist','axis-wrist')}
    coords=[f['patchNodeScreen'] for seq in frames.values() for f in seq]
    xs=[a[j] for a in coords for j in (2,4,6)];ys=[a[j] for a in coords for j in (3,5,7)]
    cx=(min(xs)+max(xs))/2;cy=(min(ys)+max(ys))/2;size=max(max(xs)-min(xs),max(ys)-min(ys))+150
    crop=(round(cx-size/2),round(cy-size/2),round(cx+size/2),round(cy+size/2));scale=640/(crop[2]-crop[0])
    output=root/'wrist-closeup-review.mp4'
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','30','-i','-',
        '-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE) as proc:
        for i in range(181):
            canvas=Image.new('RGB',(1280,720),'#eee');d=ImageDraw.Draw(canvas)
            for side,label in enumerate(frames):
                d.text((side*640+10,8),f'{label} | {i/30:.2f}s | elbow/wrist/palm',fill='black')
                d.text((side*640+10,30),'Shape-axis candidate, NOT certified joint. Same crop/scale.',fill='black')
                with Image.open(root/label/f'{i:04}.png') as image:canvas.paste(image.convert('RGB').crop(crop).resize((640,640)),(side*640,80))
                a=frames[label][i]['patchNodeScreen'];points=[(side*640+(a[j]-crop[0])*scale,80+(a[j+1]-crop[1])*scale) for j in (2,4,6)]
                d.line(points,fill='#00ffff',width=2)
                for (x,y),n in zip(points,['E','W','P']):
                    d.ellipse((x-4,y-4,x+4,y+4),outline='#00ffff',width=2);d.text((x+5,y-10),n,fill='#00ffff')
            if i in (0,30,60,120,150):canvas.save(root/f'wrist-sample-{i:04}.png')
            proc.stdin.write(canvas.tobytes())
        proc.stdin.close();err=proc.stderr.read();code=proc.wait()
        if code:raise RuntimeError(err.decode(errors='replace'))
    print(output)

if __name__=='__main__':run(Path(sys.argv[1]))
