"""Identical enlarged patch input, old versus revised axis-binding review."""
import json,sys,subprocess
from pathlib import Path
from PIL import Image,ImageDraw
import imageio_ffmpeg

def run(root):
    video=root/'patch-node-revision-review.mp4'
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','rawvideo','-pix_fmt','rgb24','-s','1536x856','-r','30','-i','-',
         '-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE) as proc:
        for i in range(361):
            canvas=Image.new('RGB',(1536,856),'#eee');d=ImageDraw.Draw(canvas)
            d.text((12,6),f'PATCH AXIS REVISION | SAME ENLARGED PATCH INPUT | {i/30:.2f}s | native unchanged',fill='black')
            for side,label in enumerate(('old-axis-large','revised-axis-large')):
                frame=json.loads((root/label/f'{i:04}.json').read_text(encoding='utf-8-sig'))
                d.text((side*768+12,28),label,fill='black')
                d.text((side*768+12,50),f'flips {frame["wholeArms"]["flippedTriangles"]} | min area ratio {frame["wholeArms"]["minTriangleAreaRatio"]:.3f}',fill='black')
                with Image.open(root/label/f'{i:04}.png') as image:canvas.paste(image.convert('RGB').resize((768,768)),(side*768,88))
                for key,color in [('nativeNodeScreen','#00ffff'),('patchNodeScreen','#ff66cc')]:
                    a=frame[key];pts=[(side*768+a[j]*.75,88+a[j+1]*.75) for j in range(0,len(a),2)]
                    d.line(pts,fill=color,width=2)
                    for (x,y),name in zip(pts,['S','E','W','P']):
                        d.ellipse((x-4,y-4,x+4,y+4),outline=color,width=2);d.text((x+5,y-8),name,fill=color)
            if i in (0,30,60,120,150,210,240,300,330):canvas.save(root/f'compare-{i:04}.png')
            proc.stdin.write(canvas.tobytes())
        proc.stdin.close();err=proc.stderr.read();code=proc.wait()
        if code:raise RuntimeError(err.decode(errors='replace'))
    print(video)

if __name__=='__main__':run(Path(sys.argv[1]))
