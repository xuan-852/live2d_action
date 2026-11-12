"""Paired virtual-performer response. Repeat source segment with explicit cuts."""
import sys,json,subprocess
from pathlib import Path
from PIL import Image,ImageDraw
import imageio_ffmpeg

def run(root):
    data=json.loads((root/'signals.json').read_text());count=len(data['times'])
    output=root/'mmd-virtual-performer-review.mp4'
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','30','-i','-',
        '-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE) as proc:
        for repeat in range(3):
            for i in range(count):
                canvas=Image.new('RGB',(1280,720),'#eee');draw=ImageDraw.Draw(canvas)
                draw.text((12,6),f'MMD VIRTUAL PERFORMER | original action time {data["times"][i]:.2f}s | repeat {repeat+1}/3 (CUT, NOT CONTINUOUS)',fill='black')
                draw.text((12,26),'Head-local and body orientation from existing MMD extraction | arms neutral | no source expressions / audio',fill='black')
                for side,label in enumerate(('head-only','head-body-physics')):
                    draw.text((side*640+12,51),label,fill='black')
                    with Image.open(root/label/f'{i:04}.png') as image:
                        canvas.paste(image.convert('RGB').crop((160,70,864,774)).resize((640,640)),(side*640,80))
                if repeat==0 and i in (0,count//3,2*count//3,count-1):canvas.save(root/f'tracker-sample-{i:04}.png')
                proc.stdin.write(canvas.tobytes())
        proc.stdin.close();err=proc.stderr.read();code=proc.wait()
        if code:raise RuntimeError(err.decode(errors='replace'))
    print(output)

if __name__=='__main__':run(Path(sys.argv[1]))
