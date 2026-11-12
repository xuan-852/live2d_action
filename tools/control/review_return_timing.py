"""One-shot common action-clock comparison; no looping or hidden frame repetition."""
import sys,json,subprocess
from pathlib import Path
from PIL import Image,ImageDraw
import imageio_ffmpeg

def run(root):
    names=['original','slow-return','ease-stop'];data={n:json.loads((root/(n+'-times.json')).read_text()) for n in names}
    count=max(len(v['actionTimes']) for v in data.values());out=root/'return-timing-review.mp4'
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','rawvideo','-pix_fmt','rgb24','-s','1536x640','-r','30','-i','-',
        '-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out)]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE) as proc:
        for i in range(count):
            canvas=Image.new('RGB',(1536,640),'#202733');d=ImageDraw.Draw(canvas)
            d.text((12,6),f'RETURN TIMING | action time {i/30:.2f}s | SINGLE PLAY | same amplitude and original physics',fill='white')
            for j,n in enumerate(names):
                d.text((j*512+8,29),n,fill='white')
                if i>=len(data[n]['actionTimes']):
                    d.text((j*512+50,230),'END OF CAPTURE (no loop / held frame)',fill='white');continue
                d.text((j*512+8,48),f'source time {data[n]["sourceTimes"][i]:.3f}s',fill='white')
                with Image.open(root/n/f'{i:04}.png') as im:canvas.paste(im.convert('RGB').crop((232,65,792,695)).resize((512,576)),(j*512,64))
            if i in (15,30,45,60,80):canvas.save(root/f'timing-{i:04}.png')
            proc.stdin.write(canvas.tobytes())
        proc.stdin.close();err=proc.stderr.read();code=proc.wait()
        if code:raise RuntimeError(err.decode(errors='replace'))
    print(out)

if __name__=='__main__':run(Path(sys.argv[1]))
