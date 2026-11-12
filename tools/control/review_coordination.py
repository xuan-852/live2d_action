"""Side-by-side coordination schedules, no learned-policy claim."""
import sys,subprocess
from pathlib import Path
import imageio_ffmpeg
from PIL import Image,ImageDraw

def run(root):
    output=root/'coordination-review.mp4'
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','rawvideo','-pix_fmt','rgb24','-s','1024x600','-r','30','-i','-',
        '-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    with subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE) as proc:
        for i in range(241):
            canvas=Image.new('RGB',(1024,600),'#eeeeee');d=ImageDraw.Draw(canvas)
            phase=['NATIVE SIDE LEADS','PATCH SIDE LEADS','BOTH ARMS','RETURN'][min(i//60,3)]
            d.text((8,7),f'DUAL ARM + HEAD/BODY | {phase} | {i/30:.2f}s',fill='black')
            d.text((8,26),'PROCEDURAL SCHEDULING, NOT LEARNED COORDINATION | actual SDK arm feedback',fill='black')
            for side,method in enumerate(('simultaneous','head-leading')):
                d.text((side*512+8,52),method.upper(),fill='black')
                with Image.open(root/method/f'{i:04}.png') as im:canvas.paste(im.convert('RGB').resize((512,512)),(side*512,88))
            proc.stdin.write(canvas.tobytes())
        proc.stdin.close();error=proc.stderr.read();code=proc.wait()
        if code:raise RuntimeError(error.decode(errors='replace'))
    print(output)

if __name__=='__main__':run(Path(sys.argv[1]))
