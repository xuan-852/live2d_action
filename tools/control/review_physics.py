"""Present captured heldout trajectories; NOT learned-policy rollouts."""
import subprocess
import sys
from pathlib import Path
import imageio_ffmpeg
from PIL import Image, ImageDraw


def run(root):
    output=root/'physics-off-on-review.mp4'
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','rawvideo','-vcodec','rawvideo',
             '-pix_fmt','rgb24','-s','1024x568','-r','30','-i','-',
             '-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',
             '-movflags','+faststart',str(output)]
    with subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE) as proc:
        for ep in (4,5):
            for frame in range(121):
                canvas=Image.new('RGB',(1024,568),'#eeeeee')
                draw=ImageDraw.Draw(canvas)
                draw.text((12,8),'CAPTURE INPUT REPLAY - NOT A TRAINED POLICY | episode '+str(ep),fill='black')
                for side,mode in enumerate(('off','physics')):
                    draw.text((side*512+12,30),f'Physics {mode.upper()} | action time {frame/30:.2f}s | 30Hz',fill='black')
                    with Image.open(root/f'ep{ep}-{mode}'/f'{frame:04}.png') as im:
                        canvas.paste(im.convert('RGB').resize((512,512)),(side*512,56))
                proc.stdin.write(canvas.tobytes())
        proc.stdin.close()
        error=proc.stderr.read()
        code=proc.wait()
        if code:raise RuntimeError(error.decode(errors='replace'))
    print(output)


if __name__=='__main__':run(Path(sys.argv[1]))
