"""Old baseline versus calibrated input and isolated physics-weight variants."""
import sys,json,subprocess
from pathlib import Path
from PIL import Image,ImageDraw
import imageio_ffmpeg

def run(root):
    protocol=json.loads((root/'protocol.json').read_text());previous=Path(protocol['previousBaseline'])
    times=json.loads((root/'signals.json').read_text())['times']
    columns=[('OLD: clipped input',previous/'head-body-physics'),
             ('CALIBRATED: full physics',root/'calibrated-full'),
             ('SECONDARY weights 0.5',root/'secondary-half'),
             ('BODY + SECONDARY weights 0.5',root/'body-secondary-half')]
    output=root/'liveliness-calibration-review.mp4'
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','rawvideo','-pix_fmt','rgb24','-s','1920x720','-r','30','-i','-',
        '-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE) as proc:
        for repeat in range(3):
            for i,t in enumerate(times):
                canvas=Image.new('RGB',(1920,720),'#eee');d=ImageDraw.Draw(canvas)
                d.text((12,5),f'MMD LIVELINESS CALIBRATION | original time {t:.2f}s | repeat {repeat+1}/3 (CUT, NOT CONTINUOUS)',fill='black')
                d.text((12,25),'Same source/time; columns 2-4 SAME input. Secondary includes all non-body physics outputs. No new training.',fill='black')
                for j,(label,folder) in enumerate(columns):
                    d.text((j*480+10,52),label,fill='black')
                    with Image.open(folder/f'{i:04}.png') as image:
                        canvas.paste(image.convert('RGB').crop((232,65,792,812)).resize((480,640)),(j*480,80))
                if repeat==0 and i in (0,18,36,53):canvas.save(root/f'calibration-{i:04}.png')
                proc.stdin.write(canvas.tobytes())
        proc.stdin.close();err=proc.stderr.read();code=proc.wait()
        if code:raise RuntimeError(err.decode(errors='replace'))
    print(output)

if __name__=='__main__':run(Path(sys.argv[1]))
