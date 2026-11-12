"""Exact-input native foreground route A/B."""
import sys,json,subprocess
from pathlib import Path
from PIL import Image,ImageDraw
import imageio_ffmpeg

def run(root):
    source=Path(json.loads((root/'protocol.json').read_text())['source']);out=root/'native-front-review.mp4'
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','30','-i','-',
        '-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out)]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE) as proc:
        for e in json.loads((root/'checks.json').read_text()):
            name=e['source']
            for i in range(e['frames']):
                image=Image.new('RGB',(1280,720),'#eee');d=ImageDraw.Draw(image)
                d.text((12,6),f'{name} | {i/30:.2f}s | identical parameters/physics/projection, native sorting only',fill='black')
                for col,(label,folder) in enumerate([('BEFORE: original layers',source/name/'with-arms'),('AFTER: native distal foreground',root/name/'capture')]):
                    d.text((col*640+12,40),label,fill='black')
                    with Image.open(folder/f'{i:04}.png') as im:image.paste(im.convert('RGB').resize((640,640)),(col*640,80))
                if i==e['frames']//2:image.save(root/(name+'-compare.png'))
                proc.stdin.write(image.tobytes())
        proc.stdin.close();err=proc.stderr.read();code=proc.wait()
        if code:raise RuntimeError(err.decode(errors='replace'))
    print(out)

if __name__=='__main__':run(Path(sys.argv[1]))
