"""Fixed close-ups of both hands, original frames, unsupported tasks labelled."""
import sys,json,subprocess
from pathlib import Path
from PIL import Image,ImageDraw
import imageio_ffmpeg
import numpy as np

def run(root):
    sdk=root/'sdk';rows=[json.loads(x) for x in (sdk/'track.jsonl').read_text().splitlines()]
    frames=[json.loads((sdk/'capture'/f'{r["frame"]:04}.json').read_text(encoding='utf-8-sig')) for r in rows]
    crops={}
    for side in ('native','patch'):
        p=np.array([f[side+'NodeScreen'][4:8] for f in frames]).reshape(-1,2)
        lo=p.min(0)-90;hi=p.max(0)+90;c=(lo+hi)/2;span=max(hi-lo)
        crops[side]=tuple(int(v) for v in [c[0]-span/2,c[1]-span/2,c[0]+span/2,c[1]+span/2])
    out=root/'hand-capability-review.mp4'
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','30','-i','-',
        '-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out)]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE) as proc:
        for row in rows:
            im=Image.new('RGB',(1280,720),'#eee');d=ImageDraw.Draw(im)
            d.text((12,6),f'CURRENT HAND CAPABILITY | {row["stage"]} | {row["time"]:.2f}s | physics OFF',fill='black')
            special=row['stage'] not in ('neutral','wrist-plus','wrist-minus','recover')
            for col,side in enumerate(('native','patch')):
                note='NEUTRAL ONLY: TASK NOT EXPOSED' if side=='native' and special else 'ACTUAL CONTROLLER RESPONSE'
                d.text((col*640+12,30),side.upper()+' | '+note,fill='black')
                with Image.open(sdk/'capture'/f'{row["frame"]:04}.png') as source:im.paste(source.convert('RGB').crop(crops[side]).resize((640,640)),(col*640,80))
            proc.stdin.write(im.tobytes())
        proc.stdin.close();error=proc.stderr.read();code=proc.wait()
        if code:raise RuntimeError(error.decode(errors='replace'))
    print(out)

if __name__=='__main__':run(Path(sys.argv[1]))
