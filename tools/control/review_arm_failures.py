"""Existing failure footage only; labelled slow playback, no synthetic frames."""
import json,sys,subprocess
from pathlib import Path
from PIL import Image,ImageDraw
import imageio_ffmpeg

def run(output,source,visible):
    out=output/'failure-diagnosis-review.mp4'
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','10','-i','-',
         '-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out)]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE) as proc:
        for case,start,end in [('yayoi-right-hand-rise1',0,20),('makoto-hair',34,49),('makoto-guard1',0,24)]:
            curves=json.loads((source/case/'with-arms-curves.json').read_text())
            for i in range(start,end+1):
                canvas=Image.new('RGB',(1280,720),'#eee');d=ImageDraw.Draw(canvas)
                d.text((12,6),f'{case} | ORIGINAL frame {i} / action {i/30:.3f}s | 3x SLOW PLAYBACK, no new frames',fill='black')
                vals={k:v['values'][i] for k,v in zip(curves['ids'],curves['channels'])}
                d.text((12,28),'Native Param97 %.2f / Param99 %.2f'%(vals['Param97'],vals['Param99']),fill='black')
                for col,frame in enumerate([max(start,i-1),i]):
                    d.text((col*640+12,49),('PREVIOUS ' if col==0 else 'CURRENT ')+str(frame),fill='black')
                    with Image.open(visible/case/'capture'/f'{frame:04}.png') as im:
                        canvas.paste(im.convert('RGB').resize((640,640)),(col*640,80))
                proc.stdin.write(canvas.tobytes())
        proc.stdin.close();err=proc.stderr.read();code=proc.wait()
        if code:raise RuntimeError(err.decode(errors='replace'))
    print(out)

if __name__=='__main__':run(*(Path(x) for x in sys.argv[1:4]))
