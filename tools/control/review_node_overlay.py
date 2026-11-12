"""Original versus bound-node overlay; no guessed camera projection."""
import sys,json,subprocess
from pathlib import Path
import imageio_ffmpeg
from PIL import Image,ImageDraw

def run(root):
    values=json.loads((root/'stimuli.json').read_text())['values']
    output=root/'node-joint-overlay.mp4'
    command=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','rawvideo','-pix_fmt','rgb24','-s','1536x856','-r','30','-i','-',
        '-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    with subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE) as proc:
        for i in range(361):
            frame=json.loads((root/'capture'/f'{i:04}.json').read_text(encoding='utf-8-sig'))
            image=Image.new('RGB',(1536,856),'#eeeeee');d=ImageDraw.Draw(image)
            section=['SHOULDER ONLY','ELBOW ONLY','WRIST ONLY','COMBINED'][min(i//90,3)]
            d.text((12,8),f'RANDOM PARAMETER / JOINT NODE VALIDATION | {section} | {i/30:.2f}s | PHYSICS OFF',fill='black')
            d.text((12,29),'Native [94,97,99]: '+str([round(x,2) for x in values[i][:3]])+' | Patch [shoulder,elbow,wrist]: '+str([round(x,2) for x in values[i][3:]]),fill='black')
            d.text((12,53),'LEFT: unchanged capture | RIGHT: frozen material-node chain | cyan native / pink patch | S shoulder E elbow W wrist P palm',fill='black')
            with Image.open(root/'capture'/f'{i:04}.png') as source:
                source=source.convert('RGB').resize((768,768));image.paste(source,(0,88));image.paste(source,(768,88))
            for key,color in [('nativeNodeScreen','#00ffff'),('patchNodeScreen','#ff66cc')]:
                coords=frame[key];points=[(768+coords[j]*.75,88+coords[j+1]*.75) for j in range(0,len(coords),2)]
                d.line(points,fill=color,width=3)
                for (x,y),label in zip(points,['S','E','W','P']):
                    d.ellipse((x-4,y-4,x+4,y+4),outline=color,width=2);d.text((x+6,y-10),label,fill=color)
            if i in (30,60,120,150,210,240,300,330):image.save(root/f'sample-{i:04}.png')
            proc.stdin.write(image.tobytes())
        proc.stdin.close();err=proc.stderr.read();code=proc.wait()
        if code:raise RuntimeError(err.decode(errors='replace'))
    print(output)

if __name__=='__main__':run(Path(sys.argv[1]))
