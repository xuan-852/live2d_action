"""Separated old captures and new persistent lifecycle review."""
import sys,json,subprocess
from pathlib import Path
from PIL import Image,ImageDraw
import imageio_ffmpeg

def writer(path):
    return subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x720','-r','30','-i','-',
        '-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)

def picture(path,title,subtitle):
    out=Image.new('RGB',(960,720),'#202733');d=ImageDraw.Draw(out)
    d.text((12,8),title,fill='white');d.text((12,30),subtitle,fill='white')
    if path:
        with Image.open(path) as im:out.paste(im.convert('RGB').resize((640,640)),(160,80))
    return out

def finish(proc):
    proc.stdin.close();err=proc.stderr.read();code=proc.wait()
    if code:raise RuntimeError(err.decode(errors='replace'))

def run(root,previous):
    old=root/'01-old-sources-separated.mp4';proc=writer(old)
    for entry in json.loads((previous/'checks.json').read_text()):
        if not entry['name'].startswith('sample-'):continue
        for _ in range(30):proc.stdin.write(picture(None,'INDEPENDENT CAPTURE START: '+entry['name'],'Model/physics RESET here. Black separator is NOT part of source motion.').tobytes())
        for i in range(entry['frames']):proc.stdin.write(picture(previous/entry['name']/f'{i:04}.png',entry['name'],f'source time {i/30:.2f}s | no cross-clip connection').tobytes())
    finish(proc)
    new=root/'02-persistent-lifecycle.mp4';proc=writer(new)
    labels=json.loads((root/'timeline.json').read_text())['labels']
    for i,label in enumerate(labels):
        im=picture(root/'capture'/f'{i:04}.png','PERSISTENT MODEL / PHYSICS | NO RESET',f'{i/30:.2f}s | {label["source"]} | {label["phase"]}')
        if i in (54,150,250,400,650):im.save(root/f'lifecycle-{i:04}.png')
        proc.stdin.write(im.tobytes())
    finish(proc)
    page='<!doctype html><meta charset="utf-8"><title>边界与生命周期检查</title><style>body{background:#202733;color:white;font:18px sans-serif;margin:24px}video{width:100%;max-width:1200px}</style><h1>边界定位与完整生命周期</h1>'
    page+='<h2>1. 旧五条独立录像：明确分隔</h2><p>黑屏是人为分隔，不是动作停顿；检查每段内部是否仍有突变。</p><video controls src="01-old-sources-separated.mp4"></video>'
    page+='<h2>2. 同一物理会话：进入→源动作→退出→稳定</h2><p>五条同设置，不逐条调参；观察进入/源内/退出/稳定阶段分别在哪里异常。仅头身，非训练结果。</p><video controls src="02-persistent-lifecycle.mp4"></video>'
    (root/'review.html').write_text(page,encoding='utf-8');print(root/'review.html')

if __name__=='__main__':run(Path(sys.argv[1]),Path(sys.argv[2]))
