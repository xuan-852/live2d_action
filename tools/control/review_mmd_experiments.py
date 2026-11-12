"""Unified user review, continuous A/B plus complete individual-source reel."""
import sys,json,subprocess,html
from pathlib import Path
from PIL import Image,ImageDraw
import imageio_ffmpeg

def encode(root,output,pairs):
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','rawvideo','-pix_fmt','rgb24','-s','1280x720','-r','30','-i','-',
        '-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE) as proc:
        for left,right,count in pairs:
            timelines={n:json.loads((root/(n+'-timeline.json')).read_text()) for n in {left,right}}
            for i in range(count):
                canvas=Image.new('RGB',(1280,720),'#eee');d=ImageDraw.Draw(canvas)
                d.text((12,5),f'MMD DRIVING TEST | segment action time {i/30:.2f}s | no new training',fill='black')
                for col,name in enumerate((left,right)):
                    label=timelines[name]['labels'][i]
                    d.text((col*640+10,27),name,fill='black')
                    d.text((col*640+10,47),f'{label["source"]} | {label["kind"]} | pass {label["repeat"]+1}',fill='black')
                    with Image.open(root/name/f'{i:04}.png') as image:
                        canvas.paste(image.convert('RGB').crop((160,70,864,774)).resize((640,640)),(col*640,80))
                if i in (15,45,90,150,300):canvas.save(root/(output.stem+'-'+left+f'-{i:04}.png'))
                proc.stdin.write(canvas.tobytes())
        proc.stdin.close();err=proc.stderr.read();code=proc.wait()
        if code:raise RuntimeError(err.decode(errors='replace'))

def run(root):
    checks=json.loads((root/'checks.json').read_text());lookup={x['name']:x for x in checks}
    encode(root,root/'01-continuous-comparison.mp4',[('continuous-cut','continuous-smooth',lookup['continuous-cut']['frames'])])
    # Two distinct source clips per pair; shorter column clearly ends instead of fake hold.
    # Single-source reel uses same capture at two crops? Avoid redundant columns: paired baseline neutral pose.
    samples=[x for x in checks if x['name'].startswith('sample-')]
    out=root/'02-source-suitability.mp4'
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','rawvideo','-pix_fmt','rgb24','-s','960x720','-r','30','-i','-',
         '-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out)]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE) as proc:
        for entry in samples:
            name=entry['name']
            for i in range(entry['frames']):
                image=Image.new('RGB',(960,720),'#eee');d=ImageDraw.Draw(image)
                d.text((12,8),f'{name} | SOURCE CUT | {i/30:.2f}s / {entry["duration"]:.2f}s',fill='black')
                d.text((12,30),'Same frozen head/body mapping; arms neutral; no morphs; NOT generated behavior',fill='black')
                with Image.open(root/name/f'{i:04}.png') as im:image.paste(im.convert('RGB').resize((640,640)),(160,80))
                proc.stdin.write(image.tobytes())
        proc.stdin.close();err=proc.stderr.read();code=proc.wait()
        if code:raise RuntimeError(err.decode(errors='replace'))
    page='''<!doctype html><meta charset="utf-8"><title>MMD双实验评审</title><style>body{background:#202733;color:#eee;font:18px sans-serif;margin:24px}video{width:100%;max-width:1400px}section{margin-bottom:40px}</style><h1>MMD双实验评审</h1><p>实验1：直接切换 vs 连续过渡，同一时长，单次播放，物理历史不中断。实验2：5条不同源完整片段，同一冻结映射，不逐条调参。均仅头身驱动，无新训练、双臂或表情动作。</p>'''
    for title,name in [('实验1：连续连接','01-continuous-comparison.mp4'),('实验2：五条源适配','02-source-suitability.mp4')]:
        page+=f'<section><h2>{title}</h2><video controls preload="metadata" src="{name}"></video></section>'
    page+='<p>评审：连接是否跳变/急停？灵动是否保留？哪条源失真或过小？请记视频名和时间。源适配仅5条不同动作，不能当10–20条样本；对照合成段落不是自然对话真值。</p>'
    (root/'review.html').write_text(page,encoding='utf-8');print(root/'review.html')

if __name__=='__main__':run(Path(sys.argv[1]))
