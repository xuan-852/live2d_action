"""Whole comparison gallery with local media validation and absolute relation diagnostics."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import imageio_ffmpeg
from PIL import Image,ImageDraw
from tools.whole.batch import read,sample_document,CASES


def source_at(run,name,t):
    doc=read(run/'inputs'/name/(name+'.json'))
    ref=read(run/'inputs'/name/(name+'.json.reference.json'))
    phase=float(np.interp(t,[f['t'] for f in doc['frames']],ref['sourceFrameTimes']))
    lo=min(int(phase),len(ref['points'])-1); hi=min(lo+1,len(ref['points'])-1); w=phase-lo
    return {k:np.array(v)*(1-w)+np.array(ref['points'][hi][k])*w for k,v in ref['points'][lo].items()},ref


def source_image(run,name,t):
    points,ref=source_at(run,name,t)
    array=np.array(list(points.values())); x0,y0,x1,y1=ref['bounds']
    # Frozen clip bounds, no frame-by-frame rescale. XY only, source mirrored metadata unchanged.
    span=max(x1-x0,y1-y0)*1.25; centre=np.array([(x0+x1)/2,(y0+y1)/2])
    image=Image.new('RGB',(512,512),(28,31,38));draw=ImageDraw.Draw(image)
    def xy(p):
        v=(np.array(p[:2])-centre)*430/span
        return (256+v[0],270-v[1])
    for a,b in ref['edges']:
        if a in points and b in points:draw.line([xy(points[a]),xy(points[b])],fill='white',width=3)
    for k,v in points.items():
        x,y=xy(v);draw.ellipse((x-4,y-4,x+4,y+4),fill=(180,200,255))
    draw.text((8,8),f'{name} source t={t:.2f}',fill='white')
    return image


def relation_rows(folder):
    rows=[]
    for path in sorted(folder.glob('[0-9][0-9][0-9][0-9].json')):
        f=read(path); n=np.array(f['wholeArms']['nativeChain']).reshape(4,2)
        p=np.array(f['wholeArms']['patchChain']).reshape(4,2)
        shoulder=(n[0]+p[0])/2; scale=np.linalg.norm(n[0]-p[0])
        rows.append({'t':f['actionTime'],'nativeWristRelativeShoulderCentre':((n[2]-shoulder)/scale).tolist(),
            'patchWristRelativeShoulderCentre':((p[2]-shoulder)/scale).tolist(),
            'wristSpreadShoulderWidths':float(np.linalg.norm(n[2]-p[2])/scale),
            'note':'absolute relations retained; shoulder centre is not anatomical chest point'})
    return rows


def correspondence(run,name,label):
    """Same-phase absolute two-arm relations; no correlation-only acceptance gate."""
    rows=relation_rows(run/(name+'-'+label)); records=[]
    for row in rows:
        points,_=source_at(run,name,row['t'])
        # Reference names follow the recovered compiler's explicit side binding.
        a,b=points['左腕'][:2],points['右腕'][:2]
        centre=(a+b)/2; scale=np.linalg.norm(a-b)
        if scale<1e-8: raise ValueError('Degenerate source shoulder span')
        native=(points['左手首'][:2]-centre)/scale
        patch=(points['右手首'][:2]-centre)/scale
        actual_n=np.array(row['nativeWristRelativeShoulderCentre'])
        actual_p=np.array(row['patchWristRelativeShoulderCentre'])
        records.append({'t':row['t'],'sourceNativeWrist':native.tolist(),'sourcePatchWrist':patch.tolist(),
            'actualNativeWrist':actual_n.tolist(),'actualPatchWrist':actual_p.tolist(),
            'nativePositionErrorShoulderWidths':float(np.linalg.norm(actual_n-native)),
            'patchPositionErrorShoulderWidths':float(np.linalg.norm(actual_p-patch)),
            'sourceSpread':float(np.linalg.norm(native-patch)),
            'actualSpread':row['wristSpreadShoulderWidths']})
    return {'rows':records,'scope':'absolute material-vs-source relations; proxy drift and neutral body differences remain',
            'meanNativePositionError':float(np.mean([r['nativePositionErrorShoulderWidths'] for r in records])),
            'meanPatchPositionError':float(np.mean([r['patchPositionErrorShoulderWidths'] for r in records])),
            'meanSpreadError':float(np.mean([abs(r['sourceSpread']-r['actualSpread']) for r in records])),
            'semanticAcceptance':'unknown; no automatic pass'}


def build(run):
    results=read(run/'results.json'); gallery=run/'review';gallery.mkdir(exist_ok=True)
    sections=[]; concat=[]
    for name in CASES:
        if name not in results:continue
        count=results[name]['restored']['frames']; sourcefolder=gallery/(name+'-source');sourcefolder.mkdir(exist_ok=True)
        for i in range(count):source_image(run,name,i/30).save(sourcefolder/f'{i:04d}.png')
        target=gallery/(name+'-whole.mp4')
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-filter_complex_threads','1',
            '-framerate','30','-i',str(sourcefolder/'%04d.png'),
            '-framerate','30','-i',str(run/(name+'-restored')/'%04d.png'),
            '-framerate','30','-i',str(run/(name+'-chain')/'%04d.png'),
            '-filter_complex','[1:v]scale=512:512[b];[2:v]scale=512:512[c];[0:v][b][c]hstack=inputs=3[v]',
            '-map','[v]','-c:v','libx264','-threads','1','-crf','19','-pix_fmt','yuv420p',
            '-movflags','+faststart',str(target)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        concat.append(f"file '{target.as_posix()}'")
        for label in ('restored','chain'):
            (gallery/(name+'-'+label+'-relations.json')).write_text(json.dumps(relation_rows(run/(name+'-'+label)),indent=1)+'\n')
            (gallery/(name+'-'+label+'-correspondence.json')).write_text(json.dumps(correspondence(run,name,label),indent=1)+'\n')
        held='heldout' if name in ['novel-12','novel-14','novel-21'] else 'development'
        sections.append(f'<section><h2>{name} · {held}</h2><video controls preload="metadata" src="{target.name}"></video>'
            f'<p>左：原参考骨架；中：恢复的整体基线；右：共同整臂候选。时间均为本轮动作时间。</p>'
            f'<a href="{target.name}">直接播放 MP4</a><button onclick="mark(this)">记录当前时间</button>'
            f'<textarea data-case="{name}" placeholder="记录表达、肘路线、肩根、手部与节奏问题"></textarea></section>')
    listing=gallery/'concat.txt';listing.write_text('\n'.join(concat)+'\n')
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','concat','-safe','0','-i',str(listing),
        '-c','copy','-movflags','+faststart',str(gallery/'all-whole.mp4')],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    page='''<!doctype html><meta charset="utf-8"><title>整体能力恢复对照</title>
<style>body{background:#20232b;color:white;font:15px sans-serif;margin:24px}video{width:min(100%,1200px)}section{margin:25px 0}textarea{display:block;width:90%;height:70px}a{color:#aaccff}</style>
<h1>整体能力恢复：全部8案例</h1><p>不按手势分类加例外。局部连续性收益与代价保留；本页不宣称自然度或表达通过。</p>
<p><a href="all-whole.mp4">一次播放全部案例</a> · 视频可在播放器全屏放大；原图在父目录。</p>'''+''.join(sections)+'''
<button onclick="exportNotes()">导出反馈JSON</button><script>
function mark(b){let s=b.closest('section'),v=s.querySelector('video'),t=s.querySelector('textarea');t.value+='\\n@'+v.currentTime.toFixed(3)+'s ';t.focus();}
document.querySelectorAll('textarea').forEach(t=>{t.value=localStorage.getItem('whole-'+t.dataset.case)||'';t.oninput=()=>localStorage.setItem('whole-'+t.dataset.case,t.value);});
function exportNotes(){let n=[...document.querySelectorAll('textarea')].map(t=>({case:t.dataset.case,note:t.value}));let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(n,null,2)],{type:'application/json'}));a.download='whole-feedback.json';a.click();}
</script>'''
    for name in results:
        if not (gallery/(name+'-whole.mp4')).is_file():raise FileNotFoundError(name)
    (gallery/'index.html').write_text(page,encoding='utf-8')
    print(gallery/'index.html')


if __name__=='__main__':build(Path(sys.argv[1]))
