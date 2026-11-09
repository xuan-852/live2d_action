"""Offline review and truthful proxy metrics for the frozen expression trial."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys,json,subprocess,datetime
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from tools.whole.batch import read
from tools.whole.review import source_at,source_image
from tools.whole.expression_trial import target
import imageio_ffmpeg
from PIL import Image,ImageDraw

def build(root):
    results=read(root/'results.json');protocol=read(root/'protocol.json')
    gallery=root/'review';gallery.mkdir(exist_ok=True);sections=[];concat=[];index={}
    for c,record in results.items():
        count=record['A']['frames'];sf=gallery/(c+'-source');sf.mkdir(exist_ok=True)
        for i in range(count):source_image(root,c,i/30).save(sf/f'{i:04d}.png')
        index[c]={'split':'heldout-this-trial' if c in protocol['heldoutThisTrial'] else 'exposed-regression', 'manualEdits':0}
        for label in ('A','B'):
            rows=[];actual=[];desired=[]
            for p in sorted((root/f'{c}-{label}').glob('[0-9][0-9][0-9][0-9].json')):
                f=read(p);n=np.array(f['wholeArms']['nativeChain']).reshape(4,2);b=np.array(f['wholeArms']['patchChain']).reshape(4,2)
                centre=(n[0]+b[0])/2;width=np.linalg.norm(n[0]-b[0]);v=(np.array([n[1],n[2],b[1],b[2]])-centre)/width
                s=target(source_at(root,c,f['actionTime'])[0]);actual.append(v);desired.append(s)
            actual=np.array(actual);desired=np.array(desired)
            last=f
            index[c][label]=record[label]|{
                'meanFourPointErrorShoulderWidths':float(np.linalg.norm(actual-desired,axis=2).mean()),
                'meanWristErrorShoulderWidths':float(np.linalg.norm(actual[:,[1,3]]-desired[:,[1,3]],axis=2).mean()),
                'spreadErrorShoulderWidths':float(np.abs(np.linalg.norm(actual[:,1]-actual[:,3],axis=1)-np.linalg.norm(desired[:,1]-desired[:,3],axis=1)).mean()),
                'wristRangeSum':float(np.ptp(actual[:,[1,3]],axis=0).sum()),
                'maxFourPointStepShoulderWidths':float(np.linalg.norm(np.diff(actual,axis=0),axis=2).max()),
                'endpointMaxParameterDeviationFromDefault':max(abs(x['value']-x['defaultValue']) for x in last['applied']),
                'endpointScope':'snapshot before shutdown; physics may remain nonneutral; not interrupt/reset test',
                'humanAcceptance':None,'visibility':'foreground diagnostic; no pixel-complete certification'}
        cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-filter_complex_threads','1']
        for folder in (sf,root/f'{c}-A',root/f'{c}-B'):cmd+=['-framerate','30','-i',str(folder/'%04d.png')]
        cmd+=['-filter_complex','[1:v]scale=512:512[a];[2:v]scale=512:512[b];[0:v][a][b]hstack=inputs=3[v]',
            '-map','[v]','-c:v','libx264','-threads','1','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(gallery/f'{c}.mp4')]
        subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        concat.append(f"file '{c}.mp4'")
        sections.append(f'<section id="{c}"><h2>{c} · {index[c]["split"]}</h2><video controls preload="metadata" src="{c}.mp4"></video><p>源参考 / A 当前基线 / B 整体位置关系适配；30Hz同动作时间。</p><button onclick="mark(this)">记录时间点</button><textarea data-id="{c}" placeholder="表达、运动、可见性问题"></textarea><a href="{c}.mp4">直接MP4</a></section>')
    (gallery/'concat.txt').write_text('\n'.join(concat),encoding='utf-8')
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','concat','-safe','0','-i',str(gallery/'concat.txt'),'-c','copy','-movflags','+faststart',str(gallery/'all.mp4')],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    nav=' '.join(f'<a href="#{c}">{c}</a>' for c in results)
    page='''<!doctype html><meta charset="utf-8"><title>整体表达适配 A/B</title><style>body{background:#20232b;color:#eee;font:16px sans-serif;margin:24px}a{color:#bcd4ff}video{width:100%;max-width:1536px}textarea{display:block;width:90%;height:75px}section{margin:35px 0}nav{position:sticky;top:0;background:#20232b;padding:12px}</style><h1>整体表达适配 A/B</h1><p>每列512px正常观看；全屏可放大。原图与状态在父目录。固定前景层级为两版本共有的可见性诊断限制，不表示真实遮挡已解决。</p><a href="all.mp4">连续观看全部（源 / A / B）</a><nav>'''+nav+'</nav>'+''.join(sections)+'''<button onclick="save()">导出反馈</button><script>
function mark(b){let s=b.closest('section');s.querySelector('textarea').value+='\\n@'+s.querySelector('video').currentTime.toFixed(3)+'s ';}
document.querySelectorAll('textarea').forEach(t=>{t.value=localStorage.getItem('relational-'+t.dataset.id)||'';t.oninput=()=>localStorage.setItem('relational-'+t.dataset.id,t.value)});
function save(){let a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify([...document.querySelectorAll('textarea')].map(t=>({case:t.dataset.id,note:t.value})),null,2)],{type:'application/json'}));a.download='feedback.json';a.click()}
</script>'''
    (gallery/'index.html').write_text(page,encoding='utf-8')
    import re
    refs=re.findall(r'(?:src|href)="([^"#]+)"',page)
    missing=[ref for ref in refs if not (gallery/ref).is_file()]
    if missing: raise FileNotFoundError(missing)
    (root/'media-check.json').write_text(json.dumps({'relativeReferences':len(refs),'missing':missing,
        'scope':'filesystem reference check; browser playback separately checked'},indent=2))
    (root/'index.json').write_text(json.dumps({'protocol':protocol,'results':index,
        'versions':{'implementationAndModel':'freeze.json','SDK':'sdk-freeze.json'},
        'inputRelations':'source-relations.json','resources':'resources.json',
        'failures':{'startup':'SciPy import absent before run creation; replaced by NumPy solver; no installation',
            'visual':'all cases retained; no automatic semantic acceptance',
            'lifecycle':'normal per-frame patch restore measured; interruption/failure full-model reset unverified'},
        'intervention':{'perCaseManualEdits':0,'algorithmRevisions':0,'inputPreparation':'36 input copies with hashes; prior history audited',
            'globalChanges':['shared distal visibility for A/B','new relational objective for B'],
            'engineeringMinutes':'not independently metered; timestamps include tool and reasoning waits'},
        'limitations':['no lifecycle interruption/failure reset certification','no automated pixel visibility certification','no palm/head/finger source orientation','GPU readings whole device, not attributed'],
        'finished':datetime.datetime.now().astimezone().isoformat()},indent=2),encoding='utf-8')
    sheet=Image.new('RGB',(1536,len(results)*340));draw=ImageDraw.Draw(sheet)
    for row,c in enumerate(results):
        i=min(30,results[c]['A']['frames']-1)
        for col,folder in enumerate((gallery/(c+'-source'),root/f'{c}-A',root/f'{c}-B')):
            im=Image.open(folder/f'{i:04d}.png').resize((320,320));sheet.paste(im,(col*512+96,row*340+20))
        draw.text((8,row*340+3),c+' @1s',fill='white')
    sheet.save(gallery/'all-samples.png')
    print(gallery/'index.html')

if __name__=='__main__':build(Path(sys.argv[1]))
