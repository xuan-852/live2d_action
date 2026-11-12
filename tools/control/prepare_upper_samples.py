"""Task-paired upper-motion candidates; never silently approve generated sources."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,hashlib,shutil,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image,ImageDraw
import imageio_ffmpeg
from tools.expression import source
from tools.control.sequence_training import model_nodes
from tools.control.skeleton_review import normalise,render

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def pair_request(path,requests):
    doc=json.loads(path.read_text(encoding='utf-8-sig'))
    matches=[r for r in requests if r['id']==doc.get('caption') and r.get('seed')==doc.get('seed')]
    unique={(r['text'],r['seed']) for r in matches}
    if len(unique)>1:raise ValueError('Conflicting original task descriptions')
    return matches[0] if matches else None

def run(root,inputs,interface):
    root.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,root/'frozen-prepare-upper-samples.py')
    requests=[];provenance=[]
    for name in ('requests.json','requests-reserve.json'):
        path=inputs/name
        if path.exists():
            requests+=json.loads(path.read_text(encoding='utf-8-sig'))
            shutil.copy2(path,root/name);provenance.append({'path':str(path),'sha256':digest(path)})
    home=np.array(json.loads((interface/'node-targets.json').read_text())['nodes'][0]);np.save(root/'model-home.npy',home)
    records=[];selected={'novel-01','novel-03','novel-08'}
    for path in sorted((inputs/'references').glob('*.json')):
        request=pair_request(path,requests);tr=source.load(path);duration=tr['keys'][-1][0]
        times=np.arange(int(np.floor(duration*30))+1)/30
        raw=[source.at_time(tr,float(t)) for t in times];normal,transform=normalise(raw)
        headings=[]
        for f in raw:
            row=[]
            for side in ('left','right'):
                p=np.array([f[side+n] for n in ('Shoulder','Elbow','Wrist')]);v=np.diff(p,axis=0)
                row.extend(np.arctan2(v[:,1],-v[:,0]))
            headings.append(row)
        angles=np.unwrap(headings,axis=0);angles-=angles[0]
        mapped=np.array([model_nodes(home,a) for a in angles])
        record={'id':path.stem,'sourcePath':str(path),'sourceSHA':digest(path),'originalRequest':request,
            'duration':duration,'sampledEndTime':float(times[-1]),'frames':len(times),'fps':30,
            'status':'candidate-unreviewed','trainingEligible':False,'split':'unassigned-before-quality-review',
            'knownExposure':'existing exposed references, not unseen evaluation',
            'missing':['head/torso orientation','palm orientation','finger pose','specific expression intent'],
            'retargetLimitations':['fixed shoulders and roots','front-projection directions only','depth discarded',
                'neutral relative palm','model reachability not tested','cannot treat mapping as equivalent full-body motion']}
        folder=root/path.stem;folder.mkdir();shutil.copy2(path,folder/'original-reference.json')
        (folder/'sample.json').write_text(json.dumps(record,indent=2))
        (folder/'source-nodes.json').write_text(json.dumps({'times':times.tolist(),'frames':normal,'transform':transform,
            'scope':'source 3D positional reference, not model control interface'}))
        (folder/'model-arm-nodes.json').write_text(json.dumps({'times':times.tolist(),'nodes':mapped.tolist(),
            'sides':['native-anatomical-left','patch-anatomical-right'],'order':['shoulder','elbow','wrist','palm'],
            'scope':'candidate model-scale arm-node mapping, no fabricated head/body orientation'}))
        review={'independentDescription':None,'goalAwareReview':None,'timecodedIssues':[],
            'sourceQuality':'unreviewed','mappingQuality':'unreviewed','decision':'pending',
            'reviewer':None,'taskDescriptionSource':'original request' if request else 'missing'}
        (folder/'review.json').write_text(json.dumps(review,indent=2))
        records.append(record)
        if path.stem not in selected:continue
        video=folder/'review.mp4'
        cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','rawvideo','-pix_fmt','rgb24','-s','1536x640','-r','30','-i','-',
             '-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
        with subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE) as proc:
            for i,t in enumerate(times):
                image=Image.new('RGB',(1536,640),'#202733');image.paste(render(normal[i],path.stem+' | ORIGINAL GENERATED REFERENCE',float(t)),(0,0))
                d=ImageDraw.Draw(image);d.text((1036,8),'MODEL ARM-NODE MAPPING | NOT TRAINED',fill='white')
                d.text((1036,30),'Source intent reviewed separately, no automatic approval.',fill='white')
                origin=home[:,0].mean(0)
                for side,color in enumerate(('#00ffff','#ff66cc')):
                    points=(mapped[i,side]-origin)*[1000,-1000]+[1280,180]
                    d.line([tuple(p) for p in points],fill=color,width=3)
                    for x,y in points:d.ellipse((x-4,y-4,x+4,y+4),outline=color,width=2)
                proc.stdin.write(image.tobytes())
            proc.stdin.close();err=proc.stderr.read();code=proc.wait()
            if code:raise RuntimeError(err.decode(errors='replace'))
    (root/'manifest.json').write_text(json.dumps({'requests':provenance,'modelHomeSource':str(interface),
        'modelHomeSHA':digest(interface/'node-targets.json'),'samples':records},indent=2))
    (root/'REVIEW.md').write_text('''# 上层动作候选评审

先独立观看骨架并记录实际动作与时间段，再读取sample.json中的originalRequest判断任务符合度。
分别评源动作的路线/节奏/连续性/双臂躯干协调，与映射是否丢失这些特征；不要把映射失败归给源动作。
没有方向数据的转头/掌面内容标记无法判断。没有认可前trainingEligible必须为false。
review.json填写观察、问题时间、评审者和accept/reject/uncertain；视觉模型意见只是证据，非自动奖励。
所有源记录均已曝光，不能把当前样本包装成未见测试；质量审核后再按任务/来源分组冻结训练划分。
当前三条视频为01/03/08，按既定编号选取，不按表现挑选。所有样本保留完整时长，没有6秒截断。
''',encoding='utf-8')
    print(json.dumps({'candidates':len(records),'matchedRequests':sum(r['originalRequest'] is not None for r in records),
        'distinctTaskTexts':sorted({r['originalRequest']['text'] for r in records if r['originalRequest']}),
        'trainingEligible':0,'reviewVideos':sorted(selected)},indent=2))

if __name__=='__main__':run(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]))
