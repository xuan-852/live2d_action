"""Same-trajectory layer/projection ablation. Extreme sorting is diagnostic only."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import json
from pathlib import Path
import subprocess
import sys
import datetime
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import imageio_ffmpeg
from tools.expression import unity


def run(root):
    previous=Path('E:/FuXuanLab/runs/20261108-whole-capability-01')
    variants=[('legacy','legacy-z',1,True),('front','diagnostic-order',1,True),
              ('behind','diagnostic-order',0,True),('unshortened','diagnostic-order',1,False)]
    protocol={'cases':['novel-01','novel-12'],'variants':variants,
              'unknown':'source garment ordering and palm plane; no source-derived depth inference',
              'frozenAt':datetime.datetime.now().astimezone().isoformat()}
    unity.write_spec(root/'protocol.json',protocol)
    results={}
    for name in protocol['cases']:
        for label,policy,front,projection in variants:
            curve=json.loads((previous/(name+'-restored-curves.json')).read_text())
            patch=json.loads((previous/(name+'-restored-patch.json')).read_text())
            patch['layerPolicy']=policy
            for p in patch['poses']:
                p['forearmFront']=front;p['handFront']=front
                if not projection:
                    p['projection'].update(upperLength=1,forearmLength=1)
            c=unity.write_spec(root/(name+'-'+label+'-curves.json'),curve)
            p=unity.write_spec(root/(name+'-'+label+'-patch.json'),patch)
            folder=root/(name+'-'+label)
            unity.run('LabMotion.RunMotion',folder,{'ACTION_LAB_MODEL':unity.model3_path(),
                      'ACTION_LAB_CURVES':c,'ACTION_LAB_PATCH':p},root/(name+'-'+label+'.log'))
            frames=[json.loads(f.read_text(encoding='utf-8-sig')) for f in sorted(folder.glob('[0-9][0-9][0-9][0-9].json'))]
            results[name+'-'+label]={'frames':len(frames),'sorting':frames[0]['wholeArms']['sdkSortingMode'],
                'maxRestoreError':max(f['wholeArms']['restoredMeshError'] for f in frames),
                'minAreaRatio':min(f['wholeArms']['minTriangleAreaRatio'] for f in frames)}
            unity.write_spec(root/'results.json',results)
        cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-filter_complex_threads','1']
        for label,*_ in variants:cmd+=['-framerate','30','-i',str(root/(name+'-'+label)/'%04d.png')]
        cmd+=['-filter_complex','[0:v]scale=512:512[a];[1:v]scale=512:512[b];[2:v]scale=512:512[c];[3:v]scale=512:512[d];[a][b][c][d]hstack=inputs=4[v]',
              '-map','[v]','-c:v','libx264','-threads','1','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(root/(name+'-layers.mp4'))]
        subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        print(name,'complete',flush=True)
    unity.write_spec(root/'finished.json',{'time':datetime.datetime.now().astimezone().isoformat()})


if __name__=='__main__':run(Path(sys.argv[1]))
