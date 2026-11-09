"""Full-motion candidate for the isolated sleeve/upper-skin/cap relation."""
import sys
import json
import subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.expression import unity
import imageio_ffmpeg

def run(root):
    root.mkdir(parents=True,exist_ok=False)
    old=Path('E:/FuXuanLab/runs/20261108-layer-projection-01')
    for case in ('novel-01','novel-12'):
        c=unity.write_spec(root/f'{case}-curves.json',json.loads((old/f'{case}-front-curves.json').read_text()))
        patch=json.loads((old/f'{case}-front-patch.json').read_text())
        patch['layerPolicy']='upper-between-cap-sleeve'
        p=unity.write_spec(root/f'{case}-patch.json',patch)
        folder=root/case
        unity.run('LabMotion.RunMotion',folder,{'ACTION_LAB_MODEL':unity.model3_path(),
            'ACTION_LAB_CURVES':c,'ACTION_LAB_PATCH':p,'ACTION_LAB_LAYER_FRAME':''},root/f'{case}.log')
        cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-filter_complex_threads','1',
            '-framerate','30','-i',str(old/f'{case}-legacy'/'%04d.png'),
            '-framerate','30','-i',str(folder/'%04d.png'),'-filter_complex',
            '[0:v]crop=400:420:320:200[a];[1:v]crop=400:420:320:200[b];[a][b]hstack[v]',
            '-map','[v]','-c:v','libx264','-threads','1','-crf','18','-pix_fmt','yuv420p',
            '-movflags','+faststart',str(root/f'{case}-compare.mp4')]
        subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)

if __name__=='__main__':run(Path(sys.argv[1]))
