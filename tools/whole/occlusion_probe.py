"""Same-time drawable isolation, preserving ordinary frames and failed baselines."""
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.expression import unity
from PIL import Image, ImageDraw

def run(root):
    root.mkdir(exist_ok=False,parents=True)
    old=Path('E:/FuXuanLab/runs/20261108-layer-projection-01')
    for case in ('novel-01','novel-12'):
        curve=json.loads((old/f'{case}-front-curves.json').read_text())
        curve['duration']=1
        c=unity.write_spec(root/f'{case}-curves.json',curve)
        p=unity.write_spec(root/f'{case}-patch.json',json.loads((old/f'{case}-front-patch.json').read_text()))
        if '--rooted-upper' in sys.argv:
            doc=json.loads(p.read_text());doc['rootedUpper']=True;unity.write_spec(p,doc)
        unity.run('LabMotion.RunMotion',root/case,{'ACTION_LAB_MODEL':unity.model3_path(),
            'ACTION_LAB_CURVES':c,'ACTION_LAB_PATCH':p,'ACTION_LAB_LAYER_FRAME':30},root/f'{case}.log')
        folder=root/case/'layers'
        files=list(folder.glob('*.png'))
        sheet=Image.new('RGB',(5*384,((len(files)+4)//5)*340),(32,35,42))
        draw=ImageDraw.Draw(sheet)
        for i,path in enumerate(files):
            im=Image.open(path).crop((330,220,690,570)).resize((320,310))
            x=(i%5)*384;y=(i//5)*340
            sheet.paste(im,(x,y+25));draw.text((x+8,y+5),path.stem,fill='white')
        sheet.save(root/f'{case}-isolation.png')

if __name__=='__main__':run(Path(sys.argv[1]))
