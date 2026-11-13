"""Bounded authored-hand pose check BEFORE whole-arm gesture authoring.

Independent poses, not a motion sequence or a new inverse model. No topology,
texture, reflection, palm scaling or finger geometry is introduced here.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import sys,json,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image,ImageDraw
from tools.control.pilot import Plant
from tools.control.validate_stateful_arm import APPROVED,pose
from tools.control.finger_control import FINGER_IDS
from tools.whole.batch import markers
from tools.expression import unity

ALT_IDS=('Param111','Param112','Param113','Param114','Param115')


def candidates():
    basic=[('basic-neutral',[0]*5,{}),('basic-v-a',[-25,30,0,-30,-30],{}),
        ('basic-v-b',[-25,15,30,-30,-30],{}),('basic-closed',[-30]*5,{}),
        ('basic-closed-thumb-out',[20,-30,-30,-30,-30],{}),
        ('basic-open-perspective-low',[10,20,15,10,5],{'Param100':-30}),
        ('basic-open-perspective-high',[10,20,15,10,5],{'Param100':30})]
    alt=[('alternate-neutral',[0]*5,{}),('alternate-open',[20]*5,{}),
        ('alternate-closed',[-30]*5,{}),('alternate-front-open',[20]*5,{'Param120':1}),
        ('alternate-perspective-low',[20]*5,{'Param120':1,'Param116':-30}),
        ('alternate-perspective-high',[20]*5,{'Param120':1,'Param116':30})]
    result=[]
    for name,q,extra in basic:
        result.append(dict(name=name,mode='basic',values=dict(zip(FINGER_IDS,q))|extra))
    for name,q,extra in alt:
        result.append(dict(name=name,mode='alternate',values=dict(zip(ALT_IDS,q))|dict(Param92=1,Param93=1,Param118=20,Param119=1)|extra))
    return result


def execute(root):
    root.mkdir(parents=True,exist_ok=False);plant=Plant();m=plant.model;m.set({});cases=candidates()
    patch=unity.write_spec(root/'patch.json',dict(nativeMarkers=markers(m,plant.anchors),
        correctedPatchNodes=np.array(json.loads((APPROVED/'protocol.json').read_text(encoding='utf-8'))['newNodes']).ravel().tolist(),
        nativeDistalFront=True,poses=[pose([0,0,0]) for _ in cases]))
    poses=unity.write_spec(root/'poses.json',dict(physics=False,poses=[dict(label=x['name'],ids=list(x['values']),values=list(x['values'].values())) for x in cases]))
    unity.write_spec(root/'protocol.json',dict(cases=cases,scope='independent static authored-hand poses, not continuous execution or semantic acceptance',
        noNewHandDeformation=True,baselineFingerRange=[-30,30],alternate='92=1/93=1 changes Core opacity; rendered visibility not validated; independent branch not current base-hand controller',
        note='Old nativeChain is only camera/material reference when alternate hand is visible; not alternate end-effector feedback',visualAcceptance=False))
    frozen=root/'implementation';frozen.mkdir();shutil.copy2(__file__,frozen/Path(__file__).name)
    for name in unity.SCRIPTS:shutil.copy2(unity.ROOT/'preview/unity'/name,frozen/name)
    unity.run('LabMotion.RunPoseLab',root/'sdk',dict(ACTION_LAB_MODEL=unity.model3_path(),ACTION_LAB_POSES=poses,ACTION_LAB_PATCH=patch,
        ACTION_LAB_GEOMETRY_FEEDBACK='off',ACTION_LAB_NATIVE_FINGERS='off',ACTION_LAB_GAZE='off',ACTION_LAB_PALM_PLANE='off',ACTION_LAB_PATCH_PROJECTION='off',
        ACTION_LAB_PHYSICS_OVERRIDE='',ACTION_LAB_LAYER_FRAME=''),root/'capture.log')
    frame=json.loads((root/'sdk/0000.json').read_text(encoding='utf-8-sig'))
    a=np.array(frame['wholeArms']['nativeChain']).reshape(4,2);screen=np.array(frame['nativeNodeScreen']).reshape(4,2)
    affine=np.linalg.lstsq(np.c_[a,np.ones(4)],screen,rcond=None)[0]
    centers=[];spans=[]
    for case in cases:
        m.set(case['values']);ids=['ArtMesh'+str(i) for i in range(230,237)] if case['mode']=='basic' else ['ArtMesh'+str(i) for i in range(27,37)]+['ArtMesh119','ArtMesh120','ArtMesh121','ArtMesh122']
        v=np.concatenate([m.mesh[m.names.index(n)] for n in ids if m.opacity[m.names.index(n)]>.01]);p=np.c_[v,np.ones(len(v))]@affine
        lo=p.min(0);hi=p.max(0);centers.append((lo+hi)/2);spans.append(max(hi-lo)+50)
    span=max(180,int(np.ceil(max(spans))))
    sheet=Image.new('RGB',(1280,int(np.ceil(len(cases)/4))*350),'#eeeeee');d=ImageDraw.Draw(sheet)
    for i,(case,center) in enumerate(zip(cases,centers)):
        crop=tuple(map(int,[center[0]-span/2,center[1]-span/2,center[0]+span/2,center[1]+span/2]))
        with Image.open(root/'sdk'/f'{i:04}.png') as image:
            tile=image.convert('RGB').crop(crop).resize((320,320));tile.save(root/(case['name']+'.png'));sheet.paste(tile,(i%4*320,i//4*350+30))
        d.text((i%4*320+7,i//4*350+7),case['name'],fill='black')
    sheet.save(root/'static-hand-contact.png')
    unity.write_spec(root/'review.json',dict(fixedPixelSpan=span,cases=len(cases),images='final SDK PNG; translated crop, common scale',
        judgement='visual shape/orientation first; numerical static arrival is not a hand-task pass'))
    print(root/'static-hand-contact.png')


if __name__=='__main__':execute(Path(sys.argv[1]))
