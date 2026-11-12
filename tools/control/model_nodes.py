"""Model-specific neutral node proposal. No action generation or control execution."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image,ImageDraw
from tools.control.pilot import Plant


def run(root,neutral):
    root.mkdir(parents=True,exist_ok=False);plant=Plant();model=plant.model
    frame=json.loads(neutral.read_text(encoding='utf-8-sig'))
    nodes={};edges=[]
    for side,key in [('native','nativeChain'),('patch','patchChain')]:
        chain=np.array(frame['wholeArms'][key]).reshape(-1,2)
        for index,name in enumerate(('shoulder','elbow','wrist','palm')):
            ident=f'{side}_{name}'
            nodes[ident]={'xy':chain[index].tolist(),'kind':'material-landmark' if name!='palm' else 'material-palm-reference',
                'source':f'neutral wholeArms.{key}[{index}]',
                'status':'existing material readback; not anatomical joint or freely movable point'}
            if index:edges.append([f'{side}_{("shoulder","elbow","wrist")[index-1]}',ident])
    def box(mesh):
        points=model.mesh[model.names.index(mesh)].astype(float)
        return points.min(0),points.max(0)
    low,high=box('ArtMesh128');head=(low+high)/2
    # These are explicitly proposed local reference frames, never inferred bones.
    left=np.array(nodes['patch_shoulder']['xy']);right=np.array(nodes['native_shoulder']['xy'])
    chest=(left+right)/2
    tl,th=box('ArtMesh177');body_root=np.array([(tl[0]+th[0])/2,tl[1]])
    neck=np.array([head[0],low[1]])
    for name,xy,src in [('head',head,'ArtMesh128 neutral bounding-box centre'),
                        ('neck',neck,'ArtMesh128 neutral lower boundary midpoint'),
                        ('chest',chest,'mean of existing left/right shoulder material references'),
                        ('body_root',body_root,'ArtMesh177 neutral lower-bound midpoint')]:
        nodes[name]={'xy':xy.tolist(),'kind':'virtual-reference','source':src,
            'status':'proposed model-space reference, NOT recovered anatomical pivot'}
    edges += [['body_root','chest'],['chest','neck'],['neck','head'],['chest','native_shoulder'],['chest','patch_shoulder']]
    doc={'schema':'fuxuan-node-proposal/v0','mocSHA':model.sha,'neutralSHA':hashlib.sha256(neutral.read_bytes()).hexdigest(),
        'coordinateSystem':'current model-local XY, Y up; no inferred Z',
        'sides':{'native':'anatomical left, screen right at neutral','patch':'anatomical right, screen left at neutral'},
        'nodes':nodes,'edges':edges,
        'orientationFields':{'head':['yaw','pitch','roll'],'torso':['yaw','pitch','roll']},
        'orientationStatus':'required separate pose fields; neutral display only; actual axis semantics and readback not certified; do not infer yaw/pitch from 2D lines',
        'scope':'model-specific neutral proposal only, NOT final minimal control interface or new skeleton certification',
        'excluded':'no invented hip/knee/ankle/finger joints; no new control, no training',
        'nativeAnchorProvenance':plant.anchors.evidence,
        'patchAnchorProvenance':'existing WholeArmSession ShoulderConnection barycentric material landmarks; exact neutral capture path '+str(neutral)}
    (root/'nodes.json').write_text(json.dumps(doc,indent=2))
    # Fixed-camera projection inferred solely from paired same-frame readback.
    p=np.array(frame['wholeArms']['nativeChain'][2:6]).reshape(2,2)
    s=np.array(frame['actualScreen']).reshape(2,2)
    sx=(s[1,0]-s[0,0])/(p[1,0]-p[0,0]);sy=(s[1,1]-s[0,1])/(p[1,1]-p[0,1])
    offset=s[0]-p[0]*[sx,sy]
    out=Image.new('RGB',(1600,1080),'#202733')
    with Image.open(neutral.with_suffix('.png')) as image:out.paste(image.convert('RGB'),(0,56))
    draw=ImageDraw.Draw(out)
    draw.text((18,14),'FUXUAN MODEL NODE PROPOSAL v0 | neutral pose | not an action or training result',fill='white')
    def place(name):return tuple((np.array(nodes[name]['xy'])*[sx,sy]+offset+[0,56]).tolist())
    for a,b in edges:draw.line((place(a),place(b)),fill='#aaffaa',width=3)
    for index,(name,node) in enumerate(nodes.items(),1):
        x,y=place(name);color='#ffcc55' if node['kind']=='virtual-reference' else '#44ddff'
        draw.ellipse((x-5,y-5,x+5,y+5),fill=color);draw.text((x+7,y-9),str(index),fill='white')
        draw.text((1040,95+index*30),f'{index:02} {name} [{node["kind"]}]',fill=color)
    for name in ('head','chest'):
        x,y=place(name);draw.rectangle((x-25,y-20,x+25,y+20),outline='#ffcc55',width=2)
    draw.text((1040,560),'Head / torso boxes: orientation carriers.',fill='white')
    draw.text((1040,585),'Yaw/pitch/roll are separate, not line angles.',fill='white')
    draw.text((1040,630),'Blue: existing material references.',fill='#44ddff')
    draw.text((1040,655),'Gold: proposed virtual reference points.',fill='#ffcc55')
    draw.text((1040,700),'No inferred 3D skeleton / leg joints.',fill='white')
    draw.text((1040,725),'No new motions or parameter controls.',fill='white')
    out.save(root/'model-node-proposal.png');print(root/'model-node-proposal.png')

if __name__=='__main__':run(Path(sys.argv[1]),Path(sys.argv[2]))
