"""Upper-layer motion review: fixed skeleton camera, no Live2D dependencies."""
import sys,json,hashlib,subprocess,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image,ImageDraw
import imageio_ffmpeg
from tools.expression import source

BONES=[('pelvis','spine'),('spine','chest'),('chest','neck'),('neck','head'),
       ('neck','rightShoulder'),('neck','leftShoulder'),
       ('rightShoulder','rightElbow'),('rightElbow','rightWrist'),
       ('leftShoulder','leftElbow'),('leftElbow','leftWrist'),
       ('pelvis','rightHip'),('rightHip','rightKnee'),('rightKnee','rightAnkle'),
       ('pelvis','leftHip'),('leftHip','leftKnee'),('leftKnee','leftAnkle')]


def normalise(frames):
    """One clip-level transform, never independently recenter/resize frames."""
    origin=np.array(frames[0]['pelvis'],float)
    scale=np.linalg.norm(np.array(frames[0]['head'])-origin)
    if not np.isfinite(scale) or scale<1e-6:raise ValueError('Invalid body scale')
    result=[{k:((np.array(v)-origin)/scale).tolist() for k,v in f.items()} for f in frames]
    if not all(np.isfinite(v).all() for f in result for v in f.values()):raise ValueError('Nonfinite nodes')
    return result,{'origin':origin.tolist(),'scale':float(scale),'units':'initial head-pelvis distance'}


def render(nodes,title,time):
    image=Image.new('RGB',(1024,640),'#202733');draw=ImageDraw.Draw(image)
    draw.text((12,8),title,fill='white')
    draw.text((12,28),f'action time {time:.2f}s | skeleton only | fixed front and side cameras',fill='white')
    for view,axis in enumerate((0,2)):
        def place(p):return (view*512+256+(-p[0] if axis==0 else p[2])*150,400-p[1]*150)
        draw.text((view*512+12,54),'FRONT (-X,Y)' if axis==0 else 'SIDE (Z,Y)',fill='white')
        for a,b in BONES:
            color='#ff9999' if a.startswith('right') else '#77bbff' if a.startswith('left') else '#ccd5df'
            draw.line((place(nodes[a]),place(nodes[b])),fill=color,width=4)
        for name in ('rightWrist','leftWrist','head'):
            x,y=place(nodes[name]);draw.ellipse((x-5,y-5,x+5,y+5),fill='#ff9999' if name.startswith('right') else '#77bbff' if name.startswith('left') else '#ffdf88')
    return image


def run(input_path,output):
    output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(input_path,output/'source.json')
    trajectory=source.load(input_path);duration=min(trajectory['keys'][-1][0],6)
    times=np.arange(int(duration*30)+1)/30
    frames,transform=normalise([source.at_time(trajectory,float(t)) for t in times])
    packet={'schema':'upper-node-sequence/v1','fps':30,'times':times.tolist(),'frames':frames,
        'coordinateSystem':'right-handed source XYZ, Y up; anatomical side IDs unchanged; no guessed face orientation',
        'transform':transform,'provenance':{'path':str(input_path),'sha256':hashlib.sha256(input_path.read_bytes()).hexdigest()},
        'status':'existing generated reference, not new policy output or approved natural-motion truth'}
    (output/'nodes.json').write_text(json.dumps(packet))
    (output/'review-prompt.md').write_text('''# 骨架动作评审（上层独立）

先不看任务描述，描述实际看见的动作：两侧身份、肘腕路线、躯干参与、启动/停顿/返回和明显跳变。再结合任务描述判断完成、部分完成、失败或无法判断，注明时间段。

分别评价整体姿态可读性、动作连续性、节奏、双臂/躯干协调。不要用“平滑”代替自然，不把静止当优秀，不把源骨架当正确答案。正侧视图身份相同，不因投影跨中线换侧。

这是无服饰骨架，不评价Live2D遮挡、服饰、网格质量。无掌面/指形/面部朝向证据的内容回答无法判断。意见为待复核观察，不自动变成奖励或训练成功标签。
''',encoding='utf-8')
    video=output/'skeleton-review.mp4'
    cmd=[imageio_ffmpeg.get_ffmpeg_exe(),'-y','-f','rawvideo','-pix_fmt','rgb24','-s','1024x640','-r','30','-i','-',
        '-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
    with subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=subprocess.PIPE) as proc:
        for t,nodes in zip(times,frames):proc.stdin.write(render(nodes,'EXISTING REFERENCE | NOT NEW TRAINING RESULT',t).tobytes())
        proc.stdin.close();err=proc.stderr.read();code=proc.wait()
        if code:raise RuntimeError(err.decode(errors='replace'))
    render(frames[len(frames)//2],'SKELETON REVIEW SAMPLE',times[len(times)//2]).save(output/'sample.png')
    print(video)

if __name__=='__main__':run(Path(sys.argv[1]),Path(sys.argv[2]))
