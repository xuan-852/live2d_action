"""Larger-motion task/state/feedback functional test. No online AI services."""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import sys,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.control.task_scene import build_scene
from tools.control.task_expression import TaskExpressionDirector
from tools.control.native_target import NATIVE_ROUTE_LIMITS
from tools.control.validate_stateful_arm import run
from tools.expression.unity import write_spec


def task(identifier,purpose,targets,request,confidence=.85):
    return dict(id=identifier,purpose=purpose,targets=targets,request=request,confidence=confidence)


def interventions():
    return [
        (1,task('compare-main','compare',['A','C','E'],'依次展示右侧、上方，再展开双臂作对照。')),
        (361,task('redirect-high','inspect',['C'],'重新看上方的位置。')),
        (385,task('redirect-right','inspect',['A'],'先等等，改看右边。')),
        (409,task('interrupt','cancel',[],'停止当前展示，先收回。')),
        (481,task('outside','compare',['U','C'],'先尝试远处；如果做不到，改为展示上方位置。',.6)),
        (751,task('compare-again','compare',['E','A'],'展开双臂，再回到右侧位置。')),
        (991,task('wait','wait',[],'结束展示，等待下一次任务。')),
    ]


class Provider:
    implementation_files=['tools/control/task_scene.py','tools/control/task_expression.py',__file__,
        'tools/control/dialogue_body.py','tools/control/body_session.py','tools/control/face_control.py',
        'tools/control/intent_expression_trial.py','tools/control/gesture_exploration.py']
    def __init__(self,scene):
        self.director=TaskExpressionDirector(scene);self.inputs=dict(interventions());self.packets=[]
    def initialize(self,feedback):self.director.initialize(feedback)
    def next(self,now):
        i=round(now*30)
        if i in self.inputs:self.director.submit(self.inputs[i],now)
        packet=self.director.next(now);self.packets.append(packet);return packet
    def observe(self,row,context):return self.director.observe(row,context)


def execute(root,source):
    root.mkdir(parents=True,exist_ok=False);scene,markers,preflight=build_scene(source);p=Provider(scene)
    shutil.copy2(source,root/'source-map.json')
    write_spec(root/'task-inputs.json',dict(scene=scene,interventions=interventions(),source=str(source),
        origin='task requests + confidence; feedback-driven rule baseline, no learned policy'))
    write_spec(root/'core-preflight.json',preflight)
    write_spec(root/'acceptance-protocol.json',dict(frames=1081,fps=30,duration=36,physics=True,cameraMargin=1.5,
        completion='15 consecutive actual settled frames; native pointing axis <=6deg; task events depend on feedback, not timeline',
        taskDeadline='registered reachSeconds + .5s lower fallback / .6s task-constraint fallback; no target shrink or hidden retiming',
        nodePosition=.005,palmDegrees=4,pairing=1e-6,minPatchArea=.5,flips=0,
        interventions='7 external task changes at fixed action frames, not a timed motor script; recovery C chosen after actual U failure',
        required=['larger native overhead reach','patch bilateral expansion','moving goal change + cancellation',
                  'U limited with actual feedback','C recovery','final15 frames all channels settled'],
        visualAcceptance=False,modelGenerated=False))
    write_spec(root/'task-markers.json',markers)
    run(root/'sdk',physics=True,request_stream=dict(frames=[dict(stage='task-provider') for _ in range(1080)],metadata=dict(driver='task/state/actual-feedback director')),
        request_provider=p,native_five=True,native_limits=NATIVE_ROUTE_LIMITS.copy(),
        native_fingers=True,native_gaze=True,face_control=True,task_markers=markers,camera_margin=1.5)
    write_spec(root/'requested-packets.json',p.packets)
    write_spec(root/'director-history.json',dict(events=p.director.events,outcomes=p.director.outcomes,
        bodyEvents=p.director.body.events,bodyRequests=p.director.body.session.history))


if __name__=='__main__':execute(Path(sys.argv[1]),Path(sys.argv[2]))
