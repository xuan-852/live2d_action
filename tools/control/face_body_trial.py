"""Exact prior accepted body-packet replay with optional face realization.

No new model calls: source language/affect/plans remain the original API output.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import sys,json,copy,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.control.face_expression import face_goal
from tools.control.validate_stateful_arm import run
from tools.control.native_target import NATIVE_ROUTE_LIMITS
from tools.expression.unity import write_spec


class Replay:
    implementation_files=[__file__,'tools/control/face_expression.py','tools/control/face_control.py']
    def __init__(self,source):
        self.source=source
        self.packets=json.loads((source/'requested-packets.json').read_text(encoding='utf-8'))
        history=json.loads((source/'model-history.json').read_text(encoding='utf-8'))
        self.accepted={x['plan']['id']:x['acceptedAt'] for x in history['decisions']}
    def initialize(self,feedback):pass
    def next(self,now):
        p=copy.deepcopy(self.packets[round(now*30)-1])
        p['faceGoal']=face_goal(p,now,self.accepted[p['intent']['planId']])
        p['intent']['origin']='replay of recorded DeepSeek plans, new face realization; NOT fresh model decisions'
        return p
    def observe(self,row,context):return dict(replay=True,headBodyActual=list(map(float,context)))


def execute(root,source):
    root.mkdir(parents=True,exist_ok=False)
    markers=json.loads((source/'sdk/patch.json').read_text(encoding='utf-8'))['taskMarkers']
    write_spec(root/'protocol.json',dict(source=str(source),fps=30,frames=721,physics=True,
        bodyPackets='unchanged recorded requests and timing',newModelCalls=0,
        limits='existing arm limits; eyelids speed12 acceleration180; others3/18',
        checks=['paired SDK readback','no face continuity violations','blink closure <.1',
                'final15 face settled','body node difference reported, not assumed zero'],
        semanticAcceptance=False,audioLipSync=False))
    run(root/'sdk',physics=True,request_stream=dict(frames=[dict(stage='face-replay') for _ in range(720)],metadata=dict(driver='recorded body with face')),
        request_provider=Replay(source),native_five=True,native_limits=NATIVE_ROUTE_LIMITS.copy(),
        native_fingers=True,native_gaze=True,face_control=True,face_expression=True,
        task_markers=markers,camera_margin=1.5)


if __name__=='__main__':execute(Path(sys.argv[1]),Path(sys.argv[2]))
