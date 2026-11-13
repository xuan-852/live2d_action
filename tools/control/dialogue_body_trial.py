"""Multi-turn offline dialogue fixture consumed by a feedback-driven planner."""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import sys,json,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.control.dialogue_body import DialogueBody
from tools.control.intent_expression_trial import plan
from tools.control.native_target import NATIVE_ROUTE_LIMITS
from tools.control.validate_stateful_arm import run
from tools.expression.unity import write_spec


def event(identifier,kind,text,target='user',affect='neutral',seconds=2.):
    return dict(id=identifier,kind=kind,text=text,target=target,affect=affect,utteranceSeconds=seconds)


def fixture():
    # Times belong to incoming dialogue, not authored arm keyframes.
    return [
        (1,event('listen-1','listen','用户：你说的是哪一份资料？',seconds=0)),
        (61,event('answer-1','reference','符玄：是右边这份，先看这里。','A','warm',2.7)),
        (166,event('listen-2','listen','用户：明白了，接下来呢？',seconds=0)),
        (211,event('explain','say','符玄：先核对内容，再决定下一步。',affect='warm',seconds=2.2)),
        (301,event('think','think','符玄：让我确认一下……',affect='uncertain',seconds=1.7)),
        (376,event('point-again','reference','符玄：可以对照右边这一项。','A',seconds=2.4)),
        (391,event('interrupt','cancel','用户打断：先别指了，我换个问题。',seconds=0)),
        (436,event('unavailable','reference','用户：那左边那份呢？／符玄：左侧资料需要另外核对。','B',seconds=2.1)),
        (526,event('decline','decline','符玄：目前还不能据此确定。',affect='firm',seconds=1.8)),
        (601,event('close','say','符玄：我们先保留结论，等信息完整再继续。',affect='warm',seconds=2.2)),
        (706,event('exit','cancel','结束回应，回到等待。',seconds=0)),
    ]


class Provider:
    implementation_files=['tools/control/dialogue_body.py','tools/control/dialogue_body_trial.py',
                          'tools/control/face_control.py','tools/control/body_session.py',
                          'tools/control/intent_expression_trial.py','tools/control/gesture_exploration.py']
    def __init__(self,scene,events):
        self.body=DialogueBody(scene);self.events=dict(events);self.packets=[]
    def initialize(self,feedback):self.body.initialize(feedback)
    def next(self,now):
        i=round(now*30)
        if i in self.events:self.body.submit(self.events[i],now)
        packet=self.body.next(now);self.packets.append(packet)
        return packet
    def observe(self,row,context):return self.body.observe(row,context)


def execute(root,event_source=None):
    root.mkdir(parents=True,exist_ok=False)
    old,_,markers,fixed=plan()
    approved=next(x for x in old['frames'] if x['stage']=='point/express')
    target=dict(approved['targets']['native']);target={k:v for k,v in target.items() if k in ('wrist','elbow','palm_heading')}
    scene=dict(A=dict(label='资料 A',xy=fixed['pointTarget'],gaze=[.7,-.25],
                      pointing=dict(nodes=target,headBody=approved['headBody']),
                      scope='registered pointing anchor, not arbitrary scene IK'),
               B=dict(label='资料 B',xy=[-.38,.43],gaze=[-.65,.1],pointing=None,
                      scope='outside current tested pointing registration'))
    events=fixture()
    if event_source is not None:
        supplied=json.loads(event_source.read_text(encoding='utf-8-sig'))
        events=supplied['events']
    # External model/author plans can use the same semantic interface. A supplied
    # plan cannot replace the rig's registered scene affordances or motor values.
    from tools.control.dialogue_body import validate_plan
    times=[x[0] for x in events]
    if not times or times[0]!=1 or any(type(x) is not int or x<1 or x>780 for x in times) or times!=sorted(set(times)):
        raise ValueError('event frames must be unique, ordered, within 1..780 and start at1')
    for _,p in events:validate_plan(p,scene)
    provider=Provider(scene,events)
    write_spec(root/'dialogue-events.json',dict(events=events,scene=scene,origin='authored offline dialogue; no production model calls'))
    write_spec(root/'acceptance-protocol.json',dict(fps=30,frames=781,physics=False,
        duration=26,onePersistentSDK=True,
        lifecycle=['11 semantic events applied on their event frame','no old point after frame391 cancellation',
                   'B must fall back without inventing arm reach','final15 frames settle'],
        safety=dict(armNodeLimits='existing frozen controller limits',pairing=1e-6,flips=0,minimumArea=.5),
        scope='offline semantic planner + actual feedback; not learned language generation or audio lip sync',
        visualAcceptance=False,pointing='A registered from user-approved first gesture; B unavailable'))
    write_spec(root/'task-markers.json',markers)
    run(root/'sdk',request_stream=dict(frames=[dict(stage='provider') for _ in range(780)],metadata=dict(driver='dialogue feedback provider')),
        request_provider=provider,native_five=True,native_limits=NATIVE_ROUTE_LIMITS.copy(),
        native_fingers=True,native_gaze=True,face_control=True,task_markers=markers)
    write_spec(root/'requested-packets.json',provider.packets)
    write_spec(root/'planner-history.json',dict(events=provider.body.events,bodyRequests=provider.body.session.history))


if __name__=='__main__':execute(Path(sys.argv[1]),Path(sys.argv[2]) if len(sys.argv)>2 else None)
