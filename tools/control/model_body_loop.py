"""Independent model service or external mailbox over an isolated SDK clock.

The default service model is the user-selected DeepSeek V4.1 Flash registered
in OpenCode. No canned response list or production pet/AI-memory bridge is used.
"""
import os
for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,copy,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.control.embodied_plan import EmbodiedExecution,validate_model_plan,POSTURE_KEYS,POSTURE_LIMITS
from tools.control.task_scene import build_scene
from tools.control.native_target import NATIVE_ROUTE_LIMITS
from tools.control.validate_stateful_arm import run
from tools.expression.unity import write_spec

INITIAL='请向我介绍两侧的位置：先适当展开双臂，同时转动躯干和头部，让身体也参与表达；随后示意上方C。动作可以大一些，但遇到冲突不要硬撑。说话和身体要表达同一件事。'
INTERRUPT='先停一下，不要继续指上面了。把手收回来，听我说。'
PROBE='现在请只做一次对远处U的受限试探。若实际到不了，就明确告诉我，并改为展示较近的上方C；不要反复硬撑。身体也可以配合，但不要用改小全部动作掩盖失败。'
CLOSE='明白了。现在收回手，头和身体自然回正，结束展示并等待我。'


def export_feedback(feedback):
    snapshot=copy.deepcopy(feedback)
    snapshot['headBody']=list(map(float,snapshot['headBody']))
    return snapshot


def validate_reply(reply,request,scene):
    if not isinstance(reply,dict) or set(reply)!={'requestId','observationTime','plan'}:raise ValueError('reply envelope')
    if type(reply['requestId']) is not int or reply['requestId']!=request['requestId']:raise ValueError('stale request id')
    if reply['observationTime']!=request['observationTime']:raise ValueError('stale actual observation')
    return validate_model_plan(reply['plan'],scene)


class Provider:
    implementation_files=[__file__,'tools/control/embodied_plan.py','tools/control/dialogue_body.py',
        'tools/control/body_session.py','tools/control/task_scene.py','tools/control/task_expression.py',
        'tools/control/face_control.py','tools/control/gesture_exploration.py','tools/control/intent_expression_trial.py',
        'tools/control/opencode_model_client.py','tools/control/body_following_targets.py']

    def __init__(self,root,scene,client=None):
        self.root=root;self.mailbox=root/'model-mailbox';self.mailbox.mkdir()
        self.exec=EmbodiedExecution(scene);self.scene=scene;self.history=[];self.packets=[]
        self.messages=[dict(role='user',text=INITIAL)];self.query_reason='initial-user-request'
        self.interrupted=False;self.probe_added=False;self.probe_failed=False;self.close_added=False
        self.finished=False;self.budget_stop=False;self.last_row=None
        self.client=client
        self.guards=[]
        self.body_following=None

    def initialize(self,feedback):self.exec.initialize(feedback)

    def set_execution_context(self,applied,source_patch):
        if self.body_following is None:
            from tools.control.body_following_targets import BodyFollowingTargets
            from tools.control.validate_stateful_arm import APPROVED
            nodes=json.loads((APPROVED/'protocol.json').read_text(encoding='utf-8'))['newNodes']
            self.body_following=BodyFollowingTargets(nodes)
        self.body_following.set_context(applied)

    def receive_user(self,text,reason='new-user-turn'):
        self.messages.append(dict(role='user',text=text))
        self.finished=False;self.query_reason=reason

    def _query(self,now,reason):
        index=len(self.history);feedback=self.exec.body.session.feedback
        # BodyExecutionSession intentionally owns ndarray context state. Export
        # a JSON snapshot without mutating that live state or dropping feedback.
        feedback=export_feedback(feedback)
        request=dict(requestId=index,observationTime=feedback['time'],nextActionTime=now,reason=reason,
            conversation=copy.deepcopy(self.messages),scene=copy.deepcopy(self.scene),actual=copy.deepcopy(feedback),
            lastExecution=self.exec.last_report,lastPlan=copy.deepcopy(self.exec.plan),
            activeUserRequest=copy.deepcopy(next(m for m in reversed(self.messages) if m['role']=='user')),
            lastFrame=str(self.root/'sdk/capture'/f"{round(feedback['time']*30):04}.png"),
            outputContract=dict(fields=['id','kind','text','target','affect','utteranceSeconds','posture','reachSeconds','holdSeconds','finish','explanation'],
                kinds=['reference','say','listen','think','decline','cancel'],affects=['neutral','warm','uncertain','firm'],
                postureDegrees=dict(zip(POSTURE_KEYS,POSTURE_LIMITS.tolist())),
                note='six authored expression axes, not 3D joint angles; negative/positive within limits; cancel posture all zero',
                reachSeconds=[.3,5.],holdSeconds=[.3,2.],finish='boolean; only cancel/listen may finish'),
            motorSemantics=dict(reference='ONLY reference with a scene target commands arms. E expands BOTH arms; C raises native arm; A points right; U tries the distant goal.',
                say='Speech + supplied head/torso only; arms commanded NEUTRAL. Saying 展开/举手 does not move arms.',
                listen='Listening with supplied head/torso; arms NEUTRAL.',cancel='Withdraw arms and neutralize head/torso.',
                feedback='lastPlan is the exact previously executed command; lastExecution is its actual SDK result, not evidence that the spoken description happened.'),
            instructions='Generate one coherent spoken-text/body plan using this ACTUAL state. No actuator arrays. Do not mark failure as success. Prefer selective revision when body and arm goals conflict. Choose a new unique plan id. Reply envelope requestId, observationTime, plan.',
            timing='action clock PAUSED while waiting; inference latency logged separately; no real-time claim')
        write_spec(self.mailbox/f'{index:03}.request.json',request)
        print('MODEL_REQUEST',index,reason,str(self.mailbox/f'{index:03}.request.json'),flush=True)
        start=time.monotonic();path=self.mailbox/f'{index:03}.reply.json';previous_bad=None
        if self.client is not None:
            repair=None
            for attempt in range(3):
                reply,metadata=self.client.respond(request,repair)
                write_spec(self.mailbox/f'{index:03}.api-{attempt}.json',metadata)
                try:
                    validate_reply(reply,request,self.scene)
                    break
                except (ValueError,TypeError,KeyError) as error:
                    repair=str(error)
                    write_spec(self.mailbox/f'{index:03}.api-{attempt}.rejected.json',dict(error=repair,reply=reply))
            else:raise ValueError('model plan rejected after bounded validation repair')
            write_spec(path,reply)
        while True:
            if time.monotonic()-start>540:raise TimeoutError('upper model response '+str(index))
            if not path.exists():time.sleep(.05);continue
            raw=path.read_text(encoding='utf-8-sig')
            if raw==previous_bad:time.sleep(.05);continue
            try:
                reply=json.loads(raw);plan=validate_reply(reply,request,self.scene)
                self.exec.submit(plan,now)
            except (ValueError,TypeError,KeyError) as error:
                previous_bad=raw;write_spec(self.mailbox/f'{index:03}.rejected.json',dict(error=str(error),raw=raw))
                continue
            self.history.append(dict(requestId=index,observationTime=feedback['time'],acceptedAt=now,reason=reason,
                                     inferenceWallSeconds=time.monotonic()-start,plan=copy.deepcopy(plan),
                                     source=f'independent {self.client.provider}/{self.client.model} service' if self.client is not None else 'current conversation model via tool-authored mailbox response'))
            self.messages.append(dict(role='assistant',text=plan['text'],planId=plan['id']))
            break

    def next(self,now):
        if (self.exec.plan is not None and not self.interrupted and not self.probe_added and
                self.exec.plan['kind']=='reference' and self.exec.plan['target']=='C' and now-self.exec.accepted>=.7):
            self.interrupted=True;self.receive_user(INTERRUPT,'user-interruption-during-motion')
        if self.query_reason and not self.finished:
            reason=self.query_reason;self.query_reason=None
            if len(self.history)>=10:
                self.budget_stop=True;self.finished=True
                safe=dict(id='driver-budget-safe-stop',kind='cancel',text='实验决策预算已用尽，安全收尾；不计为模型完成任务。',
                          target='user',affect='neutral',utteranceSeconds=0.,posture={k:0. for k in POSTURE_KEYS},
                          reachSeconds=3.,holdSeconds=.5,finish=True,explanation='实验驱动的机械安全收尾，不是模型生成的决定。')
                self.exec.submit(safe,now);self.guards.append(dict(time=now,reason='model-decision-budget-exhausted',modelGenerated=False))
            else:self._query(now,reason)
        packet=self.exec.next(now)
        if self.body_following is not None:packet=self.body_following.realize(packet,self.exec.plan,released=self.exec.body.idle_released)
        if self.client is not None:packet['intent']['origin']='independent model service generated plan; actual SDK feedback in next request'
        if self.budget_stop:packet['intent']['origin']='mechanical driver safety stop, NOT model-generated'
        packet['intent']['modelLoop']=dict(requestId=len(self.history)-1,finished=self.finished,
            interrupted=self.interrupted,probeFailed=self.probe_failed,actionClockPausedDuringInference=True)
        self.packets.append(packet);return packet

    def observe(self,row,context):
        report=self.exec.observe(row,context);self.last_row=copy.deepcopy(row)
        result=report['upperExecution']
        if not self.finished:
            if result['constraintFailure']:
                if self.exec.plan['target']=='U':self.probe_failed=True
                self.query_reason='actual-execution-limited'
            elif result['readyForNext']:
                if self.interrupted and not self.probe_added and self.exec.plan['kind'] in ('listen','cancel'):
                    self.probe_added=True;self.receive_user(PROBE)
                elif self.probe_failed and self.exec.plan['target']=='C' and not self.close_added:
                    self.close_added=True;self.receive_user(CLOSE)
                elif self.exec.plan['finish']:self.finished=True
                else:self.query_reason='actual-held'
        report['modelLoop']=dict(finished=self.finished,decisions=len(self.history),queryReason=self.query_reason,
                                 probeFailed=self.probe_failed,interrupted=self.interrupted)
        return report


def execute(root,source,config=None,provider='a6api2',model='deepseek-v4.1-flash'):
    root.mkdir(parents=True,exist_ok=False);scene,markers,preflight=build_scene(source)
    for key in scene:scene[key]['armReferenceFrame']='body-following material pose' if key=='E' else 'model-local fixed scene target'
    client=None
    if config is not None:
        from tools.control.opencode_model_client import OpenCodeModelClient
        client=OpenCodeModelClient(config,provider,model)
    p=Provider(root,scene,client);shutil.copy2(source,root/'source-map.json')
    if client is not None:write_spec(root/'model-service.json',client.public_info())
    write_spec(root/'scene.json',scene);write_spec(root/'core-preflight.json',preflight)
    write_spec(root/'acceptance-protocol.json',dict(frames=721,fps=30,duration=24,physics=True,cameraMargin=1.5,
        upper=f'independent {client.provider}/{client.model} service' if client is not None else 'current conversation model through external mailbox; no automatic fake outputs',
        goals=['language/body jointly generated','nontrivial torso + head and larger arms','actual mid-motion user interruption',
               'actual failure fed back to model; model chooses revised plan','final safe idle'],
        posture='six finite bounded authored expression axes; actual applied state and final geometry recorded',
        completion='actual position/terminal state + gaze/fingers + posture input stop; hold as declared',
        visualAcceptance=False,realTime=False,productionServices=False))
    run(root/'sdk',physics=True,request_stream=dict(frames=[dict(stage='live-model-provider') for _ in range(720)],metadata=dict(driver='model/actual-SDK feedback mailbox')),
        request_provider=p,native_five=True,native_limits=NATIVE_ROUTE_LIMITS.copy(),native_fingers=True,
        native_gaze=True,face_control=True,task_markers=markers,camera_margin=1.5,command_timeout=600)
    write_spec(root/'model-history.json',dict(decisions=p.history,messages=p.messages,finished=p.finished,
        interrupted=p.interrupted,probeFailed=p.probe_failed,bodyEvents=p.exec.body.events,
        budgetExhausted=p.budget_stop,driverGuards=p.guards,modelFinished=p.finished and not p.budget_stop))
    write_spec(root/'requested-packets.json',p.packets)


if __name__=='__main__':execute(Path(sys.argv[1]),Path(sys.argv[2]),Path(sys.argv[3]) if len(sys.argv)>3 else None,
                             sys.argv[4] if len(sys.argv)>4 else 'a6api2',sys.argv[5] if len(sys.argv)>5 else 'deepseek-v4.1-flash')
