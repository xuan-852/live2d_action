"""Use the explicitly selected OpenCode provider for isolated model calls.

Config/credentials stay on the machine. No headers, secrets or full config are
copied into experiment logs. Only generated content and response metadata persist.
"""
import json,os,re,time
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError,URLError

SYSTEM='''你是符玄独立Live2D实验的上层表达组织者。根据用户对话、当前身体实际状态和执行残差，生成一条语言与身体一致的计划。
只返回一个JSON对象，不要markdown、工具调用、思维过程或额外字段。字段必须是：id,kind,text,target,affect,utteranceSeconds,posture,reachSeconds,holdSeconds,finish,explanation。
kind可用reference/say/listen/think/decline/cancel；target只能为场景已有身份或user，reference不能target=user。affect为neutral/warm/uncertain/firm。id每次唯一。
posture必须是JSON对象，不能是数组。格式如{"headTurn":0,"headNod":0,"headTilt":0,"bodyTurn":0,"bodyLean":0,"bodyTilt":0}，单位为已有模型作者表达轴的度数，不是完整人体3D角；绝对上限分别20,15,12,6,4,5。让头身也参与表达，幅度适当明显，但不要每句话都摆同样动作。
reachSeconds在0.3..5，holdSeconds在0.3..2，utteranceSeconds在0..20；finish为布尔，只有cancel/listen可finish=true。cancel的posture必须全0，表示连续收回并回正。
reference的腕/肘/掌目标来自场景登记，不能改场景、注入参数或假装执行到了；身体姿态变化可能与固定手臂目标冲突。使用实际反馈选择性调整头身某一轴/目标/时机，不缩小全部动作来掩盖失败。
explanation只给一句简短的决策说明，不要思维过程。任务结束必须明确生成finish=true的安全等待计划。模型推理耗时不计入动作时钟，本实验不是实时延迟认证。
用户打断时应立即停止旧意图并收回；实际受限时不要重复硬撑或声称已做到。目标held只表示规定内实际执行达标，不证明视觉语义。'''

SYSTEM+='''
执行协议至关重要：只有kind=reference且target=场景身份才实际驱动手臂。reference/E=双臂展开；reference/C=原生手臂抬到高位；reference/A=原生手臂指右；reference/U=试探远目标。
say/listen/think/decline都把双臂设为中性，只执行头身/视线；不能用say来替代用户要求的展开、举手或指向。文字描述不等于动作执行。
lastPlan是上一条真实执行的结构化计划。以lastExecution对应的实际结果判断，不根据你之前说过的话推断动作完成。activeUserRequest是当前有效用户请求；新用户任务可以取代旧的暂停要求。一次子目标受限不自动代表整个当前任务结束。'''


def content_json(content):
    if not isinstance(content,str):raise ValueError('model content must be text')
    s=content.strip()
    if s.startswith('```'):
        s=re.sub(r'^```(?:json)?\s*','',s);s=re.sub(r'\s*```$','',s)
    result=json.loads(s)
    if not isinstance(result,dict):raise ValueError('model plan must be object')
    return result


class OpenCodeModelClient:
    def __init__(self,config,provider='a6api2',model='deepseek-v4.1-flash'):
        self.config=Path(config);d=json.loads(self.config.read_text(encoding='utf-8-sig'))
        selected=d['provider'][provider]
        if selected.get('npm')!='@ai-sdk/openai-compatible':raise ValueError('selected provider protocol is not OpenAI compatible')
        default=d.get('model','')
        if model is None and not default.startswith(provider+'/'):raise ValueError('default model must belong to explicitly selected provider')
        self.model=model if model is not None else default.split('/',1)[1]
        if self.model not in selected['models']:raise ValueError('model missing from provider registration')
        opts=selected['options'];self.base_url=opts['baseURL'].rstrip('/');self.key=opts.get('apiKey','')
        env=re.fullmatch(r'\{env:([^}]+)\}',self.key)
        if env:self.key=os.environ[env.group(1)]
        self.provider=provider

    def public_info(self):
        return dict(provider=self.provider,modelRequested=self.model,baseURL=self.base_url,
                    sourceConfig=str(self.config),credentialsLogged=False,protocol='OpenAI-compatible chat/completions')

    def respond(self,request,repair=None):
        prompt=dict(requestId=request['requestId'],reason=request['reason'],observationTime=request['observationTime'],
                    conversation=request['conversation'],scene=request['scene'],actual=request['actual'],
                    lastExecution=request['lastExecution'],lastPlan=request.get('lastPlan'),
                    activeUserRequest=request.get('activeUserRequest'),motorSemantics=request.get('motorSemantics'),
                    outputContract=request['outputContract'])
        if repair:prompt['validationErrorToCorrect']=repair
        payload=dict(model=self.model,stream=False,max_tokens=4000,
                     messages=[dict(role='system',content=SYSTEM),dict(role='user',content=json.dumps(prompt,ensure_ascii=False))])
        headers={'Content-Type':'application/json'}
        if self.key:headers['Authorization']='Bearer '+self.key
        req=Request(self.base_url+'/chat/completions',data=json.dumps(payload,ensure_ascii=False).encode('utf-8'),headers=headers,method='POST')
        start=time.monotonic()
        try:
            with urlopen(req,timeout=180) as response:data=json.load(response)
        except HTTPError as error:
            raise RuntimeError('isolated model HTTP status '+str(error.code)) from None
        except (URLError,TimeoutError):raise RuntimeError('isolated model transport failed or timed out') from None
        choice=data['choices'][0];content=choice['message'].get('content')
        # Never persist reasoning_content or authentication headers.
        meta=dict(provider=self.provider,modelRequested=self.model,modelReturned=data.get('model'),
                  responseId=data.get('id'),usage=data.get('usage'),finishReason=choice.get('finish_reason'),
                  wallSeconds=time.monotonic()-start,content=content)
        try:plan=content_json(content)
        except (ValueError,TypeError):plan=dict(invalidGeneratedContent=content)
        return dict(requestId=request['requestId'],observationTime=request['observationTime'],plan=plan),meta
