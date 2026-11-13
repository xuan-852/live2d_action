# 独立 DeepSeek 模型参与的身体表达闭环

2026-11-13。用户授权接入生成模型、让身体动作参与且幅度适当大；随后指定读取本机OpenCode配置，并明确改用DeepSeek V4.1 Flash。当前默认测试入口为a6api2/deepseek-v4.1-flash；不再自动沿用OpenCode全局默认Astra，未修改全局配置。

## 交付范围

本轮由独立服务实际生成语言、头身姿态、双臂意图和时间要求；每次下一请求包含上一条完整计划及真实SDK执行结果。最终不是手写响应列表，也不是当前对话模型代替服务填写计划。测试对话INITIAL/INTERRUPT/PROBE/CLOSE是实验输入，模型响应没有手动改写。

功能链路：用户对话与有效任务 → 独立模型JSON计划 → 本地严格校验 → 有状态执行器 → SDK应用参数和最终材料节点 → 模型修订。24秒动作时钟持续采集，模型推理期间暂停动作时钟；8次模型请求合计墙钟约30.28秒，另有采集/求解耗时。**这是分步闭环，未认证实时桌宠延迟。** 未启动生产桌宠、未进入生产记忆/桥接链路；只访问用户明确指定的OpenCode提供方。

## 输入和实现

- `opencode_model_client.py`：仅在内存读取配置中选定提供方的地址/认证，不复制配置、不输出密钥，不记录headers或reasoning_content。通过OpenAI-compatible chat/completions调用，保留生成正文、请求/返回模型ID、usage、耗时、拒绝样本。响应不合格式时最多有限次向同一模型请求修正，不切换模型。传输协议参考：[AI SDK兼容提供方文档](https://ai-sdk.dev/providers/openai-compatible-providers)。
- `embodied_plan.py`：严格计划边界。字段id/kind/text/target/affect/utteranceSeconds/posture/reachSeconds/holdSeconds/finish/explanation；拒绝未知字段、非法/非有限值、任意执行器数组。posture为headTurn/headNod/headTilt/bodyTurn/bodyLean/bodyTilt六个作者表达轴，上限20/15/12/6/4/5；不是完整人体三维关节角，也不是力度标量。
- kind=reference才调用场景登记的手臂目标；E双臂展开、C高位、A右侧指向、U受限远目标。say/listen等只表达头身/视线并使双臂回到身体相对中性；说“举手”不等于执行了举手。这项语义明确发送给模型。
- `model_body_loop.py`：独立服务/隔离文件桥。请求包含activeUserRequest、lastPlan、lastExecution，区分旧打断和新的有效任务；响应与观测时间/请求ID绑定。结束当前表达不会禁止之后的新用户任务。语言结束与运动目标生命周期分开，短句不会提前撤销尚未执行完的动作。
- `DialogueBody.submit`可选接收经验证的头身表达轴和reachSeconds，旧入口默认行为保留。头身连接继承实际状态，手臂/手指/视线/脸部使用既有连续控制和真实读回。
- 限制、到位保持、用户打断会触发下一次独立模型请求；驱动不预制“失败后必须输出某条计划”的响应。模型可先说明限制，再选择新动作。模型决策预算耗尽时有明确标记的机械安全收尾，不算模型完成；provider异常通过abort握手关闭本次SDK，保留部分采集。

## 身体与双臂协调的修复

初期失败暴露：头身变化时，中性手和双臂展开仍固定在初始世界坐标，补丁被迫抵消身体运动。不是据此认定模型不会动或手臂能力不足。

新增 `BodyFollowingTargets`：中性、双臂展开和未参与场景指向的补丁侧使用**随身体变化的材料参考**。原生从当前SDK应用上下文＋本帧头身输入的Core求值取节点；补丁复用批准四点与已有绑定读取当前身体下的原始材料链。A/C/U原生的场景指向目标仍保持原坐标，不搬动对象来制造成功。没有3D骨骼假设、网格缩放、剪切或重新猜腕点。

六帧既有SDK上下文复核：预测补丁原始材料链与SDK sourcePatchChain最大差2.49e-8。上下文相对姿态用内部connection_key保留同一个到达时钟与执行器终点，避免每帧材料坐标变化把连接器重新启动；键仅由下层为body-following hold生成，不用于世界坐标路线，也不让模型注入。

同一条DeepSeek历史计划作下层回放：旧固定世界解释没有保持通过，末窗口补丁腕残差最大.03533；身体相对解释后双侧末窗口保持通过、腕残差约6.04e-6/3.14e-6，随后中性退出也通过。**参考系语义被修正，派生世界节点有意改变，不能称为同一世界目标的误差消融。** 该回放不算新模型决策。

## 最终独立运行：deepseek-04

- 请求和服务返回模型ID均为deepseek-v4.1-flash，a6api2；8次真实API生成，8条计划，最终未触发预算安全接管。
- 双臂展开配合头身实际保持通过；头转/点头/头侧倾最大12/12/6，身转/身倾/身侧倾最大5/3/3（作者输入口径），不是把所有动作缩小。
- 高位动作在第193帧实际运动中收到打断，当时原生材料节点最大速度.5225单位/秒；模型生成cancel并连续收回。
- 新用户任务要求试探U；实际U未到位，限制交回模型。模型先生成说明受限的说话计划，再独立生成reference/C并取得实际保持，中间没有新的外部恢复指令；随后收到结束请求，模型生成安全收回。
- 最后15帧身体、手指、视线、脸部全部收稳。独立SDK JSON复核节点/应用值差0，Core/渲染配对差0，头身输入最大残差4.76e-7；补丁最小面积比.618167、翻转0、恢复误差0，无本轮actual连续性越界报告，原生手网格0帧出画。
- 原始deadline/局部求解限制仍保留：原生25帧deadline-missed、补丁6帧，另有终端未收稳与U局部可达性原因。不能把功能链路跑通称为所有目标按时完成。

初始162项、最终179项单元测试实跑通过；新增覆盖严格模型输入、旧反馈拒绝、认证信息不入日志、JSON反馈快照、短句不撤销运动、新用户任务重开、安全预算退出、body-following连接不重启、世界目标不被重写。隔离Unity实际编译/采集完成；provider故障另1帧abort烟测确认关闭握手，未留下等待进程。

## 失败保留和证据

证据根：`E:/FuXuanLab/runs/20261113-model-body-loop-01/`。

`run-index.json`列出全部试跑：当前对话模型调试因序列化失败中止；用户改用DeepSeek前的Astra样本单独保留，不进入DeepSeek认证；DeepSeek首轮只选say/listen，没有调用手臂；第二轮曾触旧8次预算异常；第三轮暴露身体/手臂固定参考冲突；body-following-replay只检验下层；最终deepseek-04是修复后的完整运行。没有删除失败或从同版随机结果中挑选最好片段。

- `deepseek-04/model-body-review.mp4`：24秒完整同步原图、请求/理想/实际材料链、脸部、原始模型文字、头身目标/应用值和限制。
- `deepseek-04/model-body-raw.mp4`：无叠加原图。
- `deepseek-01/model-body-review.mp4`：保留首轮失败录像，不能把其头身活动称为手臂表达通过。
- `model-mailbox/*.request.json`、`*.api-*.json`、`*.reply.json`保留原始观测/服务正文/验证拒绝/接受计划；model-history与requested-packets连接实际时钟；全部SDK PNG/JSON保留。

复现（新输出目录，配置路径只读取）：

```
python -m unittest discover -s tests -p "test_*.py"
python tools/control/model_body_loop.py <新目录> <source-map.json> C:/Users/25295/.config/opencode/opencode.json a6api2 deepseek-v4.1-flash
python tools/control/review_model_body_loop.py <同一目录>
```

## 尚未认证

新录像的可读性、自然度、支撑感和手势语义等待用户审看；旧版认可不自动转移。只在固定场景能力内选择/组合，不是任意节点路线生成、任意人体关节/三维掌向、训练或学习成果。模型目前主要消费SDK数值反馈，不是自动视觉闭环评审。嘴部为文字时钟提示，没有音频/音素同步。实时推理调度、生产集成和长时任务泛化仍待后续。

本轮未提交或推送，未安装依赖、下载模型或修改OpenCode全局配置。大型数据在runs；baseline与final-files保存本轮前后版本。
