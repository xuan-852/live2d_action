# 身体执行会话：上层请求—连续执行—实际反馈

日期：2026-11-12。用户认可上一版MMD“感觉可以”，随后要求回到北极星，并授权推进整体链路。上一条单片认可对应 `20261112-mmd-integration-followup-01/bounded-native-final`，不转移到本轮新会话。证据根目录：`E:\FuXuanLab\runs\20261112-body-execution-session-01`。

北极星：上层根据任务与内部状态组织连续身体表达，下层执行节点、朝向和时间要求并反馈实际结果。本阶段交付**可供上层调用的有状态身体执行接口与完整隔离SDK验证**，不以消除全部肘误差为阶段目标，也不把测试编排当上层学习。

后续用户反馈：“可以啊整体效果，这代表我们这条路应该是走对了”，同时指出脸部表情不生动、补丁手正面所见投影长度/透视仍怪。该认可限定为本轮session-final的整体执行效果，不代表补丁投影/全部保持目标或表情通过。用户授权先Git存档，再优先处理补丁正面投影缩短；下文“未人工认可”保留为录制当时状态。

## 设计与职责

新增 `tools/control/body_session.py::BodyExecutionSession`，位于既有原生/补丁控制器与请求提供方之间，不改两侧求解器、模型、SDK或物理配置。手臂执行器、速度、分支状态仍保存在原控制器中；会话不重置这些状态。

- 输入：请求ID；model-local-xy坐标（x右/y上），native=解剖左/画面右，patch=解剖右/画面左；双侧腕目标、可选肘/肩/分支、平面掌方向；已有头身六通道输入；绝对动作时钟deadline或带原始时间的30Hz路线。
- 输出：当前请求归属、逻辑阶段、实际执行状态，双侧最终网格节点、应用参数、区间速度、残差和限制原因；头身六通道应用值及原SDK身体物理输出上下文。逻辑播放结束不等于姿态成功。
- `describe()`：返回实际后端身份、执行轴与预算、30Hz时钟、坐标系、头身输入/物理方式、补丁identity投影及不支持的独立肩平移/3D掌面/指形。不是全人体三维骨骼接口，也未新增任意头身节点逆解。
- `submit_route(id, samples, entry_seconds, now)`：显式进入段后按原时钟交付路线。源样本/头身输入绕过会话连接器，未经裁剪、缩小或重定时；原路线结束后转为保持并保留残差。
- `submit_pose(id, arms, head_body, deadline, now)`：新目标立即取得请求归属，继承执行器实际运动状态；到位与收停仍由实际反馈判定。未到位时不自动延长deadline。
- `freeze(id, now, deadline)`：以**上一帧SDK实际姿态**建立控制收停/保持目标；不会把物理粒子冻结，也不保证身体上下文继续变化时能严格锁住所有模型局部节点。
- `release(id, now, deadline)`：回到会话初始实际中性与头身输入；达到deadline且实际反馈收稳后可进入idle。idle期间仍保持反馈与底层修正，新的请求可继续接管。
- 请求历史分别记录superseded/interrupted/released与交付的源样本数；中断后的旧路线不会暗中续播。非法请求不替换已有请求，反馈是独立拥有的快照，调用方不能通过改返回对象污染会话状态。

头身在源外使用继承最近实际输入速度的五次边界连接，维持既有六输入范围；若连接超范围会明确报告，不静默当成功。SDK物理全程连续运行。原source时刻与动作时刻分别记录，`observationTime`是本次命令规划所用上一SDK帧时间，反馈`time`是本次执行后实际读回时间。

`validate_stateful_arm.run` 增加可选 `request_provider` 回调，在每帧真实SDK读回后更新会话，下一帧再取新请求；原静态request_stream入口保持。每个原生/补丁控制器只初始化一次，Unity场景/物理只创建一次。新接口当前是本地Python＋隔离Unity固定30Hz研究接口，未接生产服务或宣称实时桌面延迟。

## 固定验收场景

`body_session_trial.py` 明确是外部上层请求验收fixture。复用同一makoto-hair MMD，仅在源外添加进入/保持/换目标；没有新增训练/动作数据。450控制帧、15秒，另1初始化帧：

| 动作帧 | 上层请求 | 目的 |
|---|---|---|
| 1 | mmd-full，1秒进入 | 完整交付54个原源样本，然后保持 |
| 121 | pose-high | 从MMD接管双臂与头身，到直接高位目标 |
| 211 | mmd-interrupted，.8秒进入 | 在同一物理上下文重新接入路线 |
| 250 | freeze-actual，.5秒收停要求 | 用249帧真实姿态打断，原路线仅交付15个源样本 |
| 280 | pose-low，.8秒截止 | 从收停目标重新接管，双臂与头身同时换目标 |
| 340 | release，1.2秒截止 | 回中性并持续反馈至450帧 |

初始范围/时长/门槛在各批acceptance-protocol.json、upper-requests.json中，首次数据未覆盖。原生参数2700/162000、节点3/180；补丁135/1620、节点.72/16.2；面积≥.5、翻转0、配对/恢复≤1e-6。保持仍用腕/肘.005、掌4°、所有节点速度.02、掌角速度8°/秒。没有因组合失败放宽门槛或缩小全部动作。

## 实际结果与能力边界

`session-01` 与最终 `session-final` 各451个SDK配对帧完成，共902帧；最终补充了能力描述、身体物理上下文和反馈快照隔离，动作与首版一致。所有帧保留PNG/JSON、源码快照及complete标记。

接口/生命周期检查通过：

- 完整MMD原54个时间点全部交付，腕/肘输入最大差0、掌方向差8.53e-14°、头身输入差0。第二次路线交付15个源样本后被明确中断，没有把未交付部分算完成。
- 六个请求均在指定事件帧接管；freeze目标腕/肘/掌向与249帧实际节点一致。后续没有旧路线泄漏。控制器与SDK物理不因请求切换重置。
- 实际反馈年龄为一个控制帧1/30秒。导出的最终节点、执行器应用值、身体物理上下文与独立SDK帧JSON最大差均0，验证的是实际读回而不是理想预测。
- 全帧两侧安全/连续性/几何/配对门槛通过，无failed；本例没有头身连接裁剪。最终15帧双臂实际收稳、反馈settled，处于release/idle。头身最大输入步长3.0244°/帧被记录，不作为自然度分数。

**姿态能力没有全部通过：**

- 高位直接目标最后15帧双侧保持通过。
- freeze最后窗口原生收稳，但补丁仍有约.00504腕误差、6.77°掌方向误差，严格保持失败；冻结的是目标，不是停止整个SDK物理。
- 后续低位直接目标原生到位，但补丁最后腕误差约.02290、肘.01706，保持失败；接口持续报告limited。
- 原MMD仍保留原生肘与补丁位置限制。MMD播放到结尾时实际执行状态仍可limited，未伪造completed/reached。

因此 `lifecycleInterfacePassed=true`，但原严格 `wholeChainEngineeringPassed=false` 仍保留；不是把两处失败从验收中删掉。分开汇总是为了让上层看清“请求接管/反馈/退出可用”与“该姿态能否满足”。后续工作可以围绕对整体表达影响最大的能力限制推进，不回到逐帧指标修饰。

## 验证、复现与观看

初始95项实跑通过；新增完整源时钟保持、实际姿态打断且不续播、release不继承旧settled、受限反馈/时钟配对、非法请求不替换、反馈不可变快照、头身速度继承，共7项，完整 **102项测试实跑通过**。中间能力描述接线曾有括号语法错误，已修正并通过完整测试后完成最终SDK录制；不把失败导入当完成批。

```powershell
python -m unittest discover -s tests -p "test_*.py"
python tools/control/body_session_trial.py <new-root> E:\FuXuanLab\runs\20261112-mmd-timed-controller-01\source-map.json
python tools/control/assess_body_session.py <new-root>
python tools/control/review_stateful_arm.py <new-root>\sdk --fit-requests
python tools/control/review_stateful_arm.py <new-root>\sdk --raw
```

- `session-final/sdk/stateful-arms-raw.mp4`：完整15秒无叠加，一次会话，无剪接/循环/补帧。
- 同目录 `stateful-arms-review-fit.mp4`：请求／下一步可执行预测／实际SDK节点／原图，以及请求ID、逻辑阶段、实际执行状态、反馈时刻与限制。
- `session-final/session-assessment.json`：原严格门槛、两处保持失败、请求历史均保留；`capabilities.json`与`session-history.json`可供上层接入参考。

开始前status、相关diff与直接文件快照在baseline。此前未提交的原生边界修复未覆盖；Git检查点仍06153c8，本轮未再次提交或推送。未启动生产桌宠/AI、安装依赖、下载资源或改模型/SDK/整体动作系统。新会话画面尚无用户认可，不转移上一单片“可以”的反馈。
