# 补丁臂正面投影长度：独立候选与实际网格反馈

后续：[接回MMD与肩袖瓶颈](20261112-patch-mmd-integration.md)已实跑新参考/已有rootedUpper模式；本文保留旧候选结果与守卫语义，不代表新版本。

日期：2026-11-12。用户认可前一15秒身体会话的整体效果，同时指出补丁手“透视”怪，明确指正面看到的长度；授权先Git存档再处理。证据根目录：`E:\FuXuanLab\runs\20261112-patch-projection-control-01`。**本轮候选尚无用户视觉认可，不替换已认可基线。**表情未在本轮扩展。

## 存档与范围

已提交 **39f895c16df46129889ad24d5b0058ba51b56004**，`chore(live2d): 存档获认可的连续身体执行会话`，12个文本文件，未推送。提交前102项测试及暂存diff检查通过。钩子提示本机缺少lefthook，但Git提交实际成功；未安装依赖。存档同时记录了用户对session-final整体效果的认可及补丁投影/表情余项。

本轮改动在检查点之后，未再次提交。没有改模型拓扑、贴图、第三方SDK、原生求解器或头身映射；只修改本仓库的补丁控制器与隔离SDK包装/读回。腕点严格沿用 `20261109-wrist-axis-revision-01/protocol.json::newNodes`，未重新猜点。没有参数普查、网络训练、下载动作资源或启动生产桌宠/AI。

## 当前问题与实现

原 `WholeArmSession` 已支持上臂/前臂/手掌长度变形，但有状态控制始终把这些量锁为1，只有三个平面角轴；MMD投影缩短无法执行。现在新增显式候选模式 `--patch-projection`：

- `Target.projection=[upper,forearm,palm]` 表达相对当帧原始材料链的正面投影长度；源MMD三段投影比原样输入，不因执行范围而裁剪请求。腕/肘位置、掌向、时钟、头身输入保留。它不是可访问的三维骨骼或掌面法线。
- 新 `PatchProjectionController` 六轴：原肩/肘/腕角＋三段投影长度。内部长度轴单位为 `(ratio-1)*100`，沿用整帧参数135/1620、节点.72/16.2预算，另留.1参数加速度量化裕量，实际门槛不放宽。正面执行包络：上臂.75..1、前臂.55..1、手掌.4..1；宽度1、shear0，无翻面。
- 现有手掌正比例投影本身有正面积行列式；旧.55下限不能覆盖本条MMD手掌最小.4836。因此仅显式正面控制模式把手掌候选包络扩到.4，SDK实际验证包含.45；旧 `ReachValid` 和默认三轴入口仍保留原范围。
- 新 `PatchProjectionPlant` 镜像现有 `ShoulderConnection`，使用相同已认可四点的同一最大最小重心权重绑定、刚性MLS肩连接、袖口混合、沿轴投影与旋转；不把理想刚杆链当最终网格。每帧用实际SDK应用参数刷新原始材料上下文。直接任务的预测/实际最大节点偏差约8.6e-8，实际网格仍为最终真值。
- 联合有界逆解同时处理节点、掌向及长度；完整候选只在帧级与几何检查通过时执行。受限步使用任务空间误差而非参数各走同一比例；掌向偏好权重3公开记录，防止缩短时随意牺牲掌向，但不保证冲突目标能全部满足。hold继续按当帧上下文拟合并保留deadline。
- SDK包装输出六轴应用值、最终三段所见长度比、六轴逐画件面积梯度。请求比、执行器比与最终材料链比分别记录，不能互相冒充。
- 身体会话的freeze保留实际读回投影；projectionStateMet也参与settled判断，防止位置到达但长度未到位时伪报成功。默认旧接口仍固定投影，候选需显式启用。

## 几何保护的投影语义

第一版沿用绝对面积门槛，把正常投影导致的面积减少也当成额外塌陷，实际MMD多次被卡住。该绝对门槛版本与一次连续性求解失败均保留，未隐藏。

候选投影模式同时记录原始面积与以下守卫量，各画件取最坏三角形：

`guard = min(rawArea / expectedProjectionFloor, sameAngleIdentityProjectionArea, rawArea * .55/.30)`

- 手部期望投影面积因子为掌长×掌宽；前臂袖口用现有helper的前臂/掌部投影下界；上臂附件用上臂因子。对MLS/袖口它是现有分段近似，不是3D形状正确性的证明。
- 同角度、未缩短时的面积守卫独立保留，不能靠“归一化”掩盖本来就有的角度塌陷；原始绝对面积也有独立下界。
- 规划guard≥.55、原始面积≥.30；实际验收guard≥.5且原始面积≥.28，翻转0。**这不是旧绝对面积≥.5门槛下通过。**旧三轴模式不改其几何标准。
- 新模式中SDK读回偶尔略低于规划裕量时，允许在实际硬门槛以上不继续恶化并恢复，避免因.55附近数值误差连零步也被判不可行；不降低实际.5守卫门槛。
- 原始面积、同角度未投影面积、期望因子、守卫面积均保留。独立Python公式对SDK守卫数组最大差约6.95e-8；这只验证公式接线，不认证自然度。

最终MMD实际原始面积最小约.4573（低于旧.5），投影守卫最小约.5497；无翻转/实际failed。必须同时看原图、连接、袖口与轮廓，不能只看归一化数字。

## 固定直接任务结果

`direct-range-final`：30帧进入＋上臂/前臂/手掌独立缩短各60帧＋组合60帧＋缩短时正负转腕60帧＋返回60帧，共391个SDK配对帧、13秒。物理关闭以分离投影；不宣称覆盖身体变化场景。初始范围、时长、腕/肘.005、掌向4°、比例误差.03及几何门槛在开跑前写入。

- 上臂实际最短约.800，前臂.750，手掌.450；宽度不变，能够从长度轴执行，而非靠额外弯曲假装缩短。
- 五个动态跟踪阶段共300帧，全部通过位置/掌向与比例门槛，包括组合投影与正负转腕；无failed，返回最后15帧稳定。进入/返回过程仍保留逐帧误差，未拿它们的非零误差假称全程到位。
- 手掌独立缩短段60/60帧达标，腕RMS约.000017、掌向RMS约.000036°；缩短时转腕掌向RMS约.673°，比例最大误差约.0103，均在声明范围。
- 旧固定投影对照仍接收节点请求但明确报告projection-control-unsupported；手掌长度保持约1。直接对照后期手掌测试扩到.45，旧批的未执行投影请求最低.65，不能冒称两批全部投影请求数值完全相同。腕/肘/掌向路线和原时钟关系保留，视频明确显示新增投影能力。

## MMD与整体会话：进展和取舍

最终候选为 `mmd-range-final` / `body-range-final`，源仍同一makoto-hair，未改帧率、删帧或降低全动作。MMD原54帧位置/朝向/头身保持原请求，同时新增原先未执行的手掌投影比。

| 补丁源54帧 | 已认可固定投影基线 | 当前候选 |
|---|---:|---:|
| 腕RMS | .029876 | .008165 |
| 肘RMS | .017998 | .013621 |
| 掌向RMS | 1.870° | **6.562°（退化）** |
| 实际最小上臂/前臂/手掌比 | 约1/1/1 | .832/.779/.485 |
| 所有比例同时达标 | 未支持 | 15/54 |

候选已经能执行接近源最短48%的手掌投影，但不是整条MMD投影/姿态通过。几何、绝对节点、方向与比例仍有冲突，最大逐帧比例残差约.216；掌向存在明显取舍。不能只用“更短”或位置误差下降宣称更自然，不能把受限解视作模型全局上限。已认可基线保留，不自动替换。

15秒连续会话451个配对帧，全帧两侧安全/连续性/配对通过、无failed，最终中性收稳；完整源54帧、打断15帧前缀、实际姿态打断、请求归属与实际反馈检查保留。高位与低位目标最后保持通过；freeze最终能收稳，但没有在预定整个窗口内都满足，`freezeHold=false`、`wholeChainEngineeringPassed=false`继续保留。候选与基线的freeze目标来自各自实际姿态，不能假称整段的所有目标完全相同。

## 验证、版本与观看

初始102项实跑通过；新增投影身份与分段缩短、有界逆解、范围残差、非法输入、JSON反馈、源比不裁剪、freeze保留实际长度/禁止伪settled、守卫区分缩短与塌陷、比例雅可比等回归，最终 **111项测试实跑通过**。C#仅隔离工程编译/录制，不启动生产桌宠。

保留所有中间版本：绝对面积版、投影守卫版、不同受限误差权衡、原.55手掌包络、当前.4候选。`direct-projection-01` 仅2个SDK帧后因numpy布尔JSON序列化中断，不算完成；已修复并加回归。比较器曾有numpy整数JSON错误，未影响SDK原录制，重新生成到新目录。所有当前验收以range-final三批与comparison JSON为准，不挑单帧替换失败。

`RecoveredArmTypes.cs` 原先无BOM，本轮补成UTF8 BOM；已保存实际已编译的原字节版本，除BOM外文本一致。补BOM后额外隔离SDK两帧encoding-smoke完成，未重写原录制快照。对应哈希/完成计数见verification.json。

- `direct-range-comparison/comparison.mp4`：13秒固定区域前后局部对照，观察三段独立缩短、恢复与袖口；相同固定取景，不逐帧改变缩放。
- `body-range-comparison/comparison.mp4`：15秒整体链路局部对照；`body-range-final/sdk/stateful-arms-raw.mp4`保留完整全身原图。
- `mmd-range-comparison/comparison.mp4`：完整4.8秒旧/新候选对照，包含进入和退出，不隐藏掌向退化。
- 各range-final/sdk的review-fit含请求、实际节点、实际长度比、阶段与限制；原PNG/JSON、协议、源码、失败与中间版全保留。

复现（新目录）：

```powershell
python -m unittest discover -s tests -p "test_*.py"
python tools/control/patch_projection_trial.py <new-direct-root>
python tools/control/mmd_timed_controller.py <new-mmd-root> E:\FuXuanLab\runs\20261112-mmd-timed-controller-01\source-map.json --patch-projection
python tools/control/body_session_trial.py <new-session-root> E:\FuXuanLab\runs\20261112-mmd-timed-controller-01\source-map.json --patch-projection
python tools/control/assess_body_session.py <new-session-root>
python tools/control/review_stateful_arm.py <new-session-root>\sdk --fit-requests
python tools/control/review_stateful_arm.py <new-session-root>\sdk --raw
```

下一步先人工审看“正面所见长度”是否符合预期，再按整段表现决定投影与掌向冲突的处理；当前是独立可回退候选，不擅自推广更短范围、翻面、掌背素材或所有动作自然度。
