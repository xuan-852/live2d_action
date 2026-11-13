# 补丁掌面投影：形状、袖口与路线的边界

日期2026-11-12。用户审看上一版MMD后明确反馈补丁透视仍怪、不如原生自然，不能把上一轮数值改善视为视觉通过。证据根 `E:\FuXuanLab\runs\20261112-patch-hand-shape-01`；baseline保留开始状态、diff及修改前文件，检查点仍39f895c，未提交/推送。

## 已确认与纠正

旧接入只消费三段长度与平面掌向，宽度1、剪切0；源开头掌长约.485，使整手轴向压扁。源旁车有palmNormalZ但没有消费；单个长度或normalZ不足以唯一确定完整投影。新候选用原有palmWidth/palmShear执行器，补齐有状态八轴控制，不改拓扑、贴图、腕四点、原生控制和头身映射。

先保留正负剪切候选，随后找到已有 `20261030-whole-performance-adaptation-01/references/mmd-hair.json` 的54帧三维FK。用腕到四个MCP平均点作长轴、小指到食指作横轴并在三维正交化。逐帧核对肩/肘/腕/指根中心的XY、30Hz时间、掌长、掌向和normalZ，误差在1e-7量级内才接受；并非凭旧文件名认定匹配。正负剪切实际随源变化，最终不使用固定猜测分支。参考文件、哈希保存在最终批中。

数学上局部投影为 `[[length,shear],[0,width]]`，单位正交掌面的投影奇异值为1和|normalZ|。加入非线性可行检查及停止检查，不能靠拉长横向去追不可达侧视。镜像Python和SDK实际末端网格读回分开；SDK从最终手网格三角形恢复仿射，独立审计实际长度/宽度/剪切，不只记录命令。

直接把横向掌投影混进袖口会挤压衣物并与保护冲突；改为裸手用掌面投影，ArtMesh261保留原来的纵向投影和转腕连接。单位横向投影行为不变。手部raw规划底线提高为.36，肩/衣物仍.30；实际raw≥.28、守卫≥.50、零翻转和原速度/加速度门槛不放宽。

接近侧面时单面素材无法忠实表现：保留原始请求，先对横向形状退让，规划参考至少保留.38投影面积并满足无额外拉伸；不修改上层腕/肘/纵向请求，返回palm-plane-limited与原始残差。该参考不是声称已执行源完整掌面。

另一失败是独立限步让宽度先闭合、剪切未收完，物理可行检查造成退出卡住。新增八轴专用共同有限步候选，在相同整帧参数、实际节点、几何、停止约束内选择可执行比例；旧六轴/原生路径不启用。这是下层联合执行，不是编排上层动作。中间mmd-route-final最后15帧保持失败，不被最终版覆盖。

## 实际验证

初始116项，最终 **128项单测实跑通过**；包含正交投影奇异值、正负分支、FK错配拒绝、原路线/时间不变、不可达侧视退让、八轴回中性及验收不能忽略掌面残差。旧评估器只检查节点会把edge-limit误报通过，已补入projectionStateMet/palmBasisStateMet；直接测试按预注册最后15帧保持窗口判定，过渡全帧仍保留。

最终以以下两批为准，均在隔离Unity完整编译录制，不启动生产桌宠/服务：

- `direct-coordinated-final/sdk`：301个PNG/JSON配对帧，10秒。正负斜向、平面恢复与最终恢复的最后15帧均达标；edge-limit的原始掌面请求0/60帧达标，明确保留失败，随后恢复。实际网格/命令基误差≤3.1e-6，无安全失败。
- `mmd-coordinated-final/sdk`：145帧，4.8秒（含进入/退出），原54帧源时间不变。无安全失败；实际raw最小.380、守卫.556、最大形状奇异值约1.000002（浮点误差），保护公式复算误差<5e-8。最后30帧保持通过。源内掌面仅5/54帧完整达标；补丁腕RMS .00726、掌向3.02°，上一版长度候选为.00499/2.95°，存在跟踪代价，不能称全面改善或完整动作通过。

所有早期批次保留：固定正负分支、全网格高面积裕量、物理可行约束、FK、袖口解耦、横向退让、协调退出。初版确有raw/守卫越界，后续有明显跟踪与退出退化；详见各自track.jsonl、实现快照和verification.json，不挑片段替代失败。

## 审看与剩余问题

`compare-coordinated-final/comparison.mp4`：左旧轴向压缩，右新掌面候选，同钟固定取景；`mmd-coordinated-final/sdk/stateful-arms-raw.mp4` 为无叠加原图，review-fit同步请求/预测/实际节点、长度、掌面请求、实际应用和限制。直接控制原图与节点版也保留。

候选仍是单面二维素材的掌面仿射投影，没有新增手指深度/姿态、掌背翻面、体积遮挡或原生作者的专用轮廓；极侧视仍不能完成，不能把几何安全当作自然透视。上/前臂仍沿用上一轮长度实现，表情未改；新八轴尚未通过15秒BodyExecutionSession组合验证。新版视觉是否比旧版自然，待用户看整段。下一步以画面指出的剩余形状问题决定是否需要局部非仿射或额外视图素材，不能继续用参数拟合掩盖。

复现（新目录）：

```powershell
python -m unittest discover -s tests -p "test_*.py"
python tools/control/palm_plane_trial.py <new-direct-root>
python tools/control/mmd_timed_controller.py <new-mmd-root> E:\FuXuanLab\runs\20261112-mmd-timed-controller-01\source-map.json --patch-projection --patch-rooted --palm-reference E:\FuXuanLab\runs\20261030-whole-performance-adaptation-01\references\mmd-hair.json
```
