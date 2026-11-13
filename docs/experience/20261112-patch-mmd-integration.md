# 补丁正面投影接回 MMD（2026-11-12）

用户要求继续处理正面所见长度，最终直接观看MMD。保留检查点39f895c，新增改动未提交。证据根：`E:\FuXuanLab\runs\20261112-patch-mmd-integration-01`；开始状态、diff、文件快照在baseline。

## 定位与实现

上一投影候选的MMD掌向RMS6.56°。先加逐帧掌向允许区间，冲突时仅退让任务约束并报告，保留运动/几何安全；palm-envelope-01为6.80°。再改成同投影零角参考，projected-reference-01为7.12°。两批各145个SDK配对帧，证明单加约束或改变保护参考不足，全部保留。

三个全目标静态探针定位真实肩袖压缩：第14源帧ArtMesh252原始最小面积比.259、守卫.292。相同请求、身体、腕点，复用已有rootedUpper分布式肩根连接后.429/.559，无翻转。static-goal-geometry与static-goal-rooted各3帧，只证明静态差别，不是连续动作验收。

新候选显式启用 `--patch-projection --patch-rooted`。Python模型镜像已有根部权重、缩短、旋转次序；反馈读SDK最终网格。已认可腕四点、贴图、拓扑、原生控制和头身映射保留；没有新增变形模式或修改第三方SDK。请求、预测、实际节点及实际长度分别记录。

保护参考采用同投影逐三角形面积，同时检查额外角度压缩、纯投影合法性、原始面积底线。单位投影恢复旧面积语义；规划原始面积.30/守卫.55、实际.28/.50及零翻转要求保留。诊断数组可独立复算，最终MMD公式误差约7e-8。不是允许缩短掩盖塌陷。

## 实跑结果与限制

116项单测实际通过。隔离Unity最终三批：mmd-final/sdk145帧、body-final/sdk451帧、direct-final/sdk391帧，全部保存PNG/JSON、协议、实际参数/物理与实现快照。

- 同源54帧、原30Hz、不重定时：补丁腕RMS .00499、肘.00333、掌向2.95°；上一投影候选.00816/.01362/6.56°。三段长度同时在.03内39/54帧；位置和掌向同时达标27/54，不能称完整目标通过。
- MMD所有阶段安全检查、最终保持通过；进入/返回和源内受限、截止失败仍保留。rooted-mmd-01先行145帧与最终源跟踪结果一致。
- 直接三段缩短、组合、投影下转腕共300动态帧全部达标；进入/返回不是全帧到位。
- 15秒会话源目标/头身输入保留、实际反馈一致、接管/退出及高低位保持通过。freeze窗口仍失败，最终虽收稳，wholeChainEngineeringPassed仍false。

原图与对照可观看不等于自然度获认可。没有实现掌背翻面、手指姿态或真实3D体积透视；表情本轮未改。仍需用户审看正面长度、肩袖连接和整体动作。

## 观看与复现

首看mmd-final/sdk/stateful-arms-raw.mp4；compare-fixed/comparison.mp4为同钟旧固定长度/新候选的固定取景对照。review-fit同步显示请求、预测、最终节点、阶段与限制；body-final/sdk原图为15秒整体链路。所有失败和中间版保留，入口REVIEW.md。

```powershell
python -m unittest discover -s tests -p "test_*.py"
python tools/control/mmd_timed_controller.py <new-root> E:\FuXuanLab\runs\20261112-mmd-timed-controller-01\source-map.json --patch-projection --patch-rooted
python tools/control/body_session_trial.py <new-root> E:\FuXuanLab\runs\20261112-mmd-timed-controller-01\source-map.json --patch-projection --patch-rooted
python tools/control/patch_projection_trial.py <new-root> --patch-rooted
```
