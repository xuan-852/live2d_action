# MMD续接：Git检查点与原生边界内联合求解

日期：2026-11-12。用户要求先提交Git检查点再继续MMD接入。关联 [原生五轴目标控制](20261112-native-target-control.md)，本次只复用同一makoto-hair源，未扩数据/训练/全参数扫描。证据：`E:\FuXuanLab\runs\20261112-mmd-integration-followup-01`。

## 检查点

已提交 **06153c876dca3a9ef60ddce2a0ce65e741166dec**，`chore(live2d): 保存五轴手臂控制与MMD接入研究检查点`。覆盖此前未提交研究代码、测试、经验记录，共116个文本文件；不是宣称本轮编写了全部历史研究。模型、SDK、录像、大型数据继续留在忽略目录和runs。开始前status、tracked diff、完整暂存diff与文件清单在baseline；提交后工作区干净，未推送。

提交前完整94项测试与diff检查通过。钩子提示本机找不到lefthook，但Git实际提交成功；未安装依赖，手动测试记录保留。以下续接代码发生在检查点之后，**尚未再次提交**。

## 排查与修改

此前源frame9..23共15帧残差，不足以证明其腕部目标不可达。5个固定SDK上下文的只读探针，把“无约束Gauss-Newton求步后逐轴裁剪”换成“参数边界内联合求步”，腕误差降至约.00008～.00040，肘误差仍存在。证据在bounded-fit-probe.json，不冒充新SDK动作验收。

原因：某轴顶到边界时，先求无约束步再裁剪会丢掉其他轴应该如何补偿的信息。剩余轴（本例尤其Param98等）的可用运动没有被正确利用；单纯扩大同一方向的步长或继续提高响应预算无法解决该求解错误。

`NativeTargetController._fit_endpoint` 改成盒约束局部二次子问题，保留每次±10单位信赖范围与原误差下降线搜索；执行仍只有一个完整帧命令。**没有删掉肘要求，没有改任务权重/位置容差，没有更改源路线、时钟、幅度、头身、物理、绑定、参数边界或速度/加速度预算。**只修边界内的联合求解。

新增候选残差分量 `policy.residualComponents`（Core候选腕/肘/掌方向残差，区别于SDK实际误差），与 `wrist-target-limited` / `elbow-target-limited` / `palm-target-limited` 原因。SDK实际误差仍用记录顶层wristError/elbowError/palmErrorDeg，不拿候选当实际。未满足肘目标的帧仍是limited，不因腕部成功改成reached。

## 同输入实际SDK对照

两版新录制 bounded-native / bounded-native-final 各145个配对帧（144控制帧，含源54帧），完整PNG/JSON、源码快照和complete标记保留。最终版只是补充残差分类，动作与首版数值相同。对照检查点已有真实SDK批 `20261112-native-target-control-01/mmd-five-final/sdk`：

- 原腕/肘输入最大差 **0**，掌方向差1.14e-13°；同源54时间点、相同动作时间、头身输入、原生及补丁预算。
- 原生声明仍为参数2700/162000，节点3/180；补丁仍135/1620与.72/16.2。本次不是再次提速对照。
- 完整获取、源、返回、保持4.8秒全部保留；源9..23没有剪掉。

| 原生源54帧 | 检查点 | 新版 |
|---|---:|---:|
| 腕RMS | .017012 | **.0010243** |
| 腕最大误差 | .050378 | **.0032929** |
| 腕≤.005帧数 | 39/54 | **54/54** |
| 掌方向RMS | 7.4515° | **.23449°** |
| 掌方向最大误差 | 23.2729° | **.66537°** |
| 掌向≤4°帧数 | 43/54 | **54/54** |
| 肘RMS | .020488 | .020283 |
| 肘≤.005帧数 | 39/54 | 39/54 |
| 三项同时达标 | 39/54 | 39/54 |

腕和掌方向在这条MMD的全部源帧到位；**肘仍在9..23帧受限，最大误差.058733**，联合姿态没有全部通过。它不是为消除腕误差而新引入的更大肘误差；肘RMS/最大误差略降，但剩余冲突并未解决。不能从一个局部有界求解器的残差证明模型全局不可能，也不把成功推广任意MMD。

两侧实际安全/连续性/几何/配对门槛通过，无failed；原生末保持通过。补丁结果与检查点相同，源腕/肘仍未通过，末保持残差仍保留。本轮不改补丁或把原生进展推广成双臂接入成功。画面自然度/支撑感、手掌与袖口、源表达保留仍待人工，不用数值替代。

## 验证与观看

新增固定节点边界回归：不喂给控制器参数答案，要求腕/掌目标正确利用剩余轴，肘冲突仍报告，执行器不越盒。完整 **95项测试实跑通过**，diff检查通过；不重复宣称检查点94即当前结果。

- `comparison/comparison.mp4`：同源原时钟1.8秒，左检查点、右新版，绿色请求/橙色实际；原生限制原因全部显示。
- `bounded-native-final/sdk/stateful-arms-raw.mp4`：完整4.8秒无叠加原图。
- 同目录 `stateful-arms-review-fit.mp4`：请求、参考、最终节点、原图、阶段与限制。
- `comparison/comparison.json`：同时记录两侧各部件误差，明确联合姿态仍未通过；首版失败/余项不覆盖。

复现（新输出目录，仓库根目录运行）：

```powershell
python -m unittest discover -s tests -p "test_*.py"
python tools/control/mmd_timed_controller.py <new-root> E:\FuXuanLab\runs\20261112-mmd-timed-controller-01\source-map.json
python tools/control/assess_stateful_arm.py <new-root>\sdk
python tools/control/review_stateful_arm.py <new-root>\sdk --fit-requests
python tools/control/review_stateful_arm.py <new-root>\sdk --raw
python tools/control/compare_mmd_controllers.py E:\FuXuanLab\runs\20261112-native-target-control-01\mmd-five-final\sdk <new-root>\sdk <new-comparison-dir>
```

下一步问题已经收敛到肘目标/弯曲分支与当前模型投影约束的兼容，以及补丁臂的位置执行。当前不能称MMD接入完全通过，不继续以“模型抬不到”解释求解器遗漏的能力。
