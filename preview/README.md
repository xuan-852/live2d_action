# Preview

2026-11-08 整体能力恢复：`WholeArmSession.cs` 接入最小提取的补丁臂/肩根连接，原8案例双臂整体对照见 `docs/WHOLE-CAPABILITY.md`。运行成功不代表表达改善；旧“另一侧只能静态”的沙盒缺口已补上执行层，但几何/自然度与生命周期仍待验证。

独立 Unity 预览与采集代码。实验工程和 SDK 位于被 Git 忽略的 `local-assets/preview-unity/`，不含桌宠业务。

## 当前最小入口

`unity/LabCapture.cs` 是 Editor 批处理入口，从原始 `model3.json` 加载，不使用旧 prefab。
默认输出四秒 `turn-nod-return` 候选动作：Unity AnimationClip 原生曲线采样，SDK 物理稳定化及固定 1/30 秒步进。设置 `ACTION_LAB_MODE=neutral` 可输出三帧物理关闭的中性基线。尚无交互预览，不代表第一阶段全部完成。

`unity/LabMotion.cs` 是本轮新增的批量入口，两个模式：

- `LabMotion.RunPoseLab`：读取 `ACTION_LAB_POSES`（姿态清单），逐个写入参数、关闭物理、输出 PNG + 紧凑几何（画件簇心/包围盒）。
- `LabMotion.RunMotion`：读取 `ACTION_LAB_CURVES`（30 Hz 曲线清单），用 AnimationClip 采样，物理作用范围可选，输出同规格逐帧状态。

两者每帧都写入 `pairingMaxDelta`（渲染网格与 `CubismDrawable.VertexPositions` 的逐顶点最大差）。当前 SDK 渲染器双缓冲在离线固定步进下会让 `MeshFilter` 落后一次更新，`WriteFrame` 因此多交换一次缓冲再读；本轮批次该值恒为 0。

Python 侧驱动在 `tools/expression/unity.py`（自动同步脚本、拒绝覆盖输出目录、检查完成标记）。

准备新工程（拒绝覆盖已有工程）：

```powershell
python tools/prepare_preview.py --sdk <本机已核查的 Cubism 目录>
```

当前本机选择：团结 `2022.3.62t7`（1.8.5），Cubism 组件 CHANGELOG `5-r.4`。SDK 的 Core 与 Components 适用不同许可；保留其 LICENSE/NOTICE，禁止自动发布。`sdk-provenance.json` 记录所选文件的来源和 SHA-256，不包含 Samples。现有 SDK 来自本机旧工程，只作为第三方 SDK 候选副本，不导入旧动作代码。

```powershell
$env:ACTION_LAB_MODEL = (Resolve-Path 'local-assets/fuxuan/unity-model/符玄.model3.json').Path
$env:ACTION_LAB_OUTPUT = Join-Path (Get-Location) 'captures/neutral-new'
& 'D:/Unity/editor/2022.3.62t7/Editor/Tuanjie.exe' -batchmode -projectPath "$PWD/local-assets/preview-unity" -executeMethod LabCapture.Run -logFile "$PWD/local-assets/preview-unity/capture.log"
python tools/check_capture.py captures/neutral-new
```

修改采集代码后，需显式将 `preview/unity/LabCapture.cs` 复制至实验工程 `Assets/ActionLab/Editor/LabCapture.cs`。输出目录必须不存在。完成标记仅在全部帧写出后生成；失败输出保留，不用于评审。

## 证据边界

- 每个编号对应 PNG 和 JSON；记录请求/应用参数、全体 part 的 opacity、全体 drawable 的渲染网格顶点/索引/顶点 alpha 首值及 renderer 层序/启停。
- 全体画件记录不等于全部渲染状态：当前未完整记录 UV、材质属性、遮罩缓冲和物理内部状态。
- 动作时间为 `frame / 30`；UTC 是渲染读回后的采样时间，`renderReadbackMs` 为 Camera.Render + GPU 读回耗时，不包含建模、编码、磁盘写入或总运行时间。
- 静态基线仅用于静止采集检查，不证明动作自然度或跨进程复现。`check_capture.py` 目前只适用于中性基线：动作中物理可能改变请求值，不能套用参数完全相等的断言。
- 所有自动参数写入者关闭；动作模式由 Unity AnimationClip 写入六条曲线，再由 SDK 物理、Core 和 render/mask 依次更新。元数据中的 `:disabled` 表示初始化时停用，物理及 render/mask 后续按显式记录重新启用。物理开关与重置交互入口仍待实现。
- 原始模型副本未发现 `.cmo3`/PSD/PSB 源资产；运行时网格实验与源级编辑能力必须分开讨论。
- 2026-11-07 批次发现并修正：离线下 SDK 渲染器双缓冲会让 `MeshFilter` 落后一次更新，因此旧记录（`gesture-001`、`body-*-001/002`、`coordinated-001`）的图像与参数状态存在一帧错位；这些版本的视觉结论仍成立，但不要再把它们当作逐帧配对证据。`LabMotion` 之后的输出带 `pairingMaxDelta` 自检。
- 参数普查（244 参数 × 极限姿态）显示：`Param94` 等驱动（画面右侧）手臂，`Param31-37` 对（画面左侧）手臂影响 ≤0.073 单位，`ParamBodyAngleX/Y/Z` 无效果，`Param_Angle_Rotation_*` 两链是飘带。详见 `docs/EXPRESSION-BATCH.md`。

## 首轮实际结果（2026-11-07）

- `captures/neutral-003/0000.png`：完整正面中性图像，双臂下垂；未引入旧补丁执行器。
- 三帧时间为 0、1/30、2/30 秒；244 个请求/应用参数逐帧相等，433 个画件的已记录状态逐帧相等。
- 三张 PNG 哈希不相等。Pillow 对第 0/1 帧的解码图像差分发现 1 个像素变化，位置为 x=552、y=386。差分图保留于该运行目录，原因未确定。
- 因此已验证独立加载和静态图像/状态采集入口，但未达到逐像素一致；不将细小像素差作为动作观看的阻断。
- `neutral-001`、`neutral-002` 是保留的取景/状态读取调试版本，不作为有效视觉基线。
- 后续以可观看动作优先；不能从单像素差直接推断原因。

## SDK 能力核查与短动作

现成本地 SDK 源码提供（存在接口不代表全部已运行验证）：

| 能力 | 入口 | 本轮验证范围 |
|---|---|---|
| 播放、循环、速度、优先级、回调 | `CubismMotionController` | 源码及官方示例核查，未运行控制器 |
| 多层权重、加法混合 | `SetLayerWeight` / `SetLayerAdditive` | 源码核查，未实测混合 |
| 参数淡入淡出 | `CubismFadeController` / FadeMotionList | 源码结构核查，未实测 |
| 物理加载、稳定化、固定步长 | `CubismPhysicsController` / `CubismPhysics.UseFixedDeltaTime` | 动作中调用稳定化及逐区间步进；可见物理贡献尚未做关闭对照 |
| 原生动画曲线采样 | Unity `AnimationClip.SampleAnimation` | 已用于 121 帧动作 |

官方示例已阅读：`Live2D/CubismUnityComponents` develop 的 `Assets/Live2D/Cubism/Samples/OriginalWorkflow/Motion/CubismMotionPreview.cs`，其入口直接调用 SDK `PlayAnimation`。本机 Samples 目录为空，未复制示例依赖或示例角色；网络 develop 不等同本地 5-r.4，接口以本地源码为准。

确实缺失的是本地符玄动作内容（model3 无 Motions）、面向交流意图的动作组织，以及补丁侧的已验证表达能力，不是通用播放器或物理求解器。后两项本轮未实现。

`captures/gesture-001/turn-nod-return.mp4` 为四秒候选，121 张 30Hz PNG 原图保留；GIF 为每三张取一张的 10Hz 观看副本，不声称增加中间帧。动作使用头部、躯干和视线六条曲线，双臂保持原始下垂状态；不是整体表达系统验收。未新增自定义混合器、播放器或物理求解器，仅添加动作内容和离线采样装配。画面整体变化偏轻微，尚未获得用户认可。

## 身体跟随对照

用户对 `gesture-001` 的反馈：头部动作对得上，但看不到身体跟随。此反馈只适用于该版本的头部表现，不代表整条动作或新版通过认可。

核查发现本地 5-r.4 的初始化顺序问题：`CubismPhysicsSubRig.Stabilization` 解析 Input.Source 但未设置 SourceIndex；`Evaluate` 仅在 Source 为空时设置 SourceIndex。此前先稳定化导致输入缓存索引未正确初始化，身体 X2/Z2 在整段中为零。最小修正为先调用一次 SDK OnLateUpdate，再 Stabilization，SDK 文件未改动。

- `body-on-001` / `body-off-001`：放大身体曲线但尚未修正初始化，输出 X2/Z2 仍为零，保留为失败对照。
- `body-on-002` / `body-off-002`：初始化修正后的身体开/关对照，均开启 SDK 物理，121 帧、30Hz，头部与视线请求曲线逐帧一致。设置 `ACTION_LAB_BODY=off` 可关闭两条身体曲线，默认开启。
- 身体开启版 X 峰值 7、Z 峰值 -4，均未超出模型参数范围；画面可见躯干倾斜、腰部与衣摆位置变化，未将此直接认证为自然的轴向转身。
- 第 39 帧物理输出 X2≈6.214、Z2≈3.859；支持身体物理链生效的判断，不代表所有物理子组已验证。
- `captures/body-on-002/body-follow.mp4`：完整四秒候选；`body-compare.gif`：左侧身体曲线关闭，右侧开启，10Hz 观看副本。原图和状态仍为 30Hz。
- 头部曲线未修改，但物理初始化和身体变形改变了最终画面，需要用户重新评价。动作回正后的物理余摆保留，未强制清零；结尾是否足够收稳尚待视觉评审。

用户进一步指出 `body-on-002` 是摆腰而非所需的身体朝向转动，因此它不作为转体验收成果。用户确认模型曾显示所需转体效果，但未记清具体参数。

为定位而非扩大测量，增加两个隔离模式（均物理关闭）：
- `ACTION_LAB_MODE=body-probe`：六个现有 BodyAngle X/Y/Z、X2/Y2/Z2 各取正负 80% 范围及中性，共 13 张独立姿态；不是连续动作，不应用时间轴运动评审。结果为 `captures/body-channels-001/channels.png`。
- `ACTION_LAB_MODE=body-turn`：仅 X2 一条原生曲线，0 → -8 → 0 → +8 → 0，四秒 121 帧，无头部主动曲线、无 Z 侧倾、无物理覆盖。用于确认转体候选，不作为最终协调表达。

隔离画面显示 X2 的胸饰、衣襟和腰部轮廓变化比第一组 X 清楚；是否符合用户记忆中的朝向转动仍需用户确认。近景对照 `captures/body-channels-001/body-x2-closeup.png`。不能把 X2 名字或参数变化本身当作转体证明。

用户随后明确确认 `body-turn-001` 的 X2 隔离效果就是所需转体。这是该版本、该通道效果的人工确认，不扩展为整套动作认可。

## 头先转、身体跟随候选

`ACTION_LAB_MODE=coordinated` 输出 `captures/coordinated-001/head-body-turn.mp4`，近景副本为同目录 `head-body-turn.gif`。仍为四秒 121 张 30Hz 原图，GIF 每三帧选一张为 10Hz，不补帧。

- 头部和视线沿用原曲线。X2 在 0.75 秒开始跟随，1.6 秒到 7，保持至 2.45 秒，3.45 秒回正；头部在 0.5 秒起动，3.2 秒回正。没有主动 BodyAngleX/Z 驱动，不使用摆腰替代转体。
- 冲突来源：模型 physics3 已将 X2 配置为物理输出，不能同时让物理和显式动画曲线独占写入。通过现有 SDK rig 的 Output.Weight=0，只停用匹配 X2 的一个物理输出；其他输出保留。原始模型文件、SDK 均未修改，未新增混合器。
- 运行检查：121 帧的 X2 请求/应用值一致，BodyAngleZ 全程为零，动作时间均匀。关键帧可见胸饰、衣襟随躯干转向，随后点头并回正；该视觉质量尚待用户确认。
- 物理余摆可能使结尾不等同逐像素中性；本轮未用它阻断动作观看。手臂仍下垂，不宣称双臂表达或完整意图系统完成。
