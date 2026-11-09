# 新对话：独立动作探索入口

## 当前入口（2026-11-08）

先读 [研究经验摘要](EXPERIENCE.md)，再按本次问题查对应经验编号。当前预览与采集状态见 [preview/README.md](../preview/README.md)，最近批次及其失败纠正见 [EXPRESSION-BATCH.md](EXPRESSION-BATCH.md)。这三者优先于下方初始建仓说明；下方“尚未建立工程/SDK”“首个任务建议”等属于历史快照，不应据此重复搭建环境。

当前批次已记录用户“全都不像人做出来的动作”的反馈，旧7/8局部指标pass不代表动作质量通过。具体采用的旧经验、已经否定的解释和新版本证据，应随任务一并交接，不依赖旧聊天上下文。

## 用户分工

本仓库的新对话负责 Live2D 动作研究；旧桌宠项目的瘦身在原对话单独讨论。不要为了动作实验回到主项目叠加补丁。

先读本文件、`AGENTS.md`、`docs/BOUNDARIES.md`、`docs/PLAN.md`，再读取当前任务直接需要的文件。

## 已准备

- 远端：`https://github.com/xuan-852/live2d_action`。
- 本地：`E:/FuXuanLab/repos/live2d_action`。
- 本地模型副本：`local-assets/fuxuan/`，被 Git 忽略，不上传。
- `local-assets/fuxuan/manifest.json` 保存逐文件 SHA-256、尺寸与来源。
- 两套副本分别为 `unity-model/` 与 `streaming-model/`，共 74 个文件，复制后逐文件哈希一致。
- 保留两套是因为旧工程两处 prefab 存在差异；不得未经核对随意选一套覆盖另一套。
- 尚未建立独立 Unity 工程，尚未引入 Cubism SDK 或补丁执行器，基础测试不代表模型已运行。

## 资源安全

随模型使用说明包含禁止二次分发等限制。原文件已随本地副本保留，禁止把模型、贴图、Unity 导入资源或第三方 SDK 自动提交到远端。新环境缺少 `local-assets` 时，应明确报告资源缺失，不默认从网络补齐。

Unity prefab、controller 等含原工程 GUID 引用。本次仅保证复制完整，不保证这些导入资源已能在新工程直接使用；需要正确版本的 SDK、依赖解析与实际加载验证。优先核对原始 model3 清单与资源引用。

## 补丁网格迭代

用户要求补丁手/臂网格本身也持续改进，不把当前变形质量当成固定上限。模型副本不包含旧工程外部 C# 补丁执行器；需要时按准入规则有选择地引入最小实现。

旧入口只作为定位线索，不默认加载整个目录：

- `Assets/Scripts/Embodied/SecondaryArmPatch.cs`：补丁变形与执行。
- `Assets/Scripts/Embodied/ShoulderConnection.cs`：肩根连接（如原工程存在）。
- `Assets/Scripts/Embodied/SecondaryArmMotion.cs`：动作契约。
- `Assets/Scripts/Live2DProbe/SecondaryArmPatchPreview.cs`：旧预览与采集。

这些路径位于旧工程 `D:/Unity/projects/Desktop_per_pro/code/desktop_unity/` 下，不是本仓库运行依赖。旧实现未经本仓库复验，不能直接继承其可信状态。

## 首个任务建议

先建立可独立加载模型的最小预览和可靠采样，不继续调旧动作。明确参数写入者和物理状态；固定动作时间采样、保留无损图像与同帧状态。确认该基础成立后，再复现补丁网格基线与改进其连接、绑定、拓扑和层序。

## 旧结论的证据限制

- 旧版动作驱动把复杂动作压成活动量标量，不能作为姿态关系重定向成果。
- 旧录制采样不均匀；固定 fps 重编码或 VFR 重封装都不会创造缺失中间帧。
- 旧状态只记录部分画件，不能用它证明全模型纹理稳定。
- 少量顶点距离恒定不能证明渲染出的整只手没有问题；参数连续也不能证明实际画面连续。
- 物理 enabled 不代表有足够可见的物理表现；不能只凭像素差判断重绘、压缩或缺少部件。
- 本仓库尚无已验收的新动作，不得自动宣称已有动作质量通过人工认可。
