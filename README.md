# Live2D Action Lab

符玄 Live2D 动作探索独立仓库。

## 当前状态

已建立独立实验环境：可从原始 `model3.json` 加载模型、按固定 1/30 秒动作时间采样并保留无损图像与同帧状态，不依赖桌宠主项目、旧 prefab 或旧动作执行器。

本轮（2026-11-07）完成了整体表演适配批次：用同一套方法处理 8 条新轨迹，基线与候选各一套，结果与边界见 `docs/EXPRESSION-BATCH.md`。

2026-11-08 已记录用户对该批次“全都不像人做出来的动作”的反馈；旧7/8局部指标pass不构成表达或观看质量通过，诊断纠正见 `docs/EXPRESSION-BATCH.md` 顶部。画面左侧手臂当前缺少可用控制，需要研究补丁实现、绑定或源资产的改进。

## 探索范围

- 模型能力测量、动作生成与基于画面关系的重定向。
- 原生手臂控制，以及用户补充的手臂/手部网格的持续改进。
- 肩部连接、上臂/前臂/手的协调、遮挡层序、变形质量与可达范围。
- 确定动作时间步的渲染、图像与状态配对、人工视觉评审。

不包含聊天、记忆、工具系统、桌面窗口、安装器或生产接入。暂不在旧主项目配置 submodule。

## 目录

| 路径 | 用途 |
|---|---|
| `docs/BOUNDARIES.md` | 工作边界与旧实现准入原则 |
| `docs/PLAN.md` | 分阶段启动计划与验收条件 |
| `docs/EXPRESSION-BATCH.md` | 整体表演适配批次：协议、结果、边界、复跑命令 |
| `docs/EXPERIENCE.md` | 精简研究经验入口：当前结论、常见反例与按需证据索引 |
| `runtime/` | 候选运行时，尚未实现 |
| `preview/` | 独立预览与采集入口（`preview/README.md` 记录实际状态） |
| `tools/expression/` | 参数普查、手臂标定、两套适配、度量与审看页 |
| `tests/` | 自动化测试与小型固定样例 |
| `assets/` | 经许可与来源检查后引入的资源 |

大规模录制、构建与中间产物放在仓库外。资源许可尚未核查，因此本次不发布模型、贴图或 SDK，也不默认给本仓库添加开源许可证。

## 开发入口

先读 `AGENTS.md`、`docs/EXPERIENCE.md` 与当前任务；`docs/HANDOFF.md` 提供资源定位。相关案例按需查 `docs/experience/ledger.json`，无需加载完整旧项目历史。

本机已准备被 Git 忽略的模型副本 `local-assets/fuxuan/`（含两套来源副本与逐文件校验清单）。这些资源没有发布到 GitHub；详见交接入口。

当前可运行的验证：

```powershell
python -m unittest discover -s tests -p "test_*.py"           # 骨架、审计、采集与适配机制
python tools/audit_model.py local-assets/fuxuan               # 资源哈希与 model3 引用
```

测试覆盖机制正确性（配对、度量、物理覆盖诚实性、失败检出），不证明动作自然或表达已获认可。采集与批次命令见 `preview/README.md` 与 `docs/EXPRESSION-BATCH.md`。
