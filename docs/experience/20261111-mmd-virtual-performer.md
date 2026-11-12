# MMD虚拟表演输入：头身首测

用户授权用MMD替代真人跟踪输入检查直播皮套式效果。复用`20261029-action-expansion-owner-03/solve/makoto-hair/map.json`已有body/headLocal欧拉角及原动作时间，未再解析VMD或搬旧控制系统。原VMD/source-map/MOC哈希在protocol。来源旧FK本轮未重新认证。

证据`E:/FuXuanLab/runs/20261111-mmd-virtual-performer-01/`。54帧/1.7667秒，30Hz；head-only物理关与head-body-physics物理开各一条，共108对PNG/JSON。用headLocal相对首帧变化映射AngleXYZ，body相对首帧变化/3映射BodyAngleXYZ，候选Y/Z反号；不把headWorld加到body上重复旋转。初帧相对化是有损校准，不保证保留源绝对姿态。没有手臂动作、平移、morph或口型，不伪造表情来源。

56项Python测试通过，两个Unity任务编译及采集成功，网格pairing均0。物理版实际X2约[-7.577,7.476]、Y2[-3.645,0]、Z2[-1.898,.628]，头部单驱动版输出均0。抽查.6秒可见头、躯干和附属物不同，证明链路有响应，不证明自然或忠实源动作。

重要限制：头pitch16/54帧、roll7/54帧、body yaw31/54帧被限幅，当前映射明显有损，不能宣称保留原MMD幅度/节奏。对照同时改变身体输入与物理开关，不能因果分离身体与物理贡献；还缺同步源表演/朝向示意，不是完整三列源到目标验证。

`mmd-virtual-performer-review.mp4`已打开Review，固定局部裁切、原始30Hz播放，短片重复3次，每次标明CUT，不把剪辑重复当连续灵动。尚未经用户人工认可。下一步根据观感再明确初始灵动目标；若继续验证，使用输入范围校准减少截断，并拆开身体/物理条件、补源朝向对照，优先复用旧记录而非全量重扫。无训练、付费服务、安装、生产接入、提交推送。
