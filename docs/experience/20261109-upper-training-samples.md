# 上层候选训练样本包：任务来源与节点序列配对

用户授权先准备任务/完整节点/质量反馈样本包，不继续盲目训练。关联K03/K04/K11/K12。输出`E:/FuXuanLab/runs/20261109-upper-training-samples-01/`；55项测试通过。

已查回原生成目录requests.json与requests-reserve.json，19份参考均以caption/id及seed双重匹配原请求；不是根据画面倒编任务。**19份的原请求完全相同**：A person stands in place, talking and making varied natural arm gestures. 因此没有19种语义任务，不能由novel编号编出介绍/拒绝等意图。请求来源哈希、原参考文件SHA及副本保留。

每样本含sample.json（任务/时长/缺失/状态）、完整source-nodes.json、model-arm-nodes.json、原参考副本、review.json。源序列保留完整时长，30Hz均匀采样到原始末尾前最后网格点，未统一截6秒。模型尺度来自当前双臂接口，保留两侧身份、肩肘腕掌顺序；以投影方向相对首帧变化映射到模型链，肩根固定、掌相对中性，头身朝向/深度没有伪造。映射仍是候选，可能损失源动作内容，不能等同完整身体示范。

预定编号01/03/08生成审看视频，左源正/侧骨架，右当前模型尺度双臂节点；novel-01已打开Review。明确标参考/映射，非新训练成果。REVIEW.md要求先独立描述实际动作，再给原任务，分别评价源质量与映射损失，标时间段/不确定性。review.json空白待审核，**trainingEligible全部false、split暂不分配**，没有将未看动作默认为自然示范。

本轮尚未调用外部视觉模型，尚未收集真实评审；缺多样任务与合格示范/反馈仍是训练准入问题。下一步实际评审三条视频并填写真实证据，区分accept/reject/uncertain；需要多任务条件生成时另获取真实多任务示范，而不是重命名现有同任务样本。不能再把候选包完成称为整体动作训练成功。无提交推送。
