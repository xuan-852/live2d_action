# 当前控制器跨MMD动作测试（运行中）

用户审看掌面候选后认为“还不错”，要求继续测试更多手臂动作。本轮复用现有控制器不改算法/预算/几何守卫，不逐片调参。关联K01/K02/K03/K07/K08/K09。

输出`E:/FuXuanLab/runs/20261113-mmd-arm-regression-01/`。四条既有MMD：yayoi-right-hand-rise1、makoto-guard1、yukiho-mouth1、azusa-bow1，和已有hair基线相比增加动作覆盖，仍非新语义留出。canonical-reference.json按文件名及sourceSha匹配唯一clip，提取已有FK点，调用bases_from_reference逐帧验证XY/掌长/掌向/normal/时钟，四条验证均通过后才进入采集。没有按图猜掌面或单独选择剪切分支。

保存当前Python控制器快照与源请求/来源证据；使用native-five、patch-projection、patch-rooted、matched-FK-palm-reference。源时间原30Hz不重定时，统一30帧进入、源样本、30返回、30保持。未做网络训练；头身/素材缺陷和不可达掌面仍按原协议报告。不同片段独立SDK，不把评审视频间切换当连续运动。

128项单测本轮实际重跑通过。批处理已启动，最终采集状态/节点安全/视觉结果待完成；逐案例异常写batch-status并保留失败，不删除。统一review.html由review_arm_regression_batch.py生成原图及fit诊断视频，待实际完成后打开。

用户“还不错”限上一轮展示版本的整体观感，不自动转移新四条；新结果不得预标自然度通过。无安装/下载/生产接入/自动提交推送。

## 实际完成与异常

四条全部Unity采集完成，共464对PNG/JSON（含各初始化帧），原图及fit诊断已生成，review.html已在Review打开。review-summary.json从原results汇总，不修改原评价。所有记录failedFrames=0、翻转0，但不能概括为连续性/自然度通过。

源阶段按当前reached判据：举手原生6/21、补丁0/21；防护12/25、1/25；近嘴12/28、0/28；鞠躬13/26、10/26。reached是现有混合判据，不等同纯腕位或自然度通过。最终中性腕误差两侧均较小，但部分最终保持窗口仍有limited，未认证全窗口收稳。

特别异常：举手最大原生参数单帧变化约59.98，近嘴约59.27；评估器仍未报failed，需要定位具体通道/参数换支及实际节点运动。不得以failed=0称动作不跳，也不得未经核查把参数整圈变化直接认定可见跳变。当前不改求解器/阈值，不重录覆盖，先保留本批用户视觉评审，再复用track定位。
