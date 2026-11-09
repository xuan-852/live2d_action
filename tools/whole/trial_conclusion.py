"""Summarise proxy results without promoting them to semantic/human acceptance."""
import json,sys,datetime
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.whole.batch import read

def build(root):
    index=read(root/'index.json');p=index['protocol'];lines=[]
    for c,d in index['results'].items():
        a,b=d['A'],d['B']
        lines.append('| '+ ' | '.join([c,d['split'],f"{a['meanWristErrorShoulderWidths']:.3f} → {b['meanWristErrorShoulderWidths']:.3f}",f"{a['spreadErrorShoulderWidths']:.3f} → {b['spreadErrorShoulderWidths']:.3f}",f"{a['maxFourPointStepShoulderWidths']:.3f} → {b['maxFourPointStepShoulderWidths']:.3f}",f"{a['wristRangeSum']:.3f} → {b['wristRangeSum']:.3f}"])+ ' |')
    resource=read(root/'resources.json') if (root/'resources.json').exists() else []
    if isinstance(resource,dict):resource=[resource]
    summary={'samples':len(resource),'peakOwnedWorkingSetBytes':max((s.get('workingSetBytes') or 0 for s in resource),default=0),
        'gpuScope':'whole device; includes unrelated work; not per-process VRAM',
        'cpuScope':'sampled owned processes cumulative CPU at each observation; exited child totals not accumulated',
        'wallStage':read(root/'stage.json'),'humanAcceptance':None}
    gpu=[]
    for s in resource:
        try:gpu.append([float(x.strip()) for x in s['gpuWholeDevice'].split('|')[0].split(',')])
        except (KeyError,ValueError):pass
    summary['peakWholeDeviceVRAMMiB']=max((v[0] for v in gpu),default=None)
    summary['peakWholeDeviceGPUPercent']=max((v[1] for v in gpu),default=None)
    summary['peakObservedOwnedCumulativeCPUSeconds']=max((s.get('cpuSeconds') or 0 for s in resource),default=0)
    index['resources']=summary
    (root/'index.json').write_text(json.dumps(index,indent=2),encoding='utf-8')
    page='''# 整体表达适配 A/B：集中评审

**当前结论：位置关系假说的实用收益须结合下面整段视频判断，自动指标不作语义或自然度通过。**

观看入口：[离线对照页](review/index.html) · [全部连续视频](review/all.mp4)。三列为同相位源 / A / B，每列正常512px，全屏可放大。原8条全部回归，新4条为本轮未调方法留出；旧novel-12不再称未见。

B机制：身体坐标中的双肘/双腕绝对目标与双腕相对向量，联合求解原生Core与补丁近似链；A为当前恢复轨迹。相机、资源、物理、时长与共同可见性层级一致。没有通过延长保持改善平滑，也没有逐案例手工修改。

| 样本 | 划分 | 腕位置误差 A→B（肩宽） | 展开误差 A→B（肩宽） | 最大步长 A→B（肩宽/帧） | 双腕范围总量 A→B |
|---|---|---|---|---|---|
'''+ '\n'.join(lines)+'''

位置/展开误差只比较材料代理与源身体关系，不能证明整体表达；范围总量也不是“动作越大越好”。最大步长包含首尾收势，失败帧全部保留。没有隐藏图像、缩小显示尺度或补重复帧。

扩展成本：12条自动输入处理，0逐条轨迹编辑，0动作名称/案例分支，0共享机制修订；不含本轮一次性方法研发与输入审计成本，也不证明未来所有输入零处理。A本身也自动运行，因此不能声称已测得人工成本下降；目前只证明B未新增逐条编辑。

执行边界：共同前景诊断层级保证被衣服盖住的手/前臂有机会呈现，不提供真实动态遮挡。肩口、掌面、手指、头部源朝向与完整中断/失败恢复仍有限制。实际body/physics和补丁近似链差异可能使目标求解收益无法进入最终画面。GPU为整机读数，非本任务专属显存。没有低端机认证。

输入、哈希、划分、计时、原图状态和失败记录见[index.json](index.json)、[protocol.json](protocol.json)、[freeze.json](freeze.json)、[resources.json](resources.json)。没有人工认可标签。
'''
    (root/'CONCLUSION.md').write_text(page,encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=True,indent=2))

if __name__=='__main__':build(Path(sys.argv[1]))
