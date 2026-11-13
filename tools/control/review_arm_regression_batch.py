"""Unified full-motion review; source boundaries are explicitly separated."""
import sys,json,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.control.review_stateful_arm import raw_video,run as review

def run(root):
    status=json.loads((root/'batch-status.json').read_text())
    page='<!doctype html><meta charset="utf-8"><title>手臂跨动作评审</title><style>body{background:#202733;color:#eee;font:18px sans-serif;margin:24px}video{width:100%;max-width:1100px}section{padding-bottom:36px}a{color:#8cf}</style><h1>当前控制器：四条动作扩展测试</h1><p>同一控制器及设置，源动作不重定时。每条独立会话：进入→源动作→返回→保持。视频之间不是连续动作；完整失败保留。只做执行测试，没有新增训练。</p>'
    for item in status:
        name=item['case'];page+=f'<section><h2>{name}</h2>'
        if item['capture']!='completed':page+='<p>采集失败，见 batch-status.json。</p></section>';continue
        sdk=root/name/'sdk'
        if not (sdk/'stateful-arms-raw.mp4').exists():raw_video(sdk)
        if not (sdk/'stateful-arms-review-fit.mp4').exists():review(sdk,fit_requests=True)
        page+=f'<video controls preload="metadata" src="{name}/sdk/stateful-arms-raw.mp4"></video><p><a href="{name}/sdk/stateful-arms-review-fit.mp4">请求/可执行预测/实际节点对照</a></p></section>'
    page+='<p>请按动作和时间指出：跳变、卡阻、松垂、形变、遮挡或动作含义失配。数值安全通过不代表自然度通过；未知/失败不删除。</p>'
    (root/'review.html').write_text(page,encoding='utf-8');print(root/'review.html')

if __name__=='__main__':run(Path(sys.argv[1]))
