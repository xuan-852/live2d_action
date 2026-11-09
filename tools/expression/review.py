"""Build an offline review page for the batch: source, baseline and candidate side by side."""
import html
import json
from pathlib import Path

STYLE = """
body{font-family:system-ui,'Segoe UI',sans-serif;background:#14161b;color:#e8e9ee;margin:0;padding:24px}
h1{font-size:20px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 8px}
.lead{color:#a9adbb;font-size:13px;max-width:980px;line-height:1.5}
table{border-collapse:collapse;margin:12px 0;font-size:13px}
th,td{border-bottom:1px solid #2c303a;padding:5px 10px;text-align:left}
.pass{color:#6ee7a8}.fail{color:#ff8a8a}.nomotion{color:#f5c26b}
.case{display:grid;grid-template-columns:520px 1fr;gap:18px;margin:22px 0;padding:16px;background:#1b1e25;border-radius:10px}
video{width:100%;border-radius:6px;background:#000}
.lab{font-size:12px;color:#a9adbb;margin:0 0 6px}
.notes{width:100%;min-height:70px;background:#12141a;color:#e8e9ee;border:1px solid #2c303a;border-radius:6px;padding:8px}
button{background:#2f6df6;border:0;color:#fff;border-radius:6px;padding:8px 14px;cursor:pointer}
code{background:#12141a;padding:1px 4px;border-radius:4px;font-size:12px}
"""

SCRIPT = """
function save(id){const el=document.getElementById('n-'+id);localStorage.setItem('lab-note-'+id,el.value);
document.getElementById('s-'+id).textContent='saved';}
function load(){document.querySelectorAll('textarea.notes').forEach(el=>{const v=localStorage.getItem('lab-note-'+el.dataset.id);if(v)el.value=v;});}
function copyAll(){const lines=[];document.querySelectorAll('textarea.notes').forEach(el=>{
lines.push(el.dataset.id+': '+(el.value||'(no note)'));});navigator.clipboard.writeText(lines.join('\\n'));
document.getElementById('copied').textContent='copied '+lines.length+' notes';}
window.addEventListener('DOMContentLoaded',load);
"""


def row(name, entry):
    def cell(method):
        comparison = entry[method]['comparison']
        status = comparison.get('status', 'n/a')
        css = {'pass': 'pass', 'fail': 'fail', 'no-motion': 'nomotion'}.get(status, '')
        lateral = comparison.get('lateralCorrelation')
        ratio = comparison.get('peakRatio')
        return (f"<td class='{css}'>{status}</td>"
                f"<td>{'' if lateral is None else round(lateral, 2)}</td>"
                f"<td>{'' if ratio is None else round(ratio, 2)}</td>")
    return (f"<tr><td>{html.escape(name)}</td><td>{entry['split']}</td><td>{entry['klass']}</td>"
            f"{cell('baseline')}{cell('candidate')}</tr>")


def build(run: Path, results: dict, summary: dict):
    check_videos(run, results)
    cases = ''.join(
        f"""<div class="case">
  <div><video controls loop muted playsinline preload="metadata" src="../cases/{html.escape(name)}/compare.mp4"></video>
  <p class="lab">left: source skeleton (front view, mirrored) · middle: baseline · right: candidate<br>
  <a href="../cases/{html.escape(name)}/compare.mp4">直接用播放器打开这条 mp4</a></p></div>
  <div>
    <h2 style="margin-top:0">{html.escape(name)} <span class="lab">({entry['split']}, source class {entry['klass']})</span></h2>
    <p class="lab">baseline: <b>{entry['baseline']['comparison'].get('status')}</b>
      · candidate: <b>{entry['candidate']['comparison'].get('status')}</b>
      · {entry['candidate']['frames']} frames @30 Hz · pairing Δ={entry['candidate']['pairingMaxDelta']}</p>
    <p class="lab">lateral corr {_fmt(entry['candidate']['comparison'].get('lateralCorrelation'))} /
      height corr {_fmt(entry['candidate']['comparison'].get('heightCorrelation'))} /
      opening ratio {_fmt(entry['candidate']['comparison'].get('peakRatio'))}
      · {html.escape(', '.join(entry['candidate']['comparison'].get('reasons') or ['no failing reason']))}</p>
    <textarea class="notes" data-id="{html.escape(name)}" id="n-{html.escape(name)}"
      placeholder="记录这条案例：读得出原意吗？哪一帧开始变味？"></textarea>
    <button onclick="save('{html.escape(name)}')">保存标注</button>
    <span class="lab" id="s-{html.escape(name)}"></span>
  </div>
</div>""" for name, entry in sorted(results.items()))

    body = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>符玄整体表演适配 · 本轮审看</title><style>{STYLE}</style></head><body>
<h1>符玄整体表演适配 · 本轮审看</h1>
<p class="lead">同一套方法处理多条新轨迹：左为源骨架（正面、镜像呈现），中为按参数名称直接映射的基线，
右为按身体关系＋实测可达网格求解的候选。三条视频均来自本轮真实采集，30 Hz 原图保留在
<code>cases/&lt;id&gt;/</code>；浏览器播放的 mp4 只是观看副本。顺序对齐同一动作时间，请按需暂停/放大。</p>
<h2>汇总</h2>
<p class="lead">整批单文件观看：<a href="all-cases.mp4">all-cases.mp4</a>（8 条依次排列）·
整批静帧：<a href="all-cases-frames.png">all-cases-frames.png</a>（每条取开口最大的候选帧）。</p>
<table><thead><tr><th>案例</th><th>划分</th><th>源类别</th>
<th>基线状态</th><th>基线横向相关</th><th>基线开口比</th>
<th>候选状态</th><th>候选横向相关</th><th>候选开口比</th></tr></thead>
<tbody>{''.join(row(name, entry) for name, entry in sorted(results.items()))}</tbody></table>
<p class="lead">基线通过 {summary['baseline']['passed']} / {summary['baseline']['cases']}，
候选通过 {summary['candidate']['passed']} / {summary['candidate']['cases']}；
跨案例“源开口量 vs 渲染开口量”秩相关：基线 {_fmt(summary['baseline']['peakSpearman'])}，
候选 {_fmt(summary['candidate']['peakSpearman'])}。状态只描述机械关系是否保留，不代表自然度或语义已获人工认可。</p>
<h2>逐条观看</h2>
{cases}
<p><button onclick="copyAll()">复制全部标注</button> <span class="lab" id="copied"></span></p>
<p class="lab">阈值在采集前冻结：横向相关 ≥ 0.5，开口比 0.5–2.0，可见位移 ≥ 0.05 归一化单位；
位移不足记为 no-motion，不算通过。标注只写入本机 localStorage，不上传。</p>
</body><script>{SCRIPT}</script></html>"""
    output = run / 'review' / 'index.html'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(body, encoding='utf-8')
    return output


def _fmt(value):
    return 'n/a' if value is None else f'{value:.2f}'


def check_videos(run: Path, results: dict):
    """Fail loudly instead of shipping a review page whose media cannot load."""
    missing = []
    for name in results:
        path = run / 'cases' / name / 'compare.mp4'
        if not path.is_file() or path.stat().st_size < 10000:
            missing.append(str(path))
    if missing:
        raise FileNotFoundError('missing or empty comparison videos: ' + ', '.join(missing))
    return len(results)


if __name__ == '__main__':
    import sys
    run = Path(sys.argv[1])
    results = json.loads((run / 'results.json').read_text(encoding='utf-8'))
    summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
    print(build(run, results, summary))
