"""Same requests and action clock: compare SDK speed budgets without retiming."""
import json
import sys
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import imageio_ffmpeg
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.control.stateful_arm import wrap
from tools.expression.unity import write_spec


def run(root, point=False, controllers=False, timed_name='timed'):
    names = ('legacy', 'normalized', timed_name) if controllers else ('point-baseline', 'point-fast') if point else ('control-baseline', 'responsive-r2', 'fast')
    controller_base = 'controller-comparison' if timed_name == 'timed' else 'controller-comparison-final'
    output = root/((controller_base+'.mp4') if controllers else 'point-comparison.mp4' if point else 'speed-comparison.mp4')
    report = root/((controller_base+'.json') if controllers else 'point-comparison.json' if point else 'comparison.json')
    if output.exists() or report.exists():
        raise FileExistsError(output)
    tracks = [[json.loads(l) for l in (root/name/'track.jsonl').read_text().splitlines()] for name in names]
    specs = [json.loads((root/name/'protocol.json').read_text()) for name in names]
    if len({len(t) for t in tracks}) != 1 or not all((root/n/'capture/complete.txt').exists() for n in names):
        raise ValueError('incomplete comparative capture')
    max_node_difference = 0.; max_heading_difference = 0.
    for frames in zip(*tracks):
        if len({(r['frame'], r['active'], r['stage'], r['localFrame'], r['time']) for r in frames}) != 1:
            raise ValueError('request/action time pairing mismatch')
        side = frames[0]['active']
        base = np.array(frames[0][side]['requested'])
        for row in frames[1:]:
            target = np.array(row[side]['requested'])
            max_node_difference = max(max_node_difference, float(np.max(abs(target[1:3]-base[1:3]))))
            heading = lambda p: np.degrees(np.arctan2(*(p[3]-p[2])[::-1]))
            max_heading_difference = max(max_heading_difference, float(abs(wrap(heading(target)-heading(base)))))
            if abs(row[side]['deadline']-frames[0][side]['deadline']) > 1e-8:
                raise ValueError('retimed deadline')
    if max_node_difference > 1e-6 or max_heading_difference > .001:
        raise ValueError('spatial reference changed')
    summaries = []
    for name, track, spec in zip(names, tracks, specs):
        if controllers:
            bootstrap = json.loads((root/name/'capture/0000.json').read_text(encoding='utf-8-sig'))
            previous = {side: np.array(bootstrap['wholeArms'][side+'Chain']).reshape(4, 2) for side in ('native', 'patch')}
            for row in track:
                for side in previous:
                    actual = np.array(row[side]['actual']); prior = previous[side]
                    v = np.max(np.linalg.norm(actual-prior, axis=1))*30
                    angle = lambda x: np.degrees(np.arctan2(*(x[3]-x[2])[::-1]))
                    rate = abs(float(wrap(angle(actual)-angle(prior))))*30
                    row[side]['comparisonSettled'] = bool(v <= .02 and rate <= 8.)
                    previous[side] = actual
        for side in ('native', 'patch'):
            stage = 'point-reach-return' if point else 'moving-target-switch'
            a = [r for r in track if r['active'] == side and r['stage'] == stage]
            reached = lambda s: s['wristError'] <= .005 and s['elbowError'] <= .005 and s['palmErrorDeg'] <= 4 and (not controllers or s['comparisonSettled'])
            arrivals = []
            for begin, end in (((0, 54), (54, 108)) if point else ((0, 21), (21, 42), (42, 63), (63, 90))):
                phase = [r for r in a if begin <= r['localFrame'] < end]
                found = next((j for j in range(len(phase)-2) if all(reached(r[side]) for r in phase[j:j+3])), None)
                arrivals.append(None if found is None else (phase[found]['localFrame']-begin)/30)
            full = [r for r in track if r['active'] == side]
            summaries.append(dict(profile=name, side=side, switchWristRMS=float(np.sqrt(np.mean([r[side]['wristError']**2 for r in a]))),
                switchElbowRMS=float(np.sqrt(np.mean([r[side]['elbowError']**2 for r in a]))),
                switchPalmRMSDeg=float(np.sqrt(np.mean([r[side]['palmErrorDeg']**2 for r in a]))),
                deadlineMissedFrames=sum('deadline-missed' in r[side]['reasons'] for r in a),
                firstStableArrivalSeconds=arrivals,
                commonDeadlineMissedFrames=sum(r[side]['now'] >= r[side]['deadline'] and not reached(r[side]) for r in a),
                failedFrames=sum(r[side]['status'] == 'failed' for r in full),
                maxNodeStep=max(r[side]['metrics']['maxNodeStep'] for r in full),
                maxNodeAcceleration=max(r[side]['metrics']['maxNodeAcceleration'] for r in full),
                minAreaRatio=min(r['geometry']['minTriangleAreaRatio'] for r in full),
                maxFlips=max(r['geometry']['flippedTriangles'] for r in full),
                maxPairing=max(r['pairing'] for r in full), maxRestore=max(r['geometry']['restoredMeshError'] for r in full),
                limits=spec['limits'], visualAcceptance='pending for new speed profiles'))
            if controllers:
                # Diagnostic angles only: never a universal force/naturalness score.
                f0 = json.loads((root/name/'capture'/f"{a[0]['frame']-1:04}.json").read_text(encoding='utf-8-sig'))
                home = np.array(f0['wholeArms'][side+'Chain']).reshape(4, 2)
                def chain_angles(nodes):
                    d = np.diff(nodes, axis=-2)
                    h = np.degrees(np.arctan2(d[..., 1], d[..., 0]))
                    return h, wrap(np.diff(h, axis=-1))
                home_h, home_j = chain_angles(home)
                actual_h, actual_j = chain_angles(np.array([r[side]['actual'] for r in a[:54]]))
                change = wrap(actual_j-home_j)
                summaries[-1].update(liftElbowDeltaMin=float(change[:, 0].min()), liftElbowDeltaMax=float(change[:, 0].max()),
                    liftWristDeltaMin=float(change[:, 1].min()), liftWristDeltaMax=float(change[:, 1].max()),
                    upperHeadingAtLocal6=float(wrap(actual_h[6, 0]-home_h[0])),
                    palmHeadingAtLocal6=float(wrap(actual_h[6, 2]-home_h[2])))
            if point:
                for key in ('WristRMS', 'ElbowRMS', 'PalmRMSDeg'):
                    summaries[-1]['point'+key] = summaries[-1].pop('switch'+key)
    write_spec(report, dict(maxReferenceNodeDifference=max_node_difference,
        maxReferenceHeadingDifferenceDeg=max_heading_difference, sameDeadlines=True, summaries=summaries))
    width = 512*len(names)
    command = [imageio_ffmpeg.get_ffmpeg_exe(), '-loglevel', 'error', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', f'{width}x650', '-r', '30', '-i', '-', '-an', '-c:v', 'libx264',
        '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)]
    with subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        for frames in zip(*tracks):
            r = frames[0]; side = r['active']; canvas = Image.new('RGB', (width, 650), '#eeeeee'); draw = ImageDraw.Draw(canvas)
            draw.text((8, 8), f"SAME REQUESTS / DEADLINES / 30 Hz ACTION CLOCK | {r['time']:.3f}s | {side} | {r['stage']}", fill='black')
            draw.text((8, 29), 'Green=request elbow/wrist; orange=actual. Full requests retained in individual videos. All failures retained.', fill='black')
            for col, (name, frame) in enumerate(zip(names, frames)):
                s = frame[side]; capture = root/name/'capture'
                with Image.open(capture/f"{frame['frame']:04}.png") as im:
                    canvas.paste(im.convert('RGB').resize((512, 512)), (col*512, 104))
                draw.text((col*512+8, 53), f"{name} | {s['status']} | wrist {s['wristError']:.4f} | palm {s['palmErrorDeg']:.1f}deg", fill='black')
                draw.text((col*512+8, 76), f"actual node speed {s['metrics']['maxNodeSpeed']:.3f}/s | node step {s['metrics']['maxNodeStep']:.4f}", fill='black')
                f = json.loads((capture/f"{frame['frame']:04}.json").read_text(encoding='utf-8-sig'))
                actual = np.array(s['actual']); screen = np.array(f[side+'NodeScreen']).reshape(4, 2)
                affine = np.linalg.lstsq(np.c_[actual, np.ones(4)], screen, rcond=None)[0]
                requested = np.c_[s['requested'], np.ones(4)] @ affine
                for j in (1, 2):
                    for xy, color in ((requested[j], '#00ef88'), (screen[j], '#ff9b30')):
                        x, y = xy/2
                        if 5 <= x <= 507 and 5 <= y <= 507:
                            x += col*512; y += 104
                            draw.ellipse((x-4, y-4, x+4, y+4), outline=color, width=2)
                draw.text((col*512+8, 625), ', '.join(s['reasons'])[:76] or 'within current checks', fill='#8c2424' if s['reasons'] else '#006b3c')
            proc.stdin.write(canvas.tobytes())
        proc.stdin.close(); error = proc.stderr.read(); code = proc.wait()
        if code:
            raise RuntimeError(error.decode(errors='replace'))
    print(output)
    for s in summaries:
        print(s['profile'], s['side'], 'RMS', round(s['pointWristRMS' if point else 'switchWristRMS'], 5), 'deadline misses', s['deadlineMissedFrames'], 'failed', s['failedFrames'])


if __name__ == '__main__':
    run(Path(sys.argv[1]), point='--point' in sys.argv or '--controllers' in sys.argv, controllers='--controllers' in sys.argv,
        timed_name=sys.argv[sys.argv.index('--timed-dir')+1] if '--timed-dir' in sys.argv else 'timed')
