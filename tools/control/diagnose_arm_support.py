"""Read-only SDK evidence: separate endpoint success from arm-chain coherence.

No new actuator commands, physical simulation or naturalness scoring. Material
segment headings are geometric diagnostics, not certified anatomical joint angles.
"""
import json
import sys
import hashlib
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import imageio_ffmpeg
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.control.stateful_arm import wrap
from tools.expression.unity import write_spec


def orientation(nodes):
    d = np.diff(np.asarray(nodes), axis=-2)
    headings = np.degrees(np.arctan2(d[..., 1], d[..., 0]))
    return headings, wrap(np.diff(headings, axis=-1))


def analyze(source, output):
    output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((source/'protocol.json').read_text())
    rows = [json.loads(l) for l in (source/'track.jsonl').read_text().splitlines()]
    if not (source/'capture/complete.txt').exists() or len(rows)+1 != protocol['frames']:
        raise ValueError('incomplete source')
    sides = {}; summary = []
    for side in ('native', 'patch'):
        a = [r for r in rows if r['active'] == side]
        boot = json.loads((source/'capture'/f"{a[0]['frame']-1:04}.json").read_text(encoding='utf-8-sig'))
        home = np.array(boot['wholeArms'][side+'Chain']).reshape(4, 2)
        base_h, base_j = orientation(home)
        actual = np.array([r[side]['actual'] for r in a])
        requested = np.array([r[side]['requested'] for r in a])
        h, j = orientation(actual); gh, gj = orientation(requested)
        dh, dj = wrap(h-base_h), wrap(j-base_j)
        goals_h, goals_j = wrap(gh-base_h), wrap(gj-base_j)
        q = np.array([r[side]['applied'] for r in a])
        first = requested[0]
        positional_cost = float(sum(np.linalg.norm(first[k]-home[k])**2 for k in (1, 2)))
        direction_cost = float(np.linalg.norm((first[3]-first[2])-(home[3]-home[2]))**2)
        lengths = np.linalg.norm(np.diff(actual, axis=1), axis=2)
        length_ratio = lengths/np.linalg.norm(np.diff(home, axis=0), axis=1)
        hold = (actual[30:54]-actual[30])
        # Actual endpoint intervals, no inferred "force" score.
        report = dict(side=side, spatialPositionCostAtStart=positional_cost,
            palmChordCostAtStart=direction_cost, initialCostRatio=positional_cost/direction_cost,
            actualActuatorMin=q.min(0).tolist(), actualActuatorMax=q.max(0).tolist(),
            actuatorUnits='model parameter units' if side == 'native' else 'patch rotation degrees',
            liftMaterialElbowDeltaMin=float(dj[:54, 0].min()), liftMaterialElbowDeltaMax=float(dj[:54, 0].max()),
            liftMaterialWristDeltaMin=float(dj[:54, 1].min()), liftMaterialWristDeltaMax=float(dj[:54, 1].max()),
            liftMaterialPalmHeadingDeltaAtLocal6=float(dh[6, 2]),
            liftUpperHeadingDeltaAtLocal6=float(dh[6, 0]),
            liftGoalUpperHeadingDelta=float(goals_h[30, 0]), liftGoalPalmHeadingDelta=float(goals_h[30, 2]),
            holdMaxMaterialDisplacement=float(np.max(np.linalg.norm(hold, axis=2))),
            segmentLengthRatioMin=length_ratio.min(0).tolist(), segmentLengthRatioMax=length_ratio.max(0).tolist(),
            scope='2D final-mesh material geometry; not anatomical force or naturalness certification')
        summary.append(report)
        sides[side] = dict(rows=a, dh=dh, dj=dj, gh=goals_h, gj=goals_j, q=q)
    # Cross-check the proposed physical cause against actual capture metadata.
    first = json.loads((source/'capture/0000.json').read_text(encoding='utf-8-sig'))
    remaining = [p['id'] for p in first['applied'] if p['id'] not in ('Param94', 'Param97', 'Param99')]
    values = []
    states = set()
    for r in rows:
        f = json.loads((source/'capture'/f"{r['frame']:04}.json").read_text(encoding='utf-8-sig'))
        params = {p['id']: p['value'] for p in f['applied']}
        values.append([params[k] for k in remaining]); states.add(f['physicsState'])
    variations = np.ptp(values, axis=0)
    non_arm_dynamic = {k: float(v) for k, v in zip(remaining, variations) if v > 1e-6}
    write_spec(output/'diagnosis.json', dict(source=str(source), sourceTrackSHA256=hashlib.sha256((source/'track.jsonl').read_bytes()).hexdigest(),
        sourceController=protocol.get('controller', 'legacy'),
        costRatioScope='raw legacy position/chord metric on the shared request; NOT the source controller objective',
        sourcePhysics=protocol['physics'], actualPhysicsStates=sorted(states),
        maxPairingDelta=max(r['pairing'] for r in rows),
        maxRestoreError=max(r['geometry']['restoredMeshError'] for r in rows),
        dynamicSdkParametersOutsideThreeNativeAxes=non_arm_dynamic, measurements=summary,
        certainty={'measured': ['transient elbow/wrist compensation', 'distal orientation lag', 'un-normalized task cost', 'physics state'],
                   'inferred': ['these chain relationships can explain ribbon-like appearance'],
                   'notProven': ['sole perceptual cause', 'new controller remedy', 'need for mesh/topology changes']}))
    # Verify angular diagnostics are invariant under common rigid rotation.
    test = np.array(rows[0]['native']['actual']); angle = .713
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    _, a = orientation(test); _, b = orientation(test @ rotation.T)
    if np.max(abs(wrap(a-b))) > 1e-8:
        raise AssertionError('relative-angle diagnostic changed under rigid rotation')
    make_plot(output, sides)
    make_video(source, output, rows, sides)
    for r in summary:
        print(r['side'], 'position/palm chord cost ratio', round(r['initialCostRatio'], 1),
              'elbow excursion', (r['liftMaterialElbowDeltaMin'], r['liftMaterialElbowDeltaMax']),
              'wrist excursion', (r['liftMaterialWristDeltaMin'], r['liftMaterialWristDeltaMax']))


def make_plot(output, sides):
    canvas = Image.new('RGB', (1600, 1000), 'white'); d = ImageDraw.Draw(canvas)
    d.text((20, 12), 'MEASURED SDK MATERIAL CHAIN | solid actual / dashed requested | degrees relative to neutral', fill='black')
    d.text((20, 32), 'Point target changes at 0 and 1.8 seconds. Physics OFF. No force/naturalness score inferred.', fill='black')
    colors = ('#d26114', '#1565bf')
    for col, side in enumerate(('native', 'patch')):
        s = sides[side]; times = np.array([r['localFrame']/30 for r in s['rows']])
        for row in range(2):
            left, top = 80+col*800, 100+row*450; width, height = 660, 330
            if row == 0:
                series = [s['dh'][:, 0], s['dh'][:, 2]]; targets = [s['gh'][:, 0], s['gh'][:, 2]]
                labels = ('upper segment heading', 'palm heading')
            else:
                series = [s['dj'][:, 0], s['dj'][:, 1]]; targets = [s['gj'][:, 0], s['gj'][:, 1]]
                labels = ('elbow relative bend', 'wrist relative bend')
            limit = max(40, int(np.ceil(np.max(abs(np.r_[np.array(series).ravel(), np.array(targets).ravel()]))/20)*20))
            def point(t, v):
                return (left+t/3.6*width, top+height/2-v/limit*height/2)
            d.text((left, top-25), f'{side.upper()} | {labels[0]} (orange), {labels[1]} (blue)', fill='black')
            for val in np.linspace(-limit, limit, 5):
                y = point(0, val)[1]; d.line((left, y, left+width, y), fill='#dddddd'); d.text((left-40, y-5), f'{val:g}', fill='black')
            for t in (0, .6, 1.2, 1.8, 2.4, 3., 3.6):
                x = point(t, 0)[0]; d.line((x, top, x, top+height), fill='#eeeeee'); d.text((x-10, top+height+10), f'{t:.1f}s', fill='black')
            d.rectangle((left, top, left+width, top+height), outline='black')
            for actual, target, color in zip(series, targets, colors):
                pts = [point(t, v) for t, v in zip(times, actual)]; goal = [point(t, v) for t, v in zip(times, target)]
                d.line(pts, fill=color, width=3)
                for i in range(0, len(goal)-1, 4):
                    d.line(goal[i:i+3], fill=color, width=1)
    canvas.save(output/'joint-timing.png')


def make_video(source, output, rows, sides):
    video = output/'arm-support-diagnosis.mp4'
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), '-loglevel', 'error', '-f', 'rawvideo', '-vcodec', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', '1280x720', '-r', '30', '-i', '-', '-an', '-c:v', 'libx264',
        '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        for r in rows:
            side = r['active']; local = r['localFrame']; s = sides[side]; record = r[side]
            frame = json.loads((source/'capture'/f"{r['frame']:04}.json").read_text(encoding='utf-8-sig'))
            canvas = Image.new('RGB', (1280, 720), '#eeeeee'); d = ImageDraw.Draw(canvas)
            d.text((12, 8), f"ACTUAL SDK | {side} | local {local/30:.3f}s | frame {r['frame']} | PHYSICS OFF", fill='black')
            d.text((12, 29), 'Orange=actual material nodes; green=request elbow/wrist. Right: same-frame raw fixed torso crop.', fill='black')
            with Image.open(source/'capture'/f"{r['frame']:04}.png") as im:
                raw = im.convert('RGB').crop((160, 100, 864, 676)).resize((640, 524))
            canvas.paste(raw, (0, 66)); canvas.paste(raw, (640, 66))
            actual = np.array(record['actual']); screen = np.array(frame[side+'NodeScreen']).reshape(4, 2)
            affine = np.linalg.lstsq(np.c_[actual, np.ones(4)], screen, rcond=None)[0]
            request = np.c_[record['requested'], np.ones(4)] @ affine
            transform = lambda p: ((p[0]-160)*640/704, (p[1]-100)*524/576+66)
            d.line([transform(p) for p in screen], fill='#f48617', width=3)
            for p in screen:
                x, y = transform(p); d.ellipse((x-3, y-3, x+3, y+3), fill='#f48617')
            for p in request[1:3]:
                x, y = transform(p); d.ellipse((x-6, y-6, x+6, y+6), outline='#00db70', width=2)
            d.text((12, 605), 'MATERIAL GEOMETRY relative to neutral (degrees):', fill='black')
            d.text((12, 625), f"upper heading {s['dh'][local,0]:+6.1f} / requested {s['gh'][local,0]:+6.1f}    palm heading {s['dh'][local,2]:+6.1f} / requested {s['gh'][local,2]:+6.1f}", fill='black')
            d.text((12, 646), f"elbow bend {s['dj'][local,0]:+6.1f} / requested {s['gj'][local,0]:+6.1f}    wrist bend {s['dj'][local,1]:+6.1f} / requested {s['gj'][local,1]:+6.1f}", fill='black')
            d.text((12, 670), f"applied actuators: {np.round(s['q'][local],2).tolist()} | wrist error {record['wristError']:.4f} | palm error {record['palmErrorDeg']:.1f}deg", fill='black')
            d.text((12, 692), 'Recorded evidence of transient compensation; NOT a proposed correct motion or automatic force rating.', fill='black')
            proc.stdin.write(canvas.tobytes())
        proc.stdin.close(); error = proc.stderr.read(); code = proc.wait()
        if code:
            raise RuntimeError(error.decode(errors='replace'))
    print(video)


if __name__ == '__main__':
    analyze(Path(sys.argv[1]), Path(sys.argv[2]))
