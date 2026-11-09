"""Batch pipeline: select cases, adapt with two methods, capture, evaluate, build review."""
import json
import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import imageio_ffmpeg
from PIL import Image, ImageDraw

from tools.expression import adapter, evaluate, source, unity

FPS = 30
LIMIT = 6.0
REFERENCES = Path('E:/FuXuanLab/runs/20261030-generated-observation-04/references')
STICK = (512, 512)


def classify(summary):
    if summary['peakSeparation'] >= 1.15:
        return 'open'
    if summary['peakRightDistance'] <= 0.7:
        return 'closed'
    return 'guide'


def select(run: Path, wanted=8):
    records = []
    for path in sorted(REFERENCES.glob('*.json')):
        trajectory = source.load(path)
        times, samples = source.series(trajectory, FPS, LIMIT)
        summary = source.summarise(samples)
        records.append({'id': trajectory['id'], 'path': str(path), 'seed': trajectory['seed'],
                        'kind': trajectory['kind'], 'sha256': trajectory['sha256'],
                        'frames': len(trajectory['frames']), 'duration': round(times[-1], 3),
                        'summary': summary, 'klass': classify(summary)})
    records.sort(key=lambda r: (r['klass'], -r['summary']['peakRightDistance']))
    chosen = []
    per_class = max(1, wanted // 3)
    for klass in ('open', 'guide', 'closed'):
        group = [r for r in records if r['klass'] == klass]
        for record in group[:per_class]:
            record['split'] = 'fit'
            chosen.append(record)
    remaining = [r for r in records if r['id'] not in {c['id'] for c in chosen}]
    for record in remaining[:wanted - len(chosen)]:
        record['split'] = 'heldout'
        record['klass_rank'] = 'extra'
        chosen.append(record)
    chosen.sort(key=lambda r: r['id'])
    return chosen, records


def stick_frame(trajectory, time, size=STICK):
    joints = {name: (-point[0], point[1]) for name, point in source.at_time(trajectory, time).items()}
    points = list(joints.values())
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1e-3) * 1.35
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    image = Image.new('RGB', (size[0], size[1]), (24, 26, 32))
    draw = ImageDraw.Draw(image)

    def place(point):
        return (size[0] / 2 + (point[0] - cx) / span * size[0], size[1] / 2 - (point[1] - cy) / span * size[1])

    bones = [('pelvis', 'spine'), ('spine', 'chest'), ('chest', 'neck'), ('neck', 'head'),
             ('rightShoulder', 'leftShoulder'), ('neck', 'rightShoulder'), ('neck', 'leftShoulder'),
             ('rightShoulder', 'rightElbow'), ('rightElbow', 'rightWrist'),
             ('leftShoulder', 'leftElbow'), ('leftElbow', 'leftWrist'),
             ('pelvis', 'rightHip'), ('pelvis', 'leftHip'),
             ('rightHip', 'rightKnee'), ('rightKnee', 'rightAnkle'),
             ('leftHip', 'leftKnee'), ('leftKnee', 'leftAnkle')]
    for a, b in bones:
        draw.line([place(joints[a]), place(joints[b])], fill=(180, 190, 205), width=4)
    for name, colour in (('rightWrist', (255, 120, 120)), ('leftWrist', (120, 180, 255)),
                         ('head', (240, 220, 160))):
        x, y = place(joints[name])
        draw.ellipse([x - 7, y - 7, x + 7, y + 7], fill=colour)
    draw.text((10, 10), 'source skeleton (front, mirrored)', fill=(200, 200, 200))
    return image


def write_sticks(trajectory, output: Path, fps=FPS, limit=LIMIT):
    output.mkdir(parents=True, exist_ok=True)
    count = int(round(min(trajectory['keys'][-1][0], limit) * fps)) + 1
    for index in range(count):
        stick_frame(trajectory, index / fps).save(output / f'{index:04d}.png')
    return count


def encode_triple(stick_dir, baseline_dir, candidate_dir, count, output: Path):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    graph = ('[0:v]scale=512:512[a];[1:v]crop=720:760:150:0,scale=512:512[b];'
             '[2:v]crop=720:760:150:0,scale=512:512[c];[a][b][c]hstack=inputs=3[v]')
    command = [ffmpeg, '-y',
               '-framerate', str(FPS), '-i', str(stick_dir / '%04d.png'),
               '-framerate', str(FPS), '-i', str(baseline_dir / '%04d.png'),
               '-framerate', str(FPS), '-i', str(candidate_dir / '%04d.png'),
               '-filter_complex', graph, '-map', '[v]',
               '-frames:v', str(count), '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', str(output)]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def run_case(run: Path, record, calibration, role_map, method, reference=None, arm='left'):
    trajectory = source.load(Path(record['path']))
    times, samples = source.series(trajectory, FPS, LIMIT)
    curves, info = (adapter.candidate_curves if method == 'candidate' else adapter.baseline_curves)(
        samples, calibration, FPS, reference, arm)
    spec = adapter.spec(curves, FPS, include_physics=True)
    spec_path = run / 'protocol' / 'curves' / f'{record["id"]}-{method}.json'
    unity.write_spec(spec_path, spec)
    output = run / 'cases' / record['id'] / method
    unity.run('LabMotion.RunMotion', output,
              {'ACTION_LAB_MODEL': unity.model3_path(), 'ACTION_LAB_CURVES': spec_path},
              run / 'logs' / f'{record["id"]}-{method}.log', timeout=1800)
    frames = [json.loads(path.read_text(encoding='utf-8-sig'))
              for path in sorted(output.glob('[0-9][0-9][0-9][0-9].json'))]
    rendered = evaluate.series(frames, role_map)
    comparison = evaluate.compare(samples, rendered, arm) if rendered else {'status': 'no-geometry'}
    pairing = max(frame['pairingMaxDelta'] for frame in frames)
    applied = all(abs(a['value'] - r['value']) < 1e-4
                  for frame in frames for r, a in zip(frame['requested'], frame['applied'])
                  if a['id'] != adapter.BODY_TURN)
    return {'method': method, 'frames': len(frames), 'pairingMaxDelta': pairing,
            'requestedEqualsAppliedExcludingX2': applied, 'adapter': info,
            'comparison': comparison, 'times': times[:len(samples)]}


def main(run: Path):
    calibration = adapter.load_calibration(run / 'calibration' / 'arm-calibration.json')
    role_map = calibration['roles']
    protocol_path = run / 'protocol' / 'protocol.json'
    if protocol_path.is_file():
        protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    else:
        chosen, all_records = select(run)
        offsets = []
        for record in chosen:
            if record['split'] != 'fit':
                continue
            _, samples = source.series(source.load(Path(record['path'])), FPS, LIMIT)
            offsets.extend(sample['left'] for sample in samples)
        reference = [sum(o[0] for o in offsets) / len(offsets), sum(o[1] for o in offsets) / len(offsets)]
        protocol = {'fps': FPS, 'limitSeconds': LIMIT,
                    'thresholds': {'lateral': evaluate.PASS_LATERAL, 'distance': evaluate.PASS_DISTANCE,
                                   'peakRatio': evaluate.PASS_PEAK_RATIO,
                                   'batchSpearman': evaluate.BATCH_SPEARMAN,
                                   'noMotionRange': evaluate.NO_MOTION_RANGE},
                    'armMapping': {'sourceJoint': 'leftWrist (display +x side)',
                                   'targetGroup': 'Param94 articulated arm (rendered +x side)'},
                    'referenceOffset': [round(reference[0], 5), round(reference[1], 5)],
                    'referenceNote': 'mean source hand offset over fit-split frames; single global constant',
                    'cases': chosen,
                    'source': {'directory': str(REFERENCES), 'kind': 't2m-generated, old run references',
                               'readOnly': True},
                    'frozen': 'selection, thresholds and reference recorded before any rendered comparison'}
        unity.write_spec(protocol_path, protocol)
    reference = protocol['referenceOffset']
    results = {}
    for record in protocol['cases']:
        entry = {'id': record['id'], 'split': record['split'], 'klass': record['klass'],
                 'source': record['summary'], 'sha256': record['sha256']}
        for method in ('baseline', 'candidate'):
            entry[method] = run_case(run, record, calibration, role_map, method, reference, 'left')
        trajectory = source.load(Path(record['path']))
        stick_dir = run / 'cases' / record['id'] / 'source'
        count = write_sticks(trajectory, stick_dir)
        encode_triple(stick_dir, run / 'cases' / record['id'] / 'baseline',
                      run / 'cases' / record['id'] / 'candidate', count,
                      run / 'cases' / record['id'] / 'compare.mp4')
        results[record['id']] = entry
    (run / 'results.json').write_text(json.dumps(results, ensure_ascii=True, indent=1), encoding='utf-8')
    summary = {'baseline': evaluate.batch({k: {'comparison': v['baseline']['comparison']} for k, v in results.items()}),
               'candidate': evaluate.batch({k: {'comparison': v['candidate']['comparison']} for k, v in results.items()})}
    (run / 'summary.json').write_text(json.dumps(summary, ensure_ascii=True, indent=1), encoding='utf-8')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[2] == '--smoke':
        run_root = Path(sys.argv[1])
        calibration = adapter.load_calibration(run_root / 'calibration' / 'arm-calibration.json')
        chosen, all_records = select(run_root, wanted=2)
        record = chosen[0]
        offsets = []
        for entry in chosen:
            if entry['split'] != 'fit':
                continue
            _, samples = source.series(source.load(Path(entry['path'])), FPS, LIMIT)
            offsets.extend(sample['left'] for sample in samples)
        reference = [sum(o[0] for o in offsets) / len(offsets), sum(o[1] for o in offsets) / len(offsets)]
        print(json.dumps({'case': record['id'], 'klass': record['klass'],
                          'reference': reference, 'summary': record['summary']}, indent=1))
        for method in ('baseline', 'candidate'):
            result = run_case(run_root, record, calibration, calibration['roles'], method, reference, 'left')
            print(method, json.dumps(result['comparison'], indent=1), 'pairing', result['pairingMaxDelta'])
        sys.exit(0)
    main(Path(sys.argv[1]))
