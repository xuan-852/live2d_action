"""Load source T2M trajectories and derive body-normalised 2D expression features."""
import json
import math
from pathlib import Path

JOINTS = {'pelvis': 0, 'rightHip': 1, 'leftHip': 2, 'spine': 3, 'rightKnee': 4, 'leftKnee': 5,
          'chest': 6, 'rightAnkle': 7, 'leftAnkle': 8, 'neck': 9, 'rightFoot': 10, 'leftFoot': 11,
          'rightCollar': 13, 'leftCollar': 14, 'head': 15, 'rightShoulder': 16, 'leftShoulder': 17,
          'rightElbow': 18, 'leftElbow': 19, 'rightWrist': 20, 'leftWrist': 21}


def load(path: Path):
    """Read one generated reference. phaseKeys give (time, frame index); frames hold 22 joints."""
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    frames = [{name: frame[str(index)] for name, index in JOINTS.items()} for frame in data['frames']]
    keys = [(key['t'], key['phase']) for key in data['phaseKeys']]
    return {'id': data.get('caption', path.stem), 'seed': data.get('seed'), 'kind': data.get('kind'),
            'frames': frames, 'keys': keys, 'sha256': data.get('motionSha256'),
            'sourceSha256': data.get('sourceSha256'), 'mirrored': data.get('presentationMirror'),
            'path': str(path)}


def blend(a, b, weight):
    return {name: [a[name][i] + (b[name][i] - a[name][i]) * weight for i in range(3)] for name in a}


def at_time(trajectory, time):
    keys = trajectory['keys']
    if time <= keys[0][0]:
        phase = keys[0][1]
    elif time >= keys[-1][0]:
        phase = keys[-1][1]
    else:
        index = 0
        while index + 1 < len(keys) and keys[index + 1][0] < time:
            index += 1
        t0, p0 = keys[index]
        t1, p1 = keys[index + 1]
        span = t1 - t0
        weight = 0.0 if span <= 0 else (time - t0) / span
        phase = p0 + (p1 - p0) * weight
    low = int(math.floor(phase))
    high = min(low + 1, len(trajectory['frames']) - 1)
    return blend(trajectory['frames'][low], trajectory['frames'][high], phase - low)


def features(joints):
    """Front-view 2D relations, normalised by head-to-pelvis distance (torso length)."""
    flat = {name: (-point[0], point[1]) for name, point in joints.items()}
    chest = ((flat['rightShoulder'][0] + flat['leftShoulder'][0]) / 2,
             (flat['rightShoulder'][1] + flat['leftShoulder'][1]) / 2)
    head = flat['head']
    scale = math.dist(head, flat['pelvis']) or 1e-6
    right = ((flat['rightWrist'][0] - chest[0]) / scale, (flat['rightWrist'][1] - chest[1]) / scale)
    left = ((flat['leftWrist'][0] - chest[0]) / scale, (flat['leftWrist'][1] - chest[1]) / scale)
    raw = (joints['rightShoulder'][0] - joints['leftShoulder'][0],
           joints['rightShoulder'][2] - joints['leftShoulder'][2])
    return {
        'right': right, 'left': left,
        'separation': math.dist(right, left),
        'rightDistance': math.hypot(*right),
        'leftDistance': math.hypot(*left),
        'yaw': math.atan2(-raw[1], raw[0] or 1e-6),
        'headLateral': (head[0] - chest[0]) / scale,
        'wristSlope': math.atan2(flat['rightWrist'][1] - flat['rightElbow'][1],
                                 flat['rightWrist'][0] - flat['rightElbow'][0]),
    }


def series(trajectory, fps=30.0, limit=None):
    duration = trajectory['keys'][-1][0] if trajectory['keys'] else 0.0
    if limit is not None:
        duration = min(duration, limit)
    count = int(round(duration * fps)) + 1
    times = [index / fps for index in range(count)]
    return times, [features(at_time(trajectory, t)) for t in times]


def summarise(samples):
    peak = max(samples, key=lambda s: s['rightDistance'])
    return {
        'peakRightDistance': round(peak['rightDistance'], 4),
        'peakRight': [round(v, 4) for v in peak['right']],
        'peakSeparation': round(max(s['separation'] for s in samples), 4),
        'minSeparation': round(min(s['separation'] for s in samples), 4),
        'meanRightDistance': round(sum(s['rightDistance'] for s in samples) / len(samples), 4),
        'lateralRange': [round(min(s['right'][0] for s in samples), 4),
                         round(max(s['right'][0] for s in samples), 4)],
        'heightRange': [round(min(s['right'][1] for s in samples), 4),
                        round(max(s['right'][1] for s in samples), 4)],
    }


if __name__ == '__main__':
    import sys
    root = Path(sys.argv[1])
    for path in sorted(root.glob('*.json'))[:4]:
        trajectory = load(path)
        times, samples = series(trajectory, limit=6.0)
        print(path.name, len(trajectory['frames']), 'frames', round(times[-1], 2), 's', summarise(samples))
