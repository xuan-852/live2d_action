"""Discover which drawables act as hand/chest/head anchors, from the influence sweep."""
import collections
import json
from pathlib import Path

RECORD = {}


def load(path: Path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def frame_map(directory: Path):
    result = {}
    for path in sorted(directory.glob('[0-9][0-9][0-9][0-9].json')):
        frame = json.loads(path.read_text(encoding='utf-8-sig'))
        result[frame['label']] = (path.stem, {d['id']: d for d in frame['drawables']}, frame)
    return result


def moved_by(frames, parameter, threshold=0.01):
    base = frames['neutral'][1]
    key = next(k for k in frames if k.startswith(parameter + ' ') and k.split()[1] == 'max')
    _, after, _ = frames[key]
    deltas = {}
    for name, drawable in after.items():
        before = base[name]
        deltas[name] = max(abs(drawable['cx'] - before['cx']), abs(drawable['cy'] - before['cy']),
                           abs(drawable['maxy'] - before['maxy']), abs(drawable['miny'] - before['miny']))
    return {k: v for k, v in deltas.items() if v > threshold}


def roles(sweep_dir: Path):
    frames = frame_map(sweep_dir)
    neutral = frames['neutral'][1]
    right_hand = moved_by(frames, 'Param94', 0.5)
    x2 = moved_by(frames, 'ParamBodyAngleX2', 0.01)
    head = moved_by(frames, 'ParamAngleZ', 0.006)
    # Left hand: mirror position of the right hand cluster in neutral space.
    right_names = set(right_hand)
    right_centre = (sum(neutral[n]['cx'] for n in right_names) / len(right_names),
                    sum(neutral[n]['cy'] for n in right_names) / len(right_names))
    candidates = [d for d in neutral.values() if d['id'] not in right_names]
    left = [d for d in candidates
            if abs(d['cx'] + right_centre[0]) < 0.16 and abs(d['cy'] - right_centre[1]) < 0.16
            and (d['maxx'] - d['minx']) < 0.12 and (d['maxy'] - d['miny']) < 0.12]
    chest = [d for d in neutral.values() if any(d['id'] == k for k in x2)]
    chest_candidates = sorted(chest, key=lambda d: (d['cx'] ** 2 + (d['cy'] - 0.05) ** 2))[:12]
    head_candidates = sorted((neutral[k] for k in head), key=lambda d: -d['cy'])[:8]
    hip_candidates = sorted(candidates, key=lambda d: d['cy'])[:6]
    return {
        'rightHand': sorted(right_hand, key=lambda k: neutral[k]['cx']),
        'leftHand': sorted(d['id'] for d in left),
        'chest': sorted(d['id'] for d in chest_candidates),
        'head': sorted(d['id'] for d in head_candidates),
        'hip': sorted(d['id'] for d in hip_candidates),
        'neutral': {k: [neutral[k]['cx'], neutral[k]['cy']] for k in
                    set(right_hand) | {d['id'] for d in left} | {d['id'] for d in chest_candidates}
                    | {d['id'] for d in head_candidates} | {d['id'] for d in hip_candidates}},
    }


def anchors(role_map, geometry):
    """Cluster centroid for a role, from one frame's drawable geometry."""
    result = {}
    for role in ('rightHand', 'leftHand', 'chest', 'head', 'hip'):
        names = [n for n in role_map[role] if n in geometry]
        if not names:
            result[role] = None
            continue
        result[role] = [sum(geometry[n]['cx'] for n in names) / len(names),
                        sum(geometry[n]['cy'] for n in names) / len(names)]
    return result


if __name__ == '__main__':
    import sys
    root = Path(sys.argv[1]) / 'calibration' / 'param-influence'
    result = roles(root)
    print(json.dumps({k: (len(v) if isinstance(v, list) else 'map') for k, v in result.items()}, indent=1))
    print('leftHand', result['leftHand'][:10])
    print('chest', result['chest'])
    print('head', result['head'])
