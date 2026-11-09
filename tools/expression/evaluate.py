"""Expression metrics computed from rendered geometry, compared against the source trajectory."""
import math

import numpy as np

from . import roles as roles_module


def fx_features(frame, role_map):
    geometry = {d['id']: d for d in frame['drawables']}
    anchor = roles_module.anchors(role_map, geometry)
    if any(anchor[role] is None for role in ('rightHand', 'leftHand', 'chest', 'head', 'hip')):
        return None
    scale = math.dist(anchor['head'], anchor['hip']) or 1e-6
    right = ((anchor['rightHand'][0] - anchor['chest'][0]) / scale,
             (anchor['rightHand'][1] - anchor['chest'][1]) / scale)
    left = ((anchor['leftHand'][0] - anchor['chest'][0]) / scale,
            (anchor['leftHand'][1] - anchor['chest'][1]) / scale)
    return {'right': right, 'left': left, 'separation': math.dist(right, left),
            'rightDistance': math.hypot(*right), 'scale': scale, 'anchors': anchor}


def series(frames, role_map):
    result = []
    for frame in frames:
        feature = fx_features(frame, role_map)
        if feature is None:
            return None
        result.append(feature)
    return result


def correlation(a, b):
    if len(a) < 3 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b):
    if len(a) < 3:
        return None
    rank_a = np.argsort(np.argsort(a))
    rank_b = np.argsort(np.argsort(b))
    return correlation(rank_a.astype(float), rank_b.astype(float))


NO_MOTION_RANGE = 0.05
PASS_LATERAL = 0.5
PASS_DISTANCE = 0.5
PASS_PEAK_RATIO = (0.5, 2.0)
BATCH_SPEARMAN = 0.5


def compare(source, rendered, arm='left'):
    """Compare opening relations as changes from the clip's own first frame.

    Absolute hand offsets are dominated by different neutral arm postures between the two
    rigs (source arms hang to hip level, Fu Xuan's rest near the waist), so the preserved
    quantity is the motion relation: outward lateral travel, vertical travel and range.
    """
    count = min(len(source), len(rendered))
    if count < 3:
        return {'status': 'insufficient-frames'}
    source = source[:count]
    rendered = rendered[:count]
    base_source = source[0][arm]
    base_rendered = rendered[0]['right']
    source_lateral = [s[arm][0] - base_source[0] for s in source]
    rendered_lateral = [r['right'][0] - base_rendered[0] for r in rendered]
    source_height = [s[arm][1] - base_source[1] for s in source]
    rendered_height = [r['right'][1] - base_rendered[1] for r in rendered]
    source_distance = list(source_lateral)
    rendered_distance = list(rendered_lateral)
    lateral = correlation(source_lateral, rendered_lateral)
    height = correlation(source_height, rendered_height)
    distance = correlation(source_distance, rendered_distance)
    peak_source = max(abs(v) for v in source_lateral)
    peak_rendered = max(abs(v) for v in rendered_lateral)
    ratio = peak_rendered / peak_source if peak_source > 1e-6 else None
    motion = max(rendered_lateral) - min(rendered_lateral)
    height_source = max(abs(v) for v in source_height)
    height_rendered = max(abs(v) for v in rendered_height)
    height_ratio = height_rendered / height_source if height_source > 1e-6 else None
    direction = None
    if np.std(source_lateral) > 1e-9 and np.std(rendered_lateral) > 1e-9:
        direction = float(np.sign(np.corrcoef(source_lateral, rendered_lateral)[0, 1]))
    status = 'measured'
    reasons = []
    if motion < NO_MOTION_RANGE:
        status = 'no-motion'
        reasons.append('rendered hand motion below threshold; cannot claim preservation')
    if ratio is not None and not (PASS_PEAK_RATIO[0] <= ratio <= PASS_PEAK_RATIO[1]):
        reasons.append('lateral opening ratio outside accepted band')
    if lateral is not None and lateral < PASS_LATERAL:
        reasons.append('lateral trajectory disagree')
    if distance is not None and distance < PASS_DISTANCE:
        reasons.append('opening trajectory disagree')
    passed = status == 'measured' and not reasons
    return {
        'status': 'pass' if passed else ('no-motion' if status == 'no-motion' else 'fail'),
        'reasons': reasons,
        'frames': count,
        'lateralCorrelation': lateral, 'heightCorrelation': height, 'distanceCorrelation': distance,
        'peakSource': round(peak_source, 4), 'peakRendered': round(peak_rendered, 4), 'peakRatio': ratio,
        'heightPeakSource': round(height_source, 4), 'heightPeakRendered': round(height_rendered, 4),
        'heightRatio': height_ratio,
        'renderedMotionRange': motion, 'direction': direction,
        'sourcePeakDelta': [round(v, 4) for v in (source_lateral[int(np.argmax(np.abs(source_lateral)))],
                                                  source_height[int(np.argmax(np.abs(source_lateral)))])],
        'renderedPeakDelta': [round(v, 4) for v in (rendered_lateral[int(np.argmax(np.abs(rendered_lateral)))],
                                                    rendered_height[int(np.argmax(np.abs(rendered_lateral)))])],
    }


def pairing(frames, threshold=1e-4):
    """Verify that recorded geometry matches the unmanaged model state in the same frame."""
    if not frames:
        return {'frames': 0, 'maxDelta': None, 'ok': False}
    worst = max(frame.get('pairingMaxDelta', 0.0) for frame in frames)
    return {'frames': len(frames), 'maxDelta': worst, 'ok': worst <= threshold}


def write_check(frames, owned):
    """Requested-vs-applied check for channels the clip owns.

    Channels deliberately handed to physics (for example ParamBodyAngleX2) are excluded, so a
    physics override is not silently reported as a successful explicit write.
    """
    mismatches = []
    for frame in frames:
        applied = {p['id']: p['value'] for p in frame['applied']}
        for parameter in frame['requested']:
            if parameter['id'] not in owned:
                continue
            value = applied.get(parameter['id'])
            if value is None or abs(value - parameter['value']) > 1e-4:
                mismatches.append({'frame': frame.get('frame'), 'id': parameter['id'],
                                   'requested': parameter['value'], 'applied': value})
    return {'owned': sorted(owned), 'mismatches': mismatches[:20], 'count': len(mismatches)}


def batch(per_case):
    names = sorted(per_case)
    source = [per_case[name]['comparison']['peakSource'] for name in names]
    rendered = [per_case[name]['comparison']['peakRendered'] for name in names]
    return {'cases': len(names), 'peakSpearman': spearman(source, rendered),
            'passed': sum(1 for name in names if per_case[name]['comparison']['status'] == 'pass'),
            'noMotion': sum(1 for name in names if per_case[name]['comparison']['status'] == 'no-motion'),
            'failed': sum(1 for name in names if per_case[name]['comparison']['status'] == 'fail')}
