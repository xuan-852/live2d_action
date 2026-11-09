"""Measure the right-arm response grid and the body/head anchors, then save calibration."""
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.expression import roles as roles_module
from tools.expression import unity

SHOULDER = 'Param94'
SWING = 'Param97'
WRIST = 'Param99'
EXTEND = 'Param118'
GRID_SHOULDER = (-30, -15, 0, 15, 30, 45, 60)
GRID_SWING = (-30, -15, 0, 15, 30)


def poses():
    result = [{'label': 'neutral', 'ids': [], 'values': []}]
    for shoulder in GRID_SHOULDER:
        for swing in GRID_SWING:
            result.append({'label': f'grid s{shoulder:+d} w{swing:+d}', 'ids': [SHOULDER, SWING],
                           'values': [float(shoulder), float(swing)]})
    for wrist in (-30, 0, 30):
        result.append({'label': f'wrist {wrist:+d}', 'ids': [SHOULDER, WRIST], 'values': [20.0, float(wrist)]})
    for extend in (0, 15, 30):
        result.append({'label': f'extend {extend:+d}', 'ids': [SHOULDER, EXTEND], 'values': [30.0, float(extend)]})
    return result


def measure(directory: Path, role_map):
    rows = []
    for path in sorted(directory.glob('[0-9][0-9][0-9][0-9].json')):
        frame = json.loads(path.read_text(encoding='utf-8-sig'))
        geometry = {d['id']: d for d in frame['drawables']}
        anchor = roles_module.anchors(role_map, geometry)
        params = {p['id']: p['value'] for p in frame['applied']}
        rows.append({'label': frame['label'], 'params': {k: params[k] for k in (SHOULDER, SWING, WRIST, EXTEND)},
                     'anchors': anchor, 'pairingMaxDelta': frame['pairingMaxDelta']})
    return rows


def hull(points):
    points = sorted(set(points))
    if len(points) < 3:
        return points

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def inside(polygon, point):
    if len(polygon) < 3:
        return False
    sign = 0
    for index, a in enumerate(polygon):
        b = polygon[(index + 1) % len(polygon)]
        value = (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
        if abs(value) < 1e-12:
            continue
        if sign == 0:
            sign = 1 if value > 0 else -1
        elif (value > 0) != (sign > 0):
            return False
    return True


def distance_inside(polygon, point):
    nearest = min(math.dist(point, a) for a in polygon)
    if inside(polygon, point):
        return 0.0
    return nearest


def build(sweep_dir: Path, grid_dir: Path, output: Path):
    role_map = roles_module.roles(sweep_dir)
    rows = measure(grid_dir, role_map)
    by_label = {row['label']: row for row in rows}
    neutral = by_label['neutral']
    hand_points = []
    grid = []
    for row in rows:
        if not row['label'].startswith('grid'):
            continue
        hand = row['anchors']['rightHand']
        hand_points.append((round(hand[0], 5), round(hand[1], 5)))
        grid.append({'shoulder': row['params'][SHOULDER], 'swing': row['params'][SWING],
                     'hand': hand, 'chest': row['anchors']['chest'], 'head': row['anchors']['head']})
    calibration = {
        'roles': role_map,
        'neutral': neutral['anchors'],
        'grid': grid,
        'reachHull': hull(hand_points),
        'wristEffect': [by_label[f'wrist {w:+d}']['anchors']['rightHand'] for w in (-30, 0, 30)],
        'extendEffect': [by_label[f'extend {e:+d}']['anchors']['rightHand'] for e in (0, 15, 30)],
        'pairingMaxDelta': max(row['pairingMaxDelta'] for row in rows),
        'source': str(grid_dir),
    }
    output.write_text(json.dumps(calibration, ensure_ascii=True, indent=1), encoding='utf-8')
    return calibration


def main(run: Path):
    spec = run / 'protocol' / 'arm-grid.json'
    unity.write_spec(spec, {'poses': poses(), 'physics': False})
    output_dir = run / 'calibration' / 'arm-grid'
    unity.run('LabMotion.RunPoseLab', output_dir, {'ACTION_LAB_MODEL': unity.model3_path(),
                                                   'ACTION_LAB_POSES': spec},
              run / 'logs' / 'arm-grid.log')
    calibration = build(run / 'calibration' / 'param-influence', output_dir,
                        run / 'calibration' / 'arm-calibration.json')
    print(json.dumps({'gridPoints': len(calibration['grid']), 'hull': len(calibration['reachHull']),
                      'pairingMaxDelta': calibration['pairingMaxDelta'],
                      'neutralHand': calibration['neutral']['rightHand']}, indent=1))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
