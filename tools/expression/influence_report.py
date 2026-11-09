"""Analyse the parameter-influence sweep: which drawables each parameter moves."""
import collections
import json
from pathlib import Path

ARM_IDS = (
    'Param31', 'Param32', 'Param33', 'Param34', 'Param36', 'Param37', 'Param38', 'Param39',
    'Param92', 'Param93', 'Param94', 'Param95', 'Param96', 'Param97', 'Param98', 'Param99',
    'Param100', 'Param101', 'Param102', 'Param103', 'Param104', 'Param105', 'Param106',
    'Param107', 'Param108', 'Param109', 'Param110', 'Param111', 'Param112', 'Param113',
    'Param114', 'Param115', 'Param116', 'Param117', 'Param118', 'Param119', 'Param120',
    'Param153',
)


def load(root: Path, name: str):
    path = root / name
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sweep(sweep_dir: Path):
    neutral = load(sweep_dir, '0000.json')
    base = {d['id']: d for d in neutral['drawables']}
    moved = collections.defaultdict(dict)
    for path in sorted(sweep_dir.glob('[0-9][0-9][0-9][0-9].json'))[1:]:
        frame = json.loads(path.read_text(encoding='utf-8-sig'))
        parameter = frame['label'].split()[0]
        for drawable in frame['drawables']:
            before = base[drawable['id']]
            delta = max(abs(drawable['cx'] - before['cx']), abs(drawable['cy'] - before['cy']),
                        abs(drawable['minx'] - before['minx']), abs(drawable['maxx'] - before['maxx']),
                        abs(drawable['miny'] - before['miny']), abs(drawable['maxy'] - before['maxy']))
            if delta > 0.004:
                moved[parameter][drawable['id']] = max(moved[parameter].get(drawable['id'], 0.0), delta)
    return base, moved


def report(sweep_dir: Path):
    base, moved = sweep(sweep_dir)
    globalish = {p for p, targets in moved.items() if len(targets) > 150}
    arms = {}
    for parameter in ARM_IDS:
        targets = {k: v for k, v in moved.get(parameter, {}).items()}
        arms[parameter] = {
            'drawables': len(targets),
            'top': sorted(((round(v, 4), k) for k, v in targets.items()), reverse=True)[:8],
        }
    return {'totalDrawables': len(base), 'globalish': sorted(globalish), 'armRelated': arms}


if __name__ == '__main__':
    import sys
    root = Path(sys.argv[1]) / 'calibration' / 'param-influence'
    print(json.dumps(report(root), ensure_ascii=True, indent=1))
