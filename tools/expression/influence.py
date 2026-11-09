"""Sweep every model parameter to its extremes and map which drawables actually move."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.expression import unity

NEUTRAL_OUTPUT = unity.ROOT / 'captures' / 'body-channels-001' / '0000.json'


def parameter_ranges():
    state = json.loads(NEUTRAL_OUTPUT.read_text(encoding='utf-8-sig'))
    return [(p['id'], p['minimum'], p['maximum'], p['defaultValue']) for p in state['applied']]


def poses(ranges):
    result = [{'label': 'neutral', 'ids': [], 'values': []}]
    low = {'ParamEyeLOpen', 'ParamEyeROpen', 'ParamMouthOpenY', 'ParamBreath'}
    for param, minimum, maximum, default in ranges:
        for tag, value in (('max', maximum), ('min', minimum)):
            if param in low and tag == 'min':
                continue
            result.append({'label': f'{param} {tag} {value:g}', 'ids': [param], 'values': [float(value)]})
    return result


def main(run: Path):
    spec = run / 'protocol' / 'param-influence.json'
    sweep = poses(parameter_ranges())
    unity.write_spec(spec, {'poses': sweep, 'physics': False})
    output = run / 'calibration' / 'param-influence'
    unity.run('LabMotion.RunPoseLab', output, {'ACTION_LAB_MODEL': unity.model3_path(),
                                               'ACTION_LAB_POSES': spec},
              run / 'logs' / 'param-influence.log', timeout=5400)
    print('poses', len(sweep), 'written', len(list(output.glob('[0-9][0-9][0-9][0-9].json'))))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
