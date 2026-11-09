"""Dense response probe for one arm chain: uniform and per-link sweeps."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.expression import unity

CHAIN = 'ArtMesh272'
ANGLES = (5, 10, 15, 20, 25, 30, 35, 40, 45, -5, -10, -15, -20, -25, -30, -35, -40, -45)


def poses():
    ids = [f'Param_Angle_Rotation_{link}_{CHAIN}' for link in range(1, 8)]
    result = [{'label': 'neutral', 'ids': [], 'values': []}]
    for angle in ANGLES:
        result.append({'label': f'uniform {angle:+d}', 'ids': ids, 'values': [float(angle)] * 7})
    result.append({'label': 'uniform +20 repeat', 'ids': ids, 'values': [20.0] * 7})
    for link in range(1, 8):
        for angle in (20, -20):
            result.append({'label': f'link{link} {angle:+d}', 'ids': [ids[link - 1]], 'values': [float(angle)]})
    return result


def main(run: Path):
    spec = run / 'protocol' / 'pose-response-272.json'
    unity.write_spec(spec, {'poses': poses(), 'physics': False})
    output = run / 'calibration' / 'pose-response-272'
    unity.run('LabMotion.RunPoseLab', output, {'ACTION_LAB_MODEL': unity.model3_path(),
                                               'ACTION_LAB_POSES': spec},
              run / 'logs' / 'pose-response-272.log')
    print('poses', len(list(output.glob('[0-9][0-9][0-9][0-9].json'))))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
