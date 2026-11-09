"""Calibrate the two 7-link arm rotation chains by isolated pose probes."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.expression import unity

CHAINS = ('ArtMesh272', 'ArtMesh249')


def poses():
    result = [{'label': 'neutral', 'ids': [], 'values': []}]
    for chain in CHAINS:
        ids = [f'Param_Angle_Rotation_{link}_{chain}' for link in range(1, 8)]
        for angle in (10, 20, 30, -10, -20, -30):
            result.append({'label': f'{chain} uniform {angle:+d}', 'ids': ids, 'values': [float(angle)] * 7})
        for link in (1, 2, 4, 5, 7):
            for angle in (30, -30):
                result.append({'label': f'{chain} link{link} {angle:+d}', 'ids': [ids[link - 1]],
                               'values': [float(angle)]})
    return result


def main(run: Path):
    spec = run / 'protocol' / 'pose-calibration.json'
    unity.write_spec(spec, {'poses': poses(), 'physics': False})
    output = run / 'calibration' / 'pose-lab'
    unity.run('LabMotion.RunPoseLab', output, {'ACTION_LAB_MODEL': unity.model3_path(),
                                               'ACTION_LAB_POSES': spec},
              run / 'logs' / 'pose-lab.log')
    labels = [(p.stem, (p.parent / (p.stem + '.label.txt')).read_text(encoding='utf-8'))
              for p in sorted(output.glob('[0-9][0-9][0-9][0-9].json'))]
    print(json.dumps({'poses': len(labels), 'first': labels[:4], 'last': labels[-2:]}, ensure_ascii=True, indent=1))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
