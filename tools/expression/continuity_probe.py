"""Frozen K01 ablation, same targets and arm-only execution for two solver versions."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '1'

import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import imageio_ffmpeg
from tools.expression import adapter, source, unity
from tools.expression.trajectory import ResponseGrid, fit


def stats(values):
    delta = np.diff(values, axis=0)
    return {'maxStep': float(np.max(np.abs(delta))),
            'maxRatePerSecond': float(np.max(np.abs(delta)) * 30),
            'span': np.ptp(values, axis=0).tolist(),
            'maxSecondDifference': float(np.max(np.abs(np.diff(delta, axis=0))))}


def execute(run):
    previous = Path('E:/FuXuanLab/runs/20261107-action-lab-expression-01')
    specdir = run / 'specs'
    specdir.mkdir(exist_ok=False)
    (run / 'logs').mkdir()
    cal_bytes = (previous / 'calibration/arm-calibration.json').read_bytes()
    (run / 'calibration.json').write_bytes(cal_bytes)
    calibration = json.loads(cal_bytes)
    protocol = json.loads((previous / 'protocol/protocol.json').read_text())
    cases = ['novel-01', 'novel-12']
    config = {'cases': cases, 'fit': ['novel-01'], 'heldout': ['novel-12'],
              'reference': protocol['referenceOffset'], 'fps': 30, 'durationLimit': 6,
              'iterations': 40, 'velocityWeight': .4, 'accelerationWeight': 1.5,
              'scope': 'K01 continuity ablation; not expression/naturalness acceptance',
              'controls': 'both arm Param94/97 only; same source targets; other parameters neutral + same SDK physics',
              'frozenAt': datetime.datetime.now().astimezone().isoformat()}
    unity.write_spec(run / 'protocol.json', config)
    scale, _ = adapter.scale_of(calibration)
    grid = ResponseGrid(calibration['grid'])
    summary = {}
    for name in cases:
        record = next(r for r in protocol['cases'] if r['id'] == name)
        raw = Path(record['path']).read_bytes()
        sourcepath = run / (name + '-source.json')
        sourcepath.write_bytes(raw)
        trajectory = source.load(sourcepath)
        times, samples = source.series(trajectory, fps=30, limit=6)
        target = np.array(calibration['neutral']['rightHand']) + (
            np.array([sample['left'] for sample in samples]) - config['reference']) * scale
        old = np.array([adapter.solve_arm(t, calibration, scale)[:2] for t in target])
        fitted, detail = fit(target, grid, scale, iterations=config['iterations'])
        detail['targets'] = target.tolist()
        unity.write_spec(run / (name + '-fit.json'), detail)
        report = {'sourceSha256': hashlib.sha256(raw).hexdigest(), 'frames': len(times),
                  'sourceSplit': record['split'], 'variants': {}}
        for label, values in (('discrete', old), ('sequence', fitted)):
            curves = {'Param94': values[:, 0].tolist(), 'Param97': values[:, 1].tolist()}
            spec = unity.write_spec(specdir / (name + '-' + label + '.json'), adapter.spec(curves, 30))
            output = run / (name + '-' + label)
            unity.run('LabMotion.RunMotion', output,
                      {'ACTION_LAB_MODEL': unity.model3_path(), 'ACTION_LAB_CURVES': spec},
                      run / 'logs' / (name + '-' + label + '.log'))
            frames = [json.loads(p.read_text(encoding='utf-8-sig'))
                      for p in sorted(output.glob('[0-9][0-9][0-9][0-9].json'))]
            actual = []
            for frame in frames:
                geom = {d['id']: d for d in frame['drawables']}
                ids = calibration['roles']['rightHand']
                actual.append([np.mean([geom[i]['cx'] for i in ids]), np.mean([geom[i]['cy'] for i in ids])])
            actual = np.array(actual)
            applied = [{p['id']: p['value'] for p in f['applied']} for f in frames]
            error = np.linalg.norm(actual - target, axis=1) / scale
            report['variants'][label] = {'control': stats(values),
                'actualHand': stats(actual / scale),
                'meanTargetErrorTorsoUnits': float(np.mean(error)),
                'maxTargetErrorTorsoUnits': float(np.max(error)),
                'maxPairingDelta': max(f['pairingMaxDelta'] for f in frames),
                'writeError': float(max(abs(applied[k][pid] - curves[pid][k])
                                       for k in range(len(frames)) for pid in curves)),
                'unknown': ['elbow route', 'palm direction', 'naturalness', 'expression preservation']}
        # Preserve aspect ratio and the entire original image, including hands.
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-threads', '1',
            '-framerate', '30', '-i', str(run / (name+'-discrete') / '%04d.png'),
            '-framerate', '30', '-i', str(run / (name+'-sequence') / '%04d.png'),
            '-filter_complex_threads', '1', '-filter_complex',
            '[0:v]scale=512:512[a];[1:v]scale=512:512[b];[a][b]hstack=inputs=2[v]',
            '-map', '[v]', '-c:v', 'libx264', '-threads', '1', '-crf', '18',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(run / (name+'-compare.mp4'))],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        summary[name] = report
        unity.write_spec(run / 'results.json', summary)
        print(name, json.dumps(report['variants']), flush=True)
    unity.write_spec(run / 'finished.json', {'time': datetime.datetime.now().astimezone().isoformat()})


if __name__ == '__main__':
    execute(Path(sys.argv[1]))
