"""Post-batch checks: protected files, leftovers, resource picture, timing."""
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EDITOR = 'Tuanjie.exe'


def protected(run: Path, intended=('README.md', 'preview/README.md')):
    before = json.loads((run / 'protected-before.json').read_text(encoding='utf-8-sig'))
    rows = []
    for record in before:
        path = ROOT / record['path']
        if not path.is_file():
            rows.append({'path': record['path'], 'state': 'missing'})
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest().upper()
        if digest == record['sha256'].upper():
            state = 'unchanged'
        elif record['path'] in intended:
            state = 'modified on purpose by this batch (documentation)'
        else:
            state = 'modified unexpectedly'
        rows.append({'path': record['path'], 'state': state, 'bytesBefore': record['bytes'],
                     'bytesNow': path.stat().st_size})
    return rows


def processes():
    try:
        output = subprocess.run(['tasklist', '/FI', f'IMAGENAME eq {EDITOR}'], capture_output=True,
                                text=True, timeout=30).stdout
    except Exception as error:  # pragma: no cover - environment dependent
        return {'error': str(error)}
    return {'unityRunning': EDITOR.lower() in output.lower(), 'raw': output.strip().splitlines()[:4]}


def disk(run: Path):
    total = 0
    files = 0
    for path in run.rglob('*'):
        if path.is_file():
            total += path.stat().st_size
            files += 1
    return {'files': files, 'bytes': total, 'megabytes': round(total / 1e6, 1)}


def machine():
    return {'cpuCount': os.cpu_count(), 'python': os.sys.version.split()[0]}


if __name__ == '__main__':
    import sys
    run = Path(sys.argv[1])
    report = {'protected': protected(run), 'processes': processes(), 'disk': disk(run), 'machine': machine()}
    (run / 'verification.json').write_text(json.dumps(report, ensure_ascii=True, indent=1), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True, indent=1)[:2000])
