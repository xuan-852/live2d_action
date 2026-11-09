"""Unity batch driver for this repository's ignored experiment project."""
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / 'local-assets' / 'preview-unity'
EDITOR = Path('D:/Unity/editor/2022.3.62t7/Editor/Tuanjie.exe')
SCRIPTS = ('LabCapture.cs', 'LabMotion.cs', 'RecoveredArmGeometry.cs',
           'RecoveredShoulderConnection.cs', 'RecoveredArmTypes.cs', 'WholeArmSession.cs')


def model3_path() -> Path:
    matches = sorted((ROOT / 'local-assets' / 'fuxuan' / 'unity-model').glob('*.model3.json'))
    if len(matches) != 1:
        raise RuntimeError(f'Expected one model3.json, found {matches}')
    return matches[0]


def sync_scripts():
    destination = PROJECT / 'Assets' / 'ActionLab' / 'Editor'
    destination.mkdir(parents=True, exist_ok=True)
    synced = []
    for name in SCRIPTS:
        source = ROOT / 'preview' / 'unity' / name
        target = destination / name
        if not target.is_file() or source.read_bytes() != target.read_bytes():
            shutil.copy2(source, target)
            synced.append(name)
    return synced


def run(entry, output: Path, env: dict, log: Path, timeout=900):
    """Run one Unity batch task. Never overwrites an existing output directory."""
    if output.exists():
        raise FileExistsError(output)
    sync_scripts()
    environment = dict(os.environ)
    environment.update({k: str(v) for k, v in env.items()})
    environment['ACTION_LAB_OUTPUT'] = str(output)
    arguments = [str(EDITOR), '-batchmode', '-projectPath', str(PROJECT),
                 '-executeMethod', entry, '-logFile', str(log)]
    completed = subprocess.run(arguments, env=environment, timeout=timeout,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise RuntimeError(f'Unity exit {completed.returncode}; see {log}')
    if not (output / 'complete.txt').is_file():
        raise RuntimeError(f'Missing completion marker in {output}')
    return completed.returncode


def write_spec(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=1), encoding='utf-8')
    return path
