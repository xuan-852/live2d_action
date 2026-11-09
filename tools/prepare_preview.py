"""Prepare an ignored, independent Unity project from explicitly selected local SDK files."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]


def prepare(sdk: Path) -> Path:
    project = ROOT / 'local-assets' / 'preview-unity'
    if project.exists():
        raise FileExistsError(f'Refusing to overwrite existing project: {project}')
    required = ('Core', 'Framework', 'Rendering', 'Editor', 'Plugins')
    for name in (*required, 'LICENSE.md', 'CHANGELOG.md'):
        if not (sdk / name).exists():
            raise FileNotFoundError(sdk / name)
    destination = project / 'Assets' / 'Live2D' / 'Cubism'
    destination.mkdir(parents=True)
    provenance = []
    for name in required:
        shutil.copytree(sdk / name, destination / name)
    for path in sdk.iterdir():
        if path.is_file():
            shutil.copy2(path, destination / path.name)
    for path in destination.rglob('*'):
        if path.is_file():
            provenance.append({'path': path.relative_to(destination).as_posix(),
                               'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    editor = project / 'Assets' / 'ActionLab' / 'Editor'
    editor.mkdir(parents=True)
    shutil.copy2(ROOT / 'preview' / 'unity' / 'LabCapture.cs', editor / 'LabCapture.cs')
    (project / 'Packages').mkdir()
    (project / 'Packages' / 'manifest.json').write_text(json.dumps({'dependencies': {
        'com.unity.modules.imageconversion': '1.0.0',
        'com.unity.modules.animation': '1.0.0',
        'com.unity.modules.audio': '1.0.0',
        'com.unity.modules.physics': '1.0.0',
        'com.unity.modules.jsonserialize': '1.0.0',
        'com.unity.modules.imgui': '1.0.0',
        'com.unity.modules.unitywebrequest': '1.0.0',
        'com.unity.modules.unitywebrequestaudio': '1.0.0',
    }}, indent=2) + '\n', encoding='utf-8')
    (project / 'ProjectSettings').mkdir()
    (project / 'ProjectSettings' / 'ProjectVersion.txt').write_text(
        'm_EditorVersion: 2022.3.62t7\nm_EditorVersionWithRevision: 2022.3.62t7 (139712ad4655)\nm_TuanjieEditorVersion: 1.8.5\n', encoding='utf-8')
    (project / 'sdk-provenance.json').write_text(json.dumps({
        'source': str(sdk.resolve()), 'selection': list(required),
        'excluded': ['Samples', 'all desktop application code and prefabs'],
        'files': provenance}, indent=2) + '\n', encoding='utf-8')
    return project


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk', type=Path, required=True)
    print(prepare(parser.parse_args().sdk))
