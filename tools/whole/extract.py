"""Explicit one-time extraction of minimal, reviewed local legacy components.

Not imported by playback or batch adaptation. Old tree is only an input to this audit.
"""
from pathlib import Path
import hashlib
import json
import re


def main():
    base = Path('D:/Unity/projects/Desktop_per_pro')
    emb = base / 'code/desktop_unity/Assets/Scripts/Embodied'
    foundation = base / 'Build/w/hp/scripts/live2d/retarget-foundation'
    dest = Path('tools/whole/recovered')
    dest.mkdir(parents=True, exist_ok=False)
    records = []

    def save(src, target, text, selection):
        if target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text.rstrip() + '\n', encoding='utf-8')
        records.append({'source': str(src), 'sourceSha256': hashlib.sha256(src.read_bytes()).hexdigest(),
                        'target': target.as_posix(), 'targetSha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                        'selection': selection,
                        'license': 'User project implementation; no standalone license identified. Local research only; no redistribution permission inferred.'})

    for name in ('core_model.py', 'anchors.py'):
        src = foundation.parent / 'retarget' / name
        text = src.read_text(encoding='utf-8').replace('from core_model import Model', 'from .core_model import Model')
        save(src, dest / name, text, 'Complete small pure Core/anchor module; relative import only')
    src = foundation / 'backends.py'
    text = src.read_text(encoding='utf-8')
    save(src, dest / 'native.py', text[:text.index('class PatchArm:')],
         'NativeArm and pure helpers only; PatchArm excluded')
    src = foundation / 'calibration/model-arm-calibration.json'
    save(src, dest / 'calibration.json', src.read_text(encoding='utf-8'),
         'Versioned calibration candidate; remeasure against local Core')
    src = emb / 'SecondaryArmPatch.cs'
    lines = src.read_text(encoding='utf-8').splitlines()
    text = ('using UnityEngine;\nnamespace ActionLab.Recovered {\npublic static class SandboxArmGeometry {\n'
            + '\n'.join(lines[20:59]) + '\n' + '\n'.join(lines[417:479]) + '\n}\n}\n')
    save(src, Path('preview/unity/RecoveredArmGeometry.cs'), text,
         'TargetIds/cloth exclusions/rest anchors/candidate bounds/pure deformation only; no playback/coordinator/IO')
    src = emb / 'ShoulderConnection.cs'
    text = src.read_text(encoding='utf-8').replace('SecondaryArmPatch', 'SandboxArmGeometry')
    text = text.replace('public sealed class ShoulderConnection',
                        'namespace ActionLab.Recovered {\npublic sealed class ShoulderConnection') + '\n}\n'
    save(src, Path('preview/unity/RecoveredShoulderConnection.cs'), text,
         'Pure material attachment and observations; geometry class renamed')
    src = emb / 'SecondaryArmMotion.cs'
    lines = src.read_text(encoding='utf-8').splitlines()
    text = '\n'.join(lines[167:186] + lines[200:215])
    text = re.sub(r'\[JsonProperty\([^\]]*\)\]\s*', '', text).replace('[JsonIgnore] ', '')
    save(src, Path('preview/unity/RecoveredArmTypes.cs'),
         'using UnityEngine;\nnamespace ActionLab.Recovered {\n' + text + '\n}\n',
         'Projection and girdle values only; no Newtonsoft or motion contract')
    (dest.parent / '__init__.py').write_text('"""Independent whole-performance experiment."""\n', encoding='utf-8')
    (dest / '__init__.py').write_text('"""Audited minimal legacy components; see provenance.json."""\n', encoding='utf-8')
    (dest / 'provenance.json').write_text(json.dumps(records, indent=2) + '\n', encoding='utf-8')
    print('Extracted', len(records), 'components; no legacy runtime imports')


if __name__ == '__main__':
    main()
