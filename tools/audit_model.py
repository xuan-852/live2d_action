"""Audit local model hashes and model3 dependencies without loading Unity assets."""
import argparse
import hashlib
import json
from pathlib import Path


def references(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from references(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            if key not in ('Name', 'FadeInTime', 'FadeOutTime'):
                yield from references(item)


def audit(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8-sig'))
    errors = []
    variants = {}
    for entry in manifest['files']:
        path = (root / entry['variant'] / entry['path']).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Manifest path escapes resource directory')
        if not path.is_file():
            errors.append(f'Missing: {entry["variant"]}/{entry["path"]}')
            continue
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if len(data) != entry['bytes'] or digest != entry['sha256']:
            errors.append(f'Integrity mismatch: {entry["variant"]}/{entry["path"]}')
        variants.setdefault(entry['variant'], {})[entry['path']] = digest
    models = []
    for path in root.rglob('*.model3.json'):
        model = json.loads(path.read_text(encoding='utf-8-sig'))
        refs = list(references(model['FileReferences']))
        for ref in refs:
            target = (path.parent / ref).resolve()
            if not target.is_relative_to(root) or not target.is_file():
                errors.append(f'Invalid reference: {path.relative_to(root)} -> {ref}')
        models.append({'model': path.relative_to(root).as_posix(), 'references': refs})
    a, b = variants.get('unity-model', {}), variants.get('streaming-model', {})
    differences = sorted(p for p in a.keys() | b.keys() if a.get(p) != b.get(p))
    return {'fileCount': len(manifest['files']), 'errors': errors,
            'variantDifferences': differences, 'models': models,
            'editableSources': [p.relative_to(root).as_posix() for p in root.rglob('*')
                                if p.suffix.lower() in ('.cmo3', '.psd', '.psb', '.can3')]}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    print(json.dumps(result, ensure_ascii=True, indent=2))
    raise SystemExit(bool(result['errors']))
