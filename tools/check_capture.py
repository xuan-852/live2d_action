"""Check baseline pairing and uniform action times, not visual motion quality."""
import argparse
import hashlib
import json
from pathlib import Path


def check(root, fps=30):
    root = Path(root)
    if fps <= 0:
        raise ValueError('fps must be positive')
    if not (root / 'complete.txt').is_file():
        raise ValueError('Capture did not complete')
    paths = sorted(root.glob('[0-9][0-9][0-9][0-9].json'))
    if not paths:
        raise ValueError('No frame states')
    hashes = []
    states = []
    for index, path in enumerate(paths):
        state = json.loads(path.read_text(encoding='utf-8-sig'))
        if path.stem != f'{index:04d}' or state['frame'] != index:
            raise ValueError('Missing or mismatched frame index')
        if abs(state['actionTime'] - index / fps) > 1e-9:
            raise ValueError('Nonuniform action time')
        image = path.with_suffix('.png').read_bytes()
        if not image.startswith(b'\x89PNG\r\n\x1a\n'):
            raise ValueError('Not PNG')
        hashes.append(hashlib.sha256(image).hexdigest())
        if state['requested'] != state['applied']:
            raise ValueError('Requested and applied parameters differ')
        if not state['drawables'] or not state['parts']:
            raise ValueError('Missing model state')
        states.append(state)
    return {'frames': len(paths), 'fps': fps, 'pngHashes': hashes,
            'identicalImages': len(set(hashes)) == 1,
            'identicalDrawableStates': all(s['drawables'] == states[0]['drawables'] for s in states),
            'parameters': len(states[0]['applied']), 'drawables': len(states[0]['drawables']),
            'scope': 'pairing, requested/applied equality, action-time grid, baseline repeatability only'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--fps', type=int, default=30)
    args = parser.parse_args()
    print(json.dumps(check(args.root, args.fps), indent=2))
