"""Re-encode review videos from captured PNG frames (no Unity rerun)."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.expression import pipeline


def main(run: Path):
    results = json.loads((run / 'results.json').read_text(encoding='utf-8'))
    for name, entry in sorted(results.items()):
        directory = run / 'cases' / name
        count = entry['candidate']['frames']
        pipeline.encode_triple(directory / 'source', directory / 'baseline', directory / 'candidate',
                               count, directory / 'compare.mp4')
        print(name, count)


if __name__ == '__main__':
    main(Path(sys.argv[1]))
