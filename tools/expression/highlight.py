"""Single-file fallbacks for review: one combined video and one still contact sheet."""
import json
import math
import subprocess
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import imageio_ffmpeg
from PIL import Image, ImageDraw

from tools.expression import evaluate, pipeline

CELL = (360, 400)
CROP = (250, 80, 900, 800)


def peak_frame(frames, role_map):
    """Frame with the largest lateral hand displacement from the clip's first frame."""
    rendered = evaluate.series(frames, role_map)
    if rendered is None:
        return 0
    base = rendered[0]['right'][0]
    best, index = -1.0, 0
    for position, feature in enumerate(rendered):
        value = abs(feature['right'][0] - base)
        if value > best:
            best, index = value, position
    return index


def contact(run: Path, results: dict, role_map, output: Path):
    rows = sorted(results)
    canvas = Image.new('RGB', (CELL[0] * 3, CELL[1] * len(rows)), (28, 31, 38))
    draw = ImageDraw.Draw(canvas)
    for row, name in enumerate(rows):
        directory = run / 'cases' / name
        frames = [json.loads(path.read_text(encoding='utf-8-sig'))
                  for path in sorted((directory / 'candidate').glob('[0-9][0-9][0-9][0-9].json'))]
        index = peak_frame(frames, role_map)
        entry = results[name]
        for column, (label, folder) in enumerate((('source', 'source'), ('baseline', 'baseline'),
                                                  ('candidate', 'candidate'))):
            path = directory / folder / f'{index:04d}.png'
            if not path.is_file():
                continue
            image = Image.open(path)
            image = image.resize(CELL) if folder == 'source' else image.crop(CROP).resize(CELL)
            canvas.paste(image, (column * CELL[0], row * CELL[1]))
            draw.text((column * CELL[0] + 6, row * CELL[1] + 6), f'{name} · {label} · f{index}',
                      fill=(240, 240, 240))
        comparison = entry['candidate']['comparison']
        draw.text((6, row * CELL[1] + CELL[1] - 18),
                  f"candidate {comparison['status']}  lat {comparison['lateralCorrelation']:.2f}  "
                  f"ratio {comparison['peakRatio']:.2f}", fill=(200, 220, 200))
    canvas.save(output)
    return output


def combined(run: Path, results: dict, output: Path):
    listing = run / 'review' / 'concat.txt'
    lines = []
    for name in sorted(results):
        path = run / 'cases' / name / 'compare.mp4'
        lines.append(f"file '{path.as_posix()}'")
    listing.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg, '-y', '-f', 'concat', '-safe', '0', '-i', str(listing),
                    '-c', 'copy', '-movflags', '+faststart', str(output)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return output


def main(run: Path):
    results = json.loads((run / 'results.json').read_text(encoding='utf-8'))
    calibration = json.loads((run / 'calibration' / 'arm-calibration.json').read_text(encoding='utf-8-sig'))
    role_map = calibration['roles']
    print(contact(run, results, role_map, run / 'review' / 'all-cases-frames.png'))
    print(combined(run, results, run / 'review' / 'all-cases.mp4'))


if __name__ == '__main__':
    main(Path(sys.argv[1]))
