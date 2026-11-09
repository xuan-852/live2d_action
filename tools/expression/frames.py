"""Frame comparison sheets for manual inspection of the batch."""
from pathlib import Path
import json
import sys

from PIL import Image, ImageDraw

CROP = (240, 80, 900, 780)


def pick_frames(directory: Path):
    frames = []
    for path in sorted(directory.glob('[0-9][0-9][0-9][0-9].json')):
        frame = json.loads(path.read_text(encoding='utf-8-sig'))
        frames.append((path.stem, frame))
    return frames


def sheet(run: Path, case: str, indices, output: Path):
    directory = run / 'cases' / case
    columns = [('source', directory / 'source'), ('baseline', directory / 'baseline'),
               ('candidate', directory / 'candidate')]
    cell = (CROP[2] - CROP[0], CROP[3] - CROP[1])
    scale = 0.62
    size = (int(cell[0] * scale), int(cell[1] * scale))
    canvas = Image.new('RGB', (len(columns) * size[0], len(indices) * (size[1] + 22)), (30, 33, 40))
    draw = ImageDraw.Draw(canvas)
    for row, index in enumerate(indices):
        for column, (label, folder) in enumerate(columns):
            path = folder / f'{index:04d}.png'
            if not path.is_file():
                continue
            image = Image.open(path)
            image = image.crop(CROP) if label != 'source' else image.resize(cell)
            canvas.paste(image.resize(size), (column * size[0], row * (size[1] + 22) + 22))
            draw.text((column * size[0] + 6, row * (size[1] + 22) + 5),
                      f'{label} f{index}', fill=(235, 235, 235))
    canvas.save(output)
    return output


if __name__ == '__main__':
    run = Path(sys.argv[1])
    case = sys.argv[2]
    indices = [int(v) for v in sys.argv[3].split(',')]
    print(sheet(run, case, indices, run / 'review' / f'{case}-frames.png'))
