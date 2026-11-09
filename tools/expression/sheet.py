"""Build contact sheets from a pose-lab run, grouped by parameter."""
from pathlib import Path
import sys

from PIL import Image, ImageDraw

CROP = (352, 120, 672, 620)


def frames(directory: Path):
    result = {}
    for label in sorted(directory.glob('[0-9][0-9][0-9][0-9].label.txt')):
        text = label.read_text(encoding='utf-8').strip()
        result.setdefault(text.split()[0], []).append((label.stem.split('.')[0], text))
    return result


def sheet(directory: Path, parameters, output: Path, columns=2, crop=CROP):
    table = frames(directory)
    rows = []
    for parameter in parameters:
        entries = table.get(parameter, [])
        if len(entries) >= 2:
            rows.append((parameter, [entries[0], entries[-1]]))
        elif entries:
            rows.append((parameter, [entries[0]]))
    if not rows:
        raise SystemExit('no matching frames')
    cell_w, cell_h = crop[2] - crop[0], crop[3] - crop[1]
    scale = 0.9
    w, h = int(cell_w * scale), int(cell_h * scale)
    canvas = Image.new('RGB', (columns * w, len(rows) * (h + 24)), (32, 36, 44))
    draw = ImageDraw.Draw(canvas)
    for row_index, (parameter, entries) in enumerate(rows):
        for column, (frame, label) in enumerate(entries[:columns]):
            image = Image.open(directory / f'{frame}.png').crop(crop).resize((w, h))
            canvas.paste(image, (column * w, row_index * (h + 24) + 24))
            draw.text((column * w + 6, row_index * (h + 24) + 6), label, fill=(240, 240, 240))
    canvas.save(output)
    return output


if __name__ == '__main__':
    directory = Path(sys.argv[1])
    parameters = sys.argv[2].split(',')
    sheet(directory, parameters, Path(sys.argv[3]))
