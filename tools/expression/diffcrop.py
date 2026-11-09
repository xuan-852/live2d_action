"""Diff-driven crops: locate what a pose actually changes, then show neutral vs pose."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def diff_box(a: Image.Image, b: Image.Image, margin=40):
    array = np.abs(np.asarray(a.convert('RGB'), dtype=np.int16) - np.asarray(b.convert('RGB'), dtype=np.int16))
    mask = array.max(axis=2) > 6
    if not mask.any():
        return None, 0
    ys, xs = np.where(mask)
    box = (max(0, xs.min() - margin), max(0, ys.min() - margin),
           min(a.width, xs.max() + margin), min(a.height, ys.max() + margin))
    return box, int(mask.sum())


def compare(directory: Path, neutral_frame: str, entries, output: Path, scale=2):
    neutral = Image.open(directory / f'{neutral_frame}.png')
    cells = []
    for frame, label in entries:
        image = Image.open(directory / f'{frame}.png')
        box, changed = diff_box(neutral, image)
        if box is None:
            cells.append((label, neutral.crop((352, 120, 672, 620)), 'no change'))
            continue
        cells.append((label, neutral.crop(box), image.crop(box)))
    width = max(c[1].width for c in cells) * scale
    height = sum(max(c[1].height, c[2].height if isinstance(c[2], Image.Image) else 0) * scale + 22 for c in cells)
    canvas = Image.new('RGB', (2 * width + 12, height), (32, 36, 44))
    draw = ImageDraw.Draw(canvas)
    y = 0
    for cell in cells:
        label = cell[0]
        draw.text((4, y + 4), label, fill=(240, 240, 240))
        y += 22
        if cell[2] == 'no change':
            canvas.paste(cell[1].resize((cell[1].width * scale, cell[1].height * scale)), (0, y))
            y += cell[1].height * scale
            continue
        for column, image in enumerate((cell[1], cell[2])):
            canvas.paste(image.resize((image.width * scale, image.height * scale)), (column * (width + 12), y))
        y += max(cell[1].height, cell[2].height) * scale
    canvas.save(output)
    return output
