"""Synchronized request/feasible prediction/final mesh/raw RGB review, all frames."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import imageio_ffmpeg


def raw_video(root):
    count = json.loads((root/'protocol.json').read_text())['frames']-1
    output = root/'stateful-arms-raw.mp4'
    if output.exists():
        raise FileExistsError(output)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-loglevel', 'error', '-framerate', '30',
        '-start_number', '1', '-i', str(root/'capture/%04d.png'), '-frames:v', str(count),
        '-an', '-c:v', 'libx264', '-crf', '16', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)], check=True)
    print(output)


def run(root, fit_requests=False):
    rows = [json.loads(l) for l in (root/'track.jsonl').read_text().splitlines()]
    output = root/('stateful-arms-review-fit.mp4' if fit_requests else 'stateful-arms-review.mp4')
    if fit_requests:
        all_points = np.concatenate([np.array(r[s][key]) for r in rows for s in ('native', 'patch') for key in ('requested', 'ideal')])
        low, high = all_points.min(axis=0), all_points.max(axis=0)
        center = (low+high)/2
        scale = min(460/max(high[0]-low[0], .01), 470/max(high[1]-low[1], .01))
    if output.exists():
        raise FileExistsError(output)
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), '-loglevel', 'error', '-f', 'rawvideo', '-vcodec', 'rawvideo',
           '-pix_fmt', 'rgb24', '-s', '1536x656', '-r', '30', '-i', '-', '-an', '-c:v', 'libx264',
           '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        for row in rows:
            i = row['frame']
            f = json.loads((root/'capture'/f'{i:04}.json').read_text(encoding='utf-8-sig'))
            canvas = Image.new('RGB', (1536, 656), '#eeeeee'); draw = ImageDraw.Draw(canvas)
            active_mode = row[row['active'] if row['active'] != 'both' else 'native'].get('mode', 'legacy')
            draw.text((12, 8), f"{root.name} | {row['time']:.3f}s | active: {row['active']} | {row['stage']} | mode {active_mode} | frame {i}", fill='black')
            draw.text((12, 28), '30 Hz offline SDK. Native=image right/anatomical left. Foreground native distal. Request diagram has fixed independent scale.' if fit_requests else '30 Hz action time, offline SDK. Full failures retained. Foreground native distal presentation.', fill='black')
            for col, title in enumerate(('REQUEST (green) / NEXT FEASIBLE PREDICTION (blue)', 'FINAL SDK MATERIAL NODES (orange)', 'RAW SAME-FRAME IMAGE')):
                draw.text((col*512+8, 52), title, fill='black')
            with Image.open(root/'capture'/f'{i:04}.png') as im:
                raw = im.convert('RGB').resize((512, 512))
                canvas.paste(raw, (512, 76)); canvas.paste(raw, (1024, 76))
            for side in ('native', 'patch'):
                actual = np.array(row[side]['actual'])
                screen = np.array(f[side+'NodeScreen']).reshape(4, 2)
                # Same orthographic camera/scale for request and actual columns.
                affine = np.linalg.lstsq(np.c_[actual, np.ones(4)], screen, rcond=None)[0]
                for key, col, color in [('requested', 0, '#00884a'), ('ideal', 0, '#226bde'), ('actual', 1, '#ed6e13')]:
                    if col == 0 and fit_requests:
                        p = (np.array(row[side][key])-center)*np.array([scale, -scale])+[256, 332]
                        points = [tuple(map(float, xy)) for xy in p]
                    else:
                        p = np.c_[row[side][key], np.ones(4)] @ affine
                        points = [(float(x/2+col*512), float(y/2+76)) for x, y in p]
                    draw.line(points, fill=color, width=2)
                    for x, y in points:
                        draw.ellipse((x-3, y-3, x+3, y+3), outline=color, width=2)
                s = row[side]
                y = 596 if side == 'native' else 623
                text = f"{side}: {s['status']} | wrist {s['wristError']:.4f} | elbow {s['elbowError'] if s['elbowError'] is not None else 0:.4f} | palm {s['palmErrorDeg'] if s['palmErrorDeg'] is not None else 0:.1f}deg"
                if s.get('mode') == 'pass' and s.get('goalCompleted'):
                    reference = s.get('policy', {}).get('referenceNodes')
                    error = np.linalg.norm(np.array(s['actual'])[2]-np.array(reference)[2]) if reference is not None else float('nan')
                    text = f"{side}: PASSED, CONTINUING | moving ref error {error:.4f} | waypoint residual {s['wristError']:.4f}"
                draw.text((8, y), text, fill='black')
                draw.text((660, y), ', '.join(s['reasons'])[:133] or 'within current checks', fill='#9c2424' if s['reasons'] else '#006b3c')
            proc.stdin.write(canvas.tobytes())
        proc.stdin.close(); error = proc.stderr.read(); code = proc.wait()
        if code:
            raise RuntimeError(error.decode(errors='replace'))
    print(output)


if __name__ == '__main__':
    if '--raw' in sys.argv:
        raw_video(Path(sys.argv[1]))
    else:
        run(Path(sys.argv[1]), fit_requests='--fit-requests' in sys.argv)
