"""Two adaptation methods from source trajectories to Fu Xuan parameters.

baseline  : per-joint angle mapped straight onto same-purpose parameters, no relation checks.
candidate : target hand position reconstructed from normalised body relations and solved
            against the measured reach grid; body/head/eyes keep the same relation rules.
"""
import json
import math
from pathlib import Path

import numpy as np

SHOULDER = 'Param94'
SWING = 'Param97'
WRIST = 'Param99'
FINGERS = ('Param110', 'Param111', 'Param112', 'Param113', 'Param114', 'Param115')
BODY_TURN = 'ParamBodyAngleX2'
BODY_LEAN = 'ParamBodyAngleZ2'
HEAD = {'x': 'ParamAngleX', 'y': 'ParamAngleY', 'z': 'ParamAngleZ'}
EYES = {'x': 'ParamEyeBallX', 'y': 'ParamEyeBallY'}


def clamp(value, low, high):
    return max(low, min(high, value))


def load_calibration(path: Path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def scale_of(calibration):
    neutral = calibration['neutral']
    head, hip = neutral['head'], neutral['hip']
    return math.dist(head, hip), neutral['chest']


def solve_arm(target, calibration, chest_scale):
    """Nearest measured (shoulder, swing) pair for a target hand position, plus residual."""
    grid = calibration['grid']
    neutral_chest = calibration['neutral']['chest']
    best = None
    for entry in grid:
        hand = entry['hand']
        distance = math.dist(hand, target)
        if best is None or distance < best[0]:
            best = (distance, entry)
    return best[1]['shoulder'], best[1]['swing'], best[0]


def target_hand(sample, scale_fx, chest_fx, scale_src, gain=1.0):
    return [chest_fx[0] + gain * sample['right'][0] * scale_fx / scale_src,
            chest_fx[1] + gain * sample['right'][1] * scale_fx / scale_src]


def candidate_curves(samples, calibration, fps, reference=None, source_arm='left'):
    """Relation-preserving adaptation.

    The target is the measured neutral hand plus the source hand's motion *relative to a
    reference posture*, scaled by the Fu Xuan torso length and solved on the measured grid.
    Copying absolute source positions would push the hand into the body, because the two
    rigs have different neutral arm postures and arm lengths.
    """
    scale_fx, chest_fx = scale_of(calibration)
    reference = reference or (0.0, 0.0)
    neutral = calibration['neutral']['rightHand']
    curves = {name: [] for name in (SHOULDER, SWING, WRIST, BODY_TURN, BODY_LEAN,
                                    HEAD['x'], HEAD['y'], HEAD['z'], EYES['x'], EYES['y'], 'ParamEyeLOpen')}
    for name in FINGERS:
        curves[name] = []
    residuals = []
    for sample in samples:
        offset = sample[source_arm]
        target = [neutral[0] + (offset[0] - reference[0]) * scale_fx,
                  neutral[1] + (offset[1] - reference[1]) * scale_fx]
        shoulder, swing, residual = solve_arm(target, calibration, scale_fx)
        residuals.append(residual)
        curves[SHOULDER].append(clamp(shoulder, -30, 60))
        curves[SWING].append(clamp(swing, -30, 30))
        curves[WRIST].append(clamp(math.degrees(sample['wristSlope']) * 0.5 - 40, -30, 30))
        curves[BODY_TURN].append(clamp(math.degrees(sample['yaw']) * 0.25, -10, 10))
        curves[BODY_LEAN].append(clamp(math.degrees(sample['headLateral']) * -2.0, -10, 10))
        curves[HEAD['x']].append(clamp(sample['headLateral'] * 40, -30, 30))
        curves[HEAD['y']].append(clamp((offset[1] - reference[1] + 0.4) * -12, -30, 30))
        curves[HEAD['z']].append(clamp(sample['headLateral'] * -12, -30, 30))
        curves[EYES['x']].append(clamp(sample['headLateral'] * 1.6, -1, 1))
        curves[EYES['y']].append(0.0)
        curves['ParamEyeLOpen'].append(1.0)
        openness = clamp((math.hypot(offset[0] - reference[0], offset[1] - reference[1]) - 0.3) * 1.5, 0, 1)
        for name in FINGERS:
            curves[name].append(10.0 + 25.0 * openness)
    return curves, {'meanResidual': float(np.mean(residuals)), 'maxResidual': float(np.max(residuals))}


def baseline_curves(samples, calibration, fps, reference=None, source_arm='left'):
    """Naive joint-angle mapping: same-purpose parameters, fixed gains, no reach check."""
    curves = {name: [] for name in (SHOULDER, SWING, WRIST, BODY_TURN, BODY_LEAN,
                                    HEAD['x'], HEAD['y'], HEAD['z'], EYES['x'], EYES['y'], 'ParamEyeLOpen')}
    for name in FINGERS:
        curves[name] = []
    reference = reference or (0.0, 0.0)
    for sample in samples:
        offset = sample[source_arm]
        elevation = math.degrees(math.atan2(-(offset[1] - reference[1]), abs(offset[0]) + 1e-6))
        curves[SHOULDER].append(clamp(0.6 * elevation - 20, -30, 60))
        curves[SWING].append(clamp(0.5 * math.degrees(sample['wristSlope']) - 10, -30, 30))
        curves[WRIST].append(clamp(-0.3 * math.degrees(sample['wristSlope']), -30, 30))
        curves[BODY_TURN].append(clamp(math.degrees(sample['yaw']) * 0.25, -10, 10))
        curves[BODY_LEAN].append(clamp(math.degrees(sample['headLateral']) * -2.0, -10, 10))
        curves[HEAD['x']].append(clamp(sample['headLateral'] * 40, -30, 30))
        curves[HEAD['y']].append(clamp((offset[1] - reference[1] + 0.4) * -12, -30, 30))
        curves[HEAD['z']].append(clamp(sample['headLateral'] * -12, -30, 30))
        curves[EYES['x']].append(clamp(sample['headLateral'] * 1.6, -1, 1))
        curves[EYES['y']].append(0.0)
        curves['ParamEyeLOpen'].append(1.0)
        for name in FINGERS:
            curves[name].append(30.0)
    return curves, {'meanResidual': None, 'maxResidual': None}


def spec(curves, fps, include_physics=True):
    ids = list(curves)
    channels = [{'values': [float(v) for v in curves[name]]} for name in ids]
    duration = (len(channels[0]['values']) - 1) / fps
    return {'fps': fps, 'duration': duration, 'physics': include_physics, 'excludeX2': True,
            'ids': ids, 'channels': channels}
