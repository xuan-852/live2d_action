"""Apply the preregistered direct-control gates; never infer visual acceptance."""
import json
import sys
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.expression.unity import write_spec


def assess(root):
    protocol = json.loads((root/'protocol.json').read_text())
    rows = [json.loads(line) for line in (root/'track.jsonl').read_text().splitlines()]
    if not (root/'capture/complete.txt').exists() or len(rows)+1 != protocol['frames']:
        raise ValueError('incomplete SDK episode')
    if [r['frame'] for r in rows] != list(range(1, protocol['frames'])):
        raise ValueError('noncontiguous frame pairing')
    gates = protocol['criteria']; results = []
    # Use the same measured terminal criterion for old and new controllers.
    if 'terminalCriteria' in protocol:
        first = json.loads((root/'capture/0000.json').read_text(encoding='utf-8-sig'))
        previous_heading = {}
        for side in ('native', 'patch'):
            p = first['wholeArms'][side+'Chain']
            previous_heading[side] = math.degrees(math.atan2(p[7]-p[5], p[6]-p[4]))
        for r in rows:
            for side in ('native', 'patch'):
                p = r[side]['actual']; angle = math.degrees(math.atan2(p[3][1]-p[2][1], p[3][0]-p[2][0]))
                rate = abs((angle-previous_heading[side]+180)%360-180)*30
                r[side]['commonHoldSettled'] = r[side]['metrics']['maxNodeSpeed'] <= .02 and rate <= 8.
                previous_heading[side] = angle
    for side in ('native', 'patch'):
        gates = protocol.get('criteriaOverrides', {}).get(side, protocol['criteria'])
        for stage in dict.fromkeys(r['stage'] for r in rows if r['active'] in (side, 'both')):
            selected = [r for r in rows if r['active'] in (side, 'both') and r['stage'] == stage]
            def met(record):
                s = record[side]
                return (s['wristError'] <= gates['wristError'] and
                        (s['elbowError'] is None or s['elbowError'] <= gates['elbowError']) and
                        (s['palmErrorDeg'] is None or s['palmErrorDeg'] <= gates['palmErrorDeg']) and
                        (s.get('mode') != 'hold' or s.get('commonHoldSettled', True)))
            continuity = all(r[side]['metrics']['maxNodeSpeed'] <= gates['nodeSpeed']+.001 and
                r[side]['metrics']['maxNodeAcceleration'] <= gates['nodeAcceleration']+.001 and
                r[side]['metrics']['maxParameterStep'] <= gates['parameterSpeed']/30+.001 and
                r[side]['metrics']['maxParameterAcceleration'] <= gates['parameterAcceleration']+.001 for r in selected)
            geometry = all(r['geometry']['minTriangleAreaRatio'] >= gates['minTriangleAreaRatio'] and
                           r['geometry']['flippedTriangles'] == 0 for r in selected)
            pairing = all(r['pairing'] <= gates['pairing'] and r['geometry']['restoredMeshError'] <= gates['restore'] for r in selected)
            windows = {}
            if stage in ('lift-hold-lower', 'visible-lift-hold-lower'):
                for name in ('liftHold', 'lowerHold'):
                    lo, hi = protocol['evaluationWindows'][name]
                    windows[name] = all(met(r) for r in selected if lo <= r['localFrame'] < hi)
            elif stage == 'point-reach-return':
                for name in ('pointHold', 'pointReturnHold'):
                    lo, hi = protocol['evaluationWindows'][name]
                    windows[name] = all(met(r) for r in selected if lo <= r['localFrame'] < hi)
                windows['allRequestedDeadlinesMet'] = all(met(r) for r in selected if r[side]['now'] >= r[side]['deadline'])
            elif stage == 'unreachable-return':
                windows['returnLast15Frames'] = all(met(r) for r in selected[-15:])
                windows['unreachableReported'] = any('outside-local-fixed-length-reach-envelope' in r[side]['reasons'] for r in selected)
            elif stage == 'moving-target-switch':
                windows['deadlineMissesReported'] = all(met(r) or r[side]['now'] < r[side]['deadline'] or 'deadline-missed' in r[side]['reasons'] for r in selected)
                windows['allRequestedDeadlinesMet'] = all(met(r) for r in selected if r[side]['now'] >= r[side]['deadline'])
            elif stage == 'pass-then-hold':
                windows['passEventWithVelocity'] = any(r[side].get('mode') == 'pass' and r[side].get('goalCompleted') and r[side]['wristVelocityError'] <= .02 for r in selected)
                windows['continuedAfterPoint'] = all(r[side]['wristVelocityError'] <= .02 for r in selected if 30 <= r['localFrame'] < 42)
                windows['finalHold'] = all(met(r) for r in selected[-15:])
            elif stage in ('overhead-hold','neutral-return'):
                windows['finalHold'] = all(met(r) for r in selected[-15:])
            else:
                windows['allTrackingFrames'] = all(met(r) for r in selected)
            safety_and_reporting = continuity and geometry and pairing and not any(r[side]['status'] == 'failed' for r in selected)
            task_met = all(windows.values())
            results.append(dict(side=side, stage=stage, continuity=continuity, geometry=geometry, pairing=pairing,
                safetyAndReporting=safety_and_reporting, taskGates=windows, taskMet=task_met,
                maxWristError=max(r[side]['wristError'] for r in selected),
                maxElbowError=max(r[side]['elbowError'] or 0 for r in selected),
                maxPalmErrorDeg=max(r[side]['palmErrorDeg'] or 0 for r in selected),
                deadlineMissedFrames=sum('deadline-missed' in r[side]['reasons'] for r in selected),
                maxConsecutiveLimitedFrames=max(r[side]['limitedFrames'] for r in selected),
                meshFloorFrames=sum('mesh-area-floor' in r[side]['reasons'] for r in selected),
                branchChangeFrames=[r['frame'] for r in selected if 'branch-change' in r[side]['reasons']],
                visualAcceptance='unverified; user review required; connection and support feeling not inferred'))
    write_spec(root/'assessment.json', dict(results=results, visualAcceptance=False))
    for r in results:
        print(r['side'], r['stage'], 'safety', r['safetyAndReporting'], r['taskGates'])
    return results


if __name__ == '__main__':
    assess(Path(sys.argv[1]))
