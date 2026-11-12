"""One existing MMD route through actual-feedback control; never retime the source."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key] = '1'
import sys, json, hashlib, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from tools.control.mmd_arm_ceiling import target_chain
from tools.control.mmd_virtual_tracker import signals
from tools.control.calibrate_virtual_tracker import calibrate
from tools.control.mmd_two_experiments import LIMITS
from tools.control.stateful_arm import DT, bend, Target
from tools.control.timed_arm import heading
from tools.control.validate_stateful_arm import run, APPROVED
from tools.control.pilot import Plant
from tools.control.native_target import NATIVE_ROUTE_LIMITS
from tools.expression import unity


def make_stream(doc, homes):
    times, raw, _, _ = signals(doc)
    if len(times) < 2 or abs(times[0]) > 1e-8:
        raise ValueError('Expected source starting at zero with at least two samples')
    hb = np.clip(calibrate(raw), -LIMITS, LIMITS)
    chains = {}
    for side, identity in [('native','anatomicalLeft'),('patch','anatomicalRight')]:
        frames = doc['sides'][identity]['frames']
        if len(frames) != len(times) or not np.allclose([f['seconds'] for f in frames], times):
            raise ValueError('MMD arm clocks differ')
        chains[side] = np.array([target_chain(homes[side], f['target']) for f in frames])
    def target(nodes, clock, mode, velocity=None, rate=0.):
        value = dict(wrist=nodes[2].tolist(), elbow=nodes[1].tolist(),
                     palm_heading=heading(nodes), branch=int(np.sign(bend(nodes))),
                     time=clock, mode=mode, angular_velocity=float(rate))
        if velocity is not None:
            value['velocity'] = velocity.tolist()
        Target(**value)  # Validate the public interface before launching Unity.
        return value
    packets = []
    # Explicit outside-source acquisition/return, never hide them in source time.
    for i in range(30):
        packets.append(dict(stage='mmd-acquire', sourceTime=None, headBody=hb[0].tolist(),
            targets={s:target(chains[s][0], 1., 'hold') for s in homes}))
    # SDK feedback is an executed interval difference, not a central derivative
    # at the sample center. Match its time convention; acquisition ends at p[0].
    velocities = {s:np.vstack([np.zeros(2),np.diff(p[:,2],axis=0)/DT]) for s,p in chains.items()}
    rates = {s:np.r_[0.,np.diff(np.degrees(np.unwrap(np.radians([heading(x) for x in p]))))/DT] for s,p in chains.items()}
    for j, t in enumerate(times):
        packets.append(dict(stage='mmd-source', sourceTime=float(t), headBody=hb[j].tolist(),
            targets={s:target(chains[s][j], (len(packets)+1)*DT, 'track', velocities[s][j], rates[s][j]) for s in homes}))
    deadline = (len(packets)+30)*DT
    for i in range(60):
        packets.append(dict(stage='mmd-return' if i<30 else 'mmd-final-hold', sourceTime=None,
            headBody=(hb[-1]*max(0.,1-(i+1)/30)).tolist(),
            targets={s:target(homes[s], deadline, 'hold') for s in homes}))
    return dict(frames=packets, metadata=dict(sourceFile=doc['sourceFile'], sourceSHA=doc['sourceSha256'],
        sourceFrames=len(times), sourceTimes=times.tolist(), sourceOffsetSeconds=31*DT,
        mapping='historical absolute projected headings and model neutral lengths times UNCLIPPED source projection ratios; fixed roots',
        limitations=['fixed root omits source shoulder translation','identity projection execution does not provide independent shortening',
                     '2D palm heading is not 3D palm normal','native local response sees requested head/body and last applied SDK context, not future physics'],
        outsideSource='30 acquire + original source samples + 30 return + 30 hold; no source retiming or sample dropping',
        headBody='unchanged frozen calibrate(raw), original limits, original SDK physics'))


def execute(root, source, native_five=True, native_limits=None):
    if native_five and native_limits is None:
        native_limits=NATIVE_ROUTE_LIMITS.copy()
    doc = json.loads(source.read_text(encoding='utf-8-sig'))
    plant = Plant(); plant.model.set({}); a = plant.anchors.read()
    homes = dict(native=np.array([a[k] for k in ('shoulder','elbow','wrist','palm')]),
        patch=np.array(json.loads((APPROVED/'protocol.json').read_text())['newNodes']))
    stream = make_stream(doc, homes)
    root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, root/'source-map.json')
    shutil.copy2(__file__, root/'mmd_timed_controller.py')
    unity.write_spec(root/'acceptance-protocol.json', dict(source=str(source), sourceMapSHA=hashlib.sha256(source.read_bytes()).hexdigest(),
        sourceSamples='all at original 30Hz; no removed transition samples',
        tracking=dict(wrist=.005, elbow=.005, palmDegrees=4),
        safety=dict(nodeSpeed=.72,nodeAcceleration=16.2,parameterSpeed=135,parameterAcceleration=1620,minArea=.5,flips=0,pairing=1e-6,restore=1e-6),
        terminal=dict(final15Frames='position+direction AND max node speed .02 / palm rate 8'),
        motionRetention='report source vs actual path/heading ranges, do not infer fidelity from low error or smoothness alone',
        visualAcceptance='pending for this MMD; prior point-task approval not transferred'))
    if native_five:
        protocol=json.loads((root/'acceptance-protocol.json').read_text())
        protocol['nativeSafety']=native_limits
        protocol['nativeBackend']='calibrated five-axis; finite candidate checked in actual node space'
        protocol['velocityConvention']='executed source sample interval; first source velocity zero after acquisition'
        unity.write_spec(root/'acceptance-protocol.json',protocol)
    run(root/'sdk', physics=True, request_stream=stream, native_five=native_five, native_limits=native_limits)


if __name__ == '__main__':
    execute(Path(sys.argv[1]), Path(sys.argv[2]),native_five='--legacy-native' not in sys.argv)
