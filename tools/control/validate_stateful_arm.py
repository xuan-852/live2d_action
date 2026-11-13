"""Fixed direct-control acceptance fixtures, atomic SDK feedback, full failures.

Fixture parameters only construct node requests; controller never receives them.
These are tests of lower control, not programmed substitutes for upper learning.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '1'
import sys
import json
import time
import threading
import shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from tools.control.stateful_arm import ArmController, Target, patch_forward, bend, DT, speed_limits, DEFAULT_PROFILE
from tools.control.timed_arm import TimedArmController
from tools.control.native_target import NativeTargetController, NATIVE_KEYS
from tools.control.patch_projection import PatchProjectionController,PatchProjectionPlant,PATCH_KEYS,PALM_PLANE_KEYS
from tools.control.pilot import Plant
from tools.expression import unity
from tools.whole.batch import markers

KEYS = ['Param94', 'Param97', 'Param99']
APPROVED = Path('E:/FuXuanLab/runs/20261109-wrist-axis-revision-01')
STAGES = [('lift-hold-lower', 210), ('wrist-positive-negative', 120),
          ('continuous-route', 180), ('moving-target-switch', 90), ('unreachable-return', 180)]
GEOMETRY_KEYS = ('minTriangleAreaRatio', 'areaGradient', 'flippedTriangles', 'meshAreaRatios', 'meshAreaGradients')


def pose(q):
    ratios=np.array([1.,1.,1.])
    if len(q)>3:
        ratios=1+np.asarray(q[3:6])/100;lo=np.array([.75,.55,.4])
        if np.any(ratios<lo-1e-6) or np.any(ratios>1+1e-6):raise ValueError('Patch actuator projection out of bounds')
        ratios=np.clip(ratios,lo,1) # only float32 readback roundoff, never a target projection
    return dict(zip(('shoulder', 'elbow', 'wrist'), map(float, q))) | {
        'projection': dict(upperLength=float(ratios[0]),forearmLength=float(ratios[1]),palmLength=float(ratios[2]),palmWidth=float(1+q[6]/100) if len(q)==8 else 1,palmShear=float(q[7]/100) if len(q)==8 else 0),
        'girdle': dict(rootX=0, rootY=0, socketX=0, socketY=0)}


def fixture(side, stage, local, forward, home, absolute):
    """Targets remain unchanged when late. No task-specific controller tuning."""
    lift = np.array([18, 8, 0]) if side == 'native' else np.array([-35, 65, -15])
    t = local*DT
    if stage == 'pass-then-hold':
        origin = absolute-t
        q = np.array([8., 3., 0.]) if side == 'native' else np.array([-20., 15., -5.])
        qv = np.array([5., 2., 0.]) if side == 'native' else np.array([-8., 4., 0.])
        nodes = forward(q if t < 1.4 else q+qv*.4)
        angle = lambda p: float(np.degrees(np.arctan2(*(p[3]-p[2])[::-1])))
        if t < 1.4:
            ahead = forward(q+qv*.01)
            from tools.control.stateful_arm import wrap
            return Target(nodes[2], origin+1., elbow=nodes[1], palm_heading=angle(nodes), mode='pass',
                          velocity=(ahead[2]-nodes[2])/.01, angular_velocity=float(wrap(angle(ahead)-angle(nodes)))/.01)
        return Target(nodes[2], origin+2., elbow=nodes[1], palm_heading=angle(nodes), mode='hold')
    if stage == 'visible-lift-hold-lower':
        lift = np.array([-35., 0, 0])
    if stage == 'point-reach-return':
        if side == 'patch':
            lift = np.array([-35., 0, 0])
        q = lift if local < 54 else np.zeros(3)
    elif stage in ('lift-hold-lower', 'visible-lift-hold-lower'):
        amount = min(t/2, 1) if t < 4 else max(0, (6-t)/2)
        q = lift*amount
    elif stage == 'wrist-positive-negative':
        q = np.array([0., 0., 20*np.sin(t*np.pi/2)])
    elif stage == 'continuous-route':
        route_time = t*2/3
        q = (lift*.55 + np.array([6*np.sin(route_time*np.pi/2), 3*np.sin(route_time*np.pi), 8*np.sin(route_time*np.pi/2)]))*np.sin(np.pi*route_time/4)**2
    elif stage == 'moving-target-switch':
        q = lift*(.85 if t < .7 else .15 if t < 1.4 else .65 if t < 2.1 else 0)
    else:
        q = np.zeros(3)
    nodes = forward(q)
    if stage == 'unreachable-return' and .5 <= t < 2:
        nodes = nodes.copy(); nodes[1:] += np.array([0, .8])
    heading = float(np.degrees(np.arctan2(*(nodes[3]-nodes[2])[::-1])))
    # Moving tests have a same-timestamp tracking requirement. Endpoint holds
    # have a fixed .6s allowance; this is logged and never shifted by execution.
    origin = absolute-t
    if stage == 'point-reach-return':
        deadline = origin+(.6 if local < 54 else 2.4)
    elif stage in ('lift-hold-lower', 'visible-lift-hold-lower'):
        deadline = absolute if t < 2 or 4 <= t < 6 else origin+(2.6 if t < 4 else 6.6)
    elif stage == 'moving-target-switch':
        deadline = origin+min(int(t/.7), 3)*.7+.6
    elif stage == 'unreachable-return':
        deadline = origin+(.5+.6 if t < 2 else 2+.6)
    else:
        deadline = absolute
    return Target(nodes[2], deadline, elbow=nodes[1], palm_heading=heading,
                  shoulder=home[0] if side == 'patch' else None, branch=int(np.sign(bend(nodes))),
                  mode='hold' if stage in ('point-reach-return', 'moving-target-switch', 'unreachable-return') else 'track')


def run(root, physics=False, context=False, only_side=None, visible_lift=False,
        profile=DEFAULT_PROFILE, speed_trial=False, point_trial=False, controller='timed', suite=False, request_stream=None,
        native_five=False, native_limits=None, request_provider=None, patch_projection=False, projection_aware=True,patch_rooted=False,palm_plane=False,native_fingers=False,native_gaze=False,task_markers=None,face_control=False,camera_margin=None):
    if camera_margin is not None and (not np.isfinite(camera_margin) or not 1.08<=camera_margin<=2.):raise ValueError('lab camera margin')
    if native_gaze and request_stream is None:raise ValueError("gaze requires request clock stream")
    if native_fingers and request_stream is None:raise ValueError("native fingers require request clock stream")
    if palm_plane and not patch_projection:raise ValueError("palm plane requires mesh-consistent projection backend")
    patch_dof=8 if palm_plane else 6
    patch_keys=PALM_PLANE_KEYS if palm_plane else PATCH_KEYS
    limits = speed_limits(profile)
    native_limits = limits if native_limits is None else native_limits
    if patch_rooted and not patch_projection:raise ValueError('Rooted connection requires the mesh-consistent patch backend')
    native_keys = NATIVE_KEYS if native_five else KEYS
    if controller not in ('legacy', 'normalized', 'timed'):
        raise ValueError('controller')
    controller_type = ArmController if controller == 'legacy' else TimedArmController
    controller_options = {} if controller == 'legacy' else {'policy': controller}
    root.mkdir(parents=True, exist_ok=False)
    frozen = root/'implementation'; frozen.mkdir()
    if face_control:shutil.copy2(Path(__file__).with_name('face_control.py'),frozen/'face_control.py')
    if native_gaze:shutil.copy2(Path(__file__).with_name('gaze_control.py'),frozen/'gaze_control.py')
    if native_fingers:shutil.copy2(Path(__file__).with_name('finger_control.py'),frozen/'finger_control.py')
    if palm_plane:
        shutil.copy2(Path(__file__).with_name('palm_plane.py'),frozen/'palm_plane.py')
        shutil.copy2(unity.ROOT/'preview/unity/RecoveredArmTypes.cs',frozen/'RecoveredArmTypes.cs')
        shutil.copy2(unity.ROOT/'preview/unity/RecoveredArmGeometry.cs',frozen/'RecoveredArmGeometry.cs')
    for name in (__file__, 'tools/control/stateful_arm.py', 'tools/control/timed_arm.py', 'tools/control/native_target.py','tools/control/patch_projection.py', 'preview/unity/LabMotion.cs',
                 'preview/unity/WholeArmSession.cs', 'preview/unity/RecoveredShoulderConnection.cs'):
        shutil.copy2(name, frozen/Path(name).name)
    if request_provider is not None:
        for name in request_provider.implementation_files:
            shutil.copy2(name,frozen/Path(name).name)
    plant = Plant(); model = plant.model
    patch_home = np.array(json.loads((APPROVED/'protocol.json').read_text())['newNodes'])
    model.set({}); a = plant.anchors.read()
    native_home = np.array([a[k] for k in ('shoulder', 'elbow', 'wrist', 'palm')])
    patch_model=PatchProjectionPlant(model,patch_home,rooted_upper=patch_rooted) if patch_projection else None
    def native_fixture(q):
        model.set(dict(zip(KEYS, q))); a = plant.anchors.read()
        return np.array([a[k] for k in ('shoulder', 'elbow', 'wrist', 'palm')])
    fixtures = {'native': native_fixture, 'patch': lambda q: patch_forward(patch_home, q)}
    schedule = [('native', name, j) for name, n in STAGES for j in range(n)]
    schedule += [('patch', name, j) for name, n in STAGES for j in range(n)]
    if only_side is not None:
        schedule = [s for s in schedule if s[0] == only_side]
    if context:
        schedule = [('both', 'body-tracking', j) for j in range(180)]
    if visible_lift:
        schedule = [('patch', 'visible-lift-hold-lower', j) for j in range(210)]
    if speed_trial:
        # Exact existing spatial/time requests, including the deliberately hard
        # .6s switch deadlines; no profile-dependent retiming or amplitude change.
        schedule = [(side, stage, j) for side in ('native', 'patch')
            for stage, n in [('moving-target-switch', 90), ('unreachable-return', 180)] for j in range(n)]
    if point_trial:
        schedule = [(side, 'point-reach-return', j) for side in ('native', 'patch') for j in range(108)]
    if suite:
        schedule = [(side, stage, j) for side in ('native', 'patch')
            for stage, n in [('continuous-route', 180), ('wrist-positive-negative', 120),
                             ('moving-target-switch', 90), ('unreachable-return', 180), ('pass-then-hold', 96)] for j in range(n)]
    if request_stream is not None:
        schedule = [('both', f['stage'], j) for j, f in enumerate(request_stream['frames'])]
    count = len(schedule)+1
    executed = []
    for side, stage, _ in schedule:
        if not executed or (executed[-1]['side'], executed[-1]['stage']) != (side, stage):
            executed.append(dict(side=side, stage=stage, frames=0))
        executed[-1]['frames'] += 1
    unity.write_spec(root/'protocol.json', dict(fps=30, frames=count, physics=physics, profile=profile, limits=limits, controller=controller,
        executedStages=executed, speedTrial=speed_trial, pointTrial=point_trial, suite=suite,
        coordinateFrame='model-local-xy; x right y up; native anatomicalLeft, patch anatomicalRight',
        wristBinding=str(APPROVED), stages=STAGES if not context else [('body-tracking', 180)],
        criteria=dict(wristError=.005, elbowError=.005, palmErrorDeg=4, nodeSpeed=limits['node_speed'],
                      nodeAcceleration=limits['node_acceleration'], parameterSpeed=limits['speed'], parameterAcceleration=limits['acceleration'],
                      minTriangleAreaRatio=.5, flippedTriangles=0, pairing=1e-6, restore=1e-6),
        terminalCriteria=dict(holdMaxNodeSpeed=.02, holdPalmAngularSpeed=8., passWristVelocityError=.02),
        endpointConnector='local endpoint fit + deadline-conditioned actuator boundary interpolation; explicit route samples bypass connector',
        evaluationWindows={'liftHold': [90, 120], 'lowerHold': [195, 210],
            'pointHold': [45, 54], 'pointReturnHold': [99, 108],
            'wristAndRoute': 'all frames; no dropped entry/exit samples',
            'switch': 'report missed fixed .6s deadlines; pass handling only if continuity and failure reporting valid',
            'unreachable': 'retain residual during .5..2s; final 15 frames must recover within tolerance'},
        taskRanges={'nativeLiftParametersUsedOnlyForFixture': [18, 8, 0],
            'patchLiftParametersUsedOnlyForFixture': [-35, 65, -15], 'wristDegrees': [-20, 20],
            'additionalVisiblePatchLiftFixture': [-35, 0, 0],
            'liftTimingSeconds': [2, 2, 2, 1], 'routeSeconds': 6,
            'unreachableYOffset': .8, 'pointSwitchTimes': [0, .7, 1.4, 2.1]},
        actuators={'native': KEYS, 'patch': ['shoulder', 'elbow', 'wrist']},
        projection='identity locked; no independent shortening or length clipping',
        shoulders='body context observed; independent root translation unsupported',
        nativeModel='Core finite differences at full last SDK applied context, corrected by actual nodes; no physics prediction',
        solver='one whole-frame constrained step, local branch retained; local stall is not global unreachability proof',
        fixture='node trajectories constructed with known forward models, not controller input actuator commands',
        visualAcceptance='pending user; no naturalness inferred from numerical checks',
        deadline='moving route time unchanged; point switches allow fixed .6s, reported without retiming'))
    if request_stream is not None:
        unity.write_spec(root/'request-stream.json', request_stream)
        protocol = json.loads((root/'protocol.json').read_text())
        protocol['externalRoute'] = request_stream['metadata']
        unity.write_spec(root/'protocol.json', protocol)
    if native_five:
        protocol = json.loads((root/'protocol.json').read_text())
        protocol['nativeBackend'] = 'five measured axes; calibrated direction/ratio initialization + Core refinement'
        protocol['actuators']['native'] = native_keys
        protocol['nativeLimits'] = native_limits
        protocol['criteriaOverrides'] = dict(native=protocol['criteria'] | dict(
            nodeSpeed=native_limits['node_speed'], nodeAcceleration=native_limits['node_acceleration'],
            parameterSpeed=native_limits['speed'], parameterAcceleration=native_limits['acceleration']))
        unity.write_spec(root/'protocol.json', protocol)
    if patch_projection:
        protocol=json.loads((root/'protocol.json').read_text())
        protocol.update(projection='frontal lengths and optional palm transverse basis; original approved material bindings',
            patchProjection=True,patchProjectionLimits=dict(minimum=[.75,.55,.4],maximum=[1,1,1],ratioTolerance=.03,palmDirectionWeight=3.,
            units='angle degrees + (length ratio - 1)*100; same whole-frame budgets for all axes'))
        if projection_aware:
            protocol['patchGeometryGuard']=dict(metric='min(per-triangle same-projection angular ratio, projection-only area/expected determinant, raw*.55/.30)',
                planned=.55,minimumActualGuard=.5,minimumActualRawArea=.28,plannedRawFloor=.30,flips=0)
        if palm_plane:protocol['patchGeometryGuard']['handRawPlanningFloor']=.36
        protocol['actuators']['patch']=patch_keys
        protocol['palmPlane']=dict(enabled=palm_plane,widthBounds=[.35,1],shearBounds=[-.95,.95],basisTolerance=.03,missingBasis='identity',backFace='unsupported; signed request retained')
        protocol['patchConnection']='existing rootedUpper distributed shoulder angle' if patch_rooted else 'rigid MLS'
        unity.write_spec(root/'protocol.json',protocol)
    if native_fingers:
        protocol=json.loads((root/'protocol.json').read_text())
        protocol['nativeFingers']=dict(ids=['Param102','Param103','Param105','Param106','Param107'],range=[-20,20],speed=90,acceleration=540,error=.15,settledSpeed=.5,mode='reachBy point pose; not finger route tracking',patch='unsupported')
        unity.write_spec(root/'protocol.json',protocol)
    if native_gaze:
        protocol=json.loads((root/'protocol.json').read_text())
        protocol['gaze']=dict(ids=['ParamEyeBallX','ParamEyeBallY'],range=[-.8,.8],speed=3.,acceleration=15.,scope='authored eye inputs; not a calibrated 3D sight ray')
        unity.write_spec(root/'protocol.json',protocol)
    patch = unity.write_spec(root/'patch.json', dict(nativeMarkers=markers(model, plant.anchors),taskMarkers=task_markers or [],
        correctedPatchNodes=patch_home.ravel().tolist(), nativeDistalFront=True,rootedUpper=patch_rooted,
        poses=[pose([0, 0, 0]) for _ in range(count)]))
    bridge = root/'bridge'; bridge.mkdir(); output = root/'capture'; errors = []
    def capture():
        try:
            unity.run('LabMotion.RunStatefulControl', output, dict(ACTION_LAB_CAMERA_MARGIN='' if camera_margin is None else camera_margin,ACTION_LAB_FACE='on' if face_control else 'off',ACTION_LAB_GAZE='on' if native_gaze else 'off',ACTION_LAB_NATIVE_FINGERS='on' if native_fingers else 'off',ACTION_LAB_PALM_PLANE='on' if palm_plane else 'off',ACTION_LAB_MODEL=unity.model3_path(),
                ACTION_LAB_PATCH=patch, ACTION_LAB_LAYER_FRAME='', ACTION_LAB_PHYSICS_OVERRIDE='',
                ACTION_LAB_CONTROL_FRAMES=count, ACTION_LAB_CONTROL_BRIDGE=bridge,
                ACTION_LAB_STATEFUL_PARAMETER_SPEED=native_limits['speed'],
                ACTION_LAB_STATEFUL_NATIVE_AXES=5 if native_five else 3,
                ACTION_LAB_GEOMETRY_FEEDBACK='on',
                ACTION_LAB_PATCH_PROJECTION='on' if patch_projection else 'off',
                ACTION_LAB_PATCH_AREA_MODE='projection-local' if patch_projection and projection_aware else 'absolute',
                ACTION_LAB_STATEFUL_PHYSICS='on' if physics else 'off'), root/'capture.log', timeout=1800)
        except Exception as exc:
            errors.append(exc)
    thread = threading.Thread(target=capture); thread.start()
    commands = {'native': np.zeros(len(native_keys)), 'patch': np.zeros(patch_dof if patch_projection else 3)}
    finger_controller=None;gaze_controller=None;face_controller=None
    controllers = {}; rows = []; applied = {}; context_requested = {}; source = patch_home.copy()
    inactive_deadline = {'native': None, 'patch': None}
    def native_predict(q):
        model.set(applied | context_requested | dict(zip(native_keys, q)))
        a = plant.anchors.read()
        return np.array([a[k] for k in ('shoulder', 'elbow', 'wrist', 'palm')])
    def patch_predict(q):
        patch_model.set_context(applied|context_requested)
        return patch_model.forward(q)
    track = (root/'track.jsonl').open('w', encoding='utf-8')
    for i in range(count):
        packet = request_stream['frames'][i-1] if request_stream is not None and i else None
        if i and request_provider is not None:
            packet = request_provider.next(i*DT)
        context_requested = dict(zip(('ParamAngleX','ParamAngleY','ParamAngleZ','ParamBodyAngleX','ParamBodyAngleY','ParamBodyAngleZ'), packet['headBody'])) if packet is not None else {}
        if i:
            active, stage, local = schedule[i-1]
            if request_provider is not None:
                stage=packet['stage']
            for side in ('native', 'patch'):
                home = native_home if side == 'native' else patch_home
                if active in (side, 'both'):
                    inactive_deadline[side] = None
                    target = fixture(side, stage, local, fixtures[side], home, i*DT)
                else:
                    if inactive_deadline[side] is None:
                        inactive_deadline[side] = i*DT+.6
                    target = Target(home[2], inactive_deadline[side], elbow=home[1],
                        palm_heading=float(np.degrees(np.arctan2(*(home[3]-home[2])[::-1]))), mode='hold')
                if stage == 'body-tracking':
                    target = fixture(side, 'continuous-route', local, fixtures[side], home, i*DT)
                if packet is not None:
                    target = Target(**packet['targets'][side])
                commands[side], _ = controllers[side].command(target, i*DT)
        body = [2*np.sin(i*DT), 1.5*np.sin(i*DT*.7)] if context else [0, 0]
        path = bridge/f'{i:04}.command.json'; temp = path.with_suffix('.tmp')
        command = dict(q=commands['native'].tolist(), patch=pose(commands['patch']), body=body)
        if packet is not None:
            command['headBody'] = packet['headBody']
        if native_fingers:
            command['nativeFingers']=finger_controller.command(**packet['nativeFingerGoal'],now=i*DT).tolist() if i else [0.]*5
        if native_gaze:
            command['gaze']=gaze_controller.command(**packet['gazeGoal'],now=i*DT).tolist() if i else [0.,0.]
        if face_control:
            command['face']=face_controller.command(packet['faceGoal'],i*DT).tolist() if i else [1.,1.,0.,0.,0.,0.]
        unity.write_spec(temp, command); temp.replace(path)
        start = time.monotonic()
        while not (bridge/f'{i:04}.ready').exists():
            if errors:
                raise errors[0]
            if time.monotonic()-start > 180:
                raise TimeoutError(str(i))
            time.sleep(.01)
        frame = json.loads((output/f'{i:04}.json').read_text(encoding='utf-8-sig'))
        applied = {v['id']: v['value'] for v in frame['applied']}
        source[:] = np.array(frame['wholeArms']['sourcePatchChain']).reshape(4, 2)
        actual = {s: np.array(frame['wholeArms'][s+'Chain']).reshape(4, 2) for s in ('native', 'patch')}
        geometry={k:frame['wholeArms'][k] for k in GEOMETRY_KEYS}
        if patch_projection and projection_aware:
            geometry['minTriangleAreaRatio']=frame['wholeArms']['geometryGuardAreaRatio']
            geometry['rawMinTriangleAreaRatio']=frame['wholeArms']['minTriangleAreaRatio']
        if i == 0:
            if face_control:
                from tools.control.face_control import FaceController,FACE_IDS
                face_controller=FaceController([applied[k] for k in FACE_IDS])
            if native_gaze:
                from tools.control.gaze_control import GazeController,GAZE_IDS
                gaze_controller=GazeController([applied[k] for k in GAZE_IDS])
            if native_fingers:
                from tools.control.finger_control import FingerController,FINGER_IDS
                finger_controller=FingerController([applied[k] for k in FINGER_IDS])
            if native_five:
                controllers['native'] = NativeTargetController(native_predict, model, plant.anchors,
                    [applied[k] for k in native_keys], actual['native'], **native_limits)
            else:
                controllers['native'] = controller_type('native', native_predict, [-29.5, -29.8, -30], [59.5, 29.8, 30],
                    [applied[k] for k in native_keys], actual['native'], **limits, **controller_options)
            if patch_projection:
                patch_model.set_context(applied)
                controllers['patch']=PatchProjectionController(patch_predict,lambda:patch_model.raw_nodes.copy(),
                    np.zeros(patch_dof),actual['patch'],**limits)
            else:
                controllers['patch'] = controller_type('patch', lambda q: patch_forward(source, q), [-56, 0, -100], [56, 145, 30],
                    np.zeros(3), actual['patch'], **limits, **controller_options)
            controllers['patch'].geometry = geometry
            if request_provider is not None:
                from tools.control.body_session import HEAD_BODY_IDS
                request_provider.initialize(dict(time=0.,arms={s:dict(actual=actual[s].tolist(),applied=controllers[s].q.tolist(),
                    **({'projection':frame['wholeArms']['projectedLengthRatios']} if s=='patch' and patch_projection else {})) for s in ('native','patch')},
                    headBody=[applied[k] for k in HEAD_BODY_IDS],capabilities=dict(
                        coordinateFrame='model-local-xy; x right y up',fps=30,
                        native=dict(identity='anatomicalLeft / screenRight',actuators=native_keys,limits=native_limits),
                        patch=dict(identity='anatomicalRight / screenLeft',actuators=patch_keys if patch_projection else ['shoulder','elbow','wrist'],
                            limits=limits,projection='frontal lengths' if patch_projection else 'identity',
                            connection='rootedUpper' if patch_rooted else 'rigid MLS'),
                        headBody=dict(inputIds=HEAD_BODY_IDS,physics='original SDK, continuous fixed step'),
                        unsupported=['independent shoulder translation','3D palm normal','patch finger targets']+([] if native_fingers else ['native finger targets']),
                        nativeFingers=bool(native_fingers),gaze=bool(native_gaze),face=bool(face_control))))
            continue
        if patch_projection:
            controllers['patch'].actual_source_nodes=source.copy()
        row = dict(frame=i, time=i*DT, active=active, stage=stage, localFrame=local,
            native=controllers['native'].observe([applied[k] for k in native_keys], actual['native']),
            patch=controllers['patch'].observe(frame['wholeArms']['patchActuators'], actual['patch'],
                geometry),
            geometry={k: frame['wholeArms'][k] for k in ('minTriangleAreaRatio', 'flippedTriangles', 'restoredMeshError')},
            pairing=frame['pairingMaxDelta'], body=body)
        if patch_projection and projection_aware:
            row['geometry']['rawMinTriangleAreaRatio']=row['geometry']['minTriangleAreaRatio']
            row['geometry']['minTriangleAreaRatio']=frame['wholeArms']['geometryGuardAreaRatio']
            row['geometry']['guardMode']=frame['wholeArms']['geometryGuardMode']
        if face_control:
            from tools.control.face_control import FACE_IDS
            row['face']=face_controller.observe([applied[k] for k in FACE_IDS])
        if native_gaze:
            from tools.control.gaze_control import GAZE_IDS
            row['gaze']=gaze_controller.observe([applied[k] for k in GAZE_IDS])
        if task_markers:
            values=np.array(frame['wholeArms']['taskMarkerPoints']).reshape(-1,2)
            row['taskMarkers']=dict(zip(frame['wholeArms']['taskMarkerIds'],values.tolist()))
        if packet is not None and 'intent' in packet:row['intent']=packet['intent']
        if native_fingers:
            from tools.control.finger_control import FINGER_IDS
            row['nativeFingers']=finger_controller.observe([applied[k] for k in FINGER_IDS])
            row['nativeFingerMeshes']=[g for g in frame['drawables'] if g['id'] in ('ArtMesh230','ArtMesh231','ArtMesh233','ArtMesh234','ArtMesh235')]
        if palm_plane:
            row['patch']['palmAffineReadback']=frame['wholeArms']['palmAffineReadback']
        if packet is not None:
            row['sourceTime'] = packet.get('sourceTime')
            row['headBodyRequest'] = packet['headBody']
        if request_provider is not None:
            from tools.control.body_session import HEAD_BODY_IDS
            row['physicalContext']={k:applied[k] for k in ('ParamBodyAngleX2','ParamBodyAngleY2','ParamBodyAngleZ2','Param251') if k in applied}
            row['session']=request_provider.observe(row,[applied[k] for k in HEAD_BODY_IDS])
        rows.append(row); track.write(json.dumps(row)+'\n'); track.flush()
        if i % 60 == 0:
            print(i, active, stage, {s: (row[s]['status'], round(row[s]['wristError'], 4)) for s in ('native', 'patch')}, flush=True)
    track.close(); thread.join()
    if errors:
        raise errors[0]
    summary = []
    for active in sorted({r['active'] for r in rows}):
        for stage in dict.fromkeys(r['stage'] for r in rows if r['active'] == active):
            subset = [r for r in rows if r['active'] == active and r['stage'] == stage]
            for side in ('native', 'patch') if active == 'both' else (active,):
                records = [r[side] for r in subset]
                summary.append(dict(side=side, stage=stage, frames=len(records),
                    finalWristError=records[-1]['wristError'],
                    tailWristRMS=float(np.sqrt(np.mean([r['wristError']**2 for r in records[-15:]]))),
                    reachedFrames=sum(r['status'] == 'reached' for r in records),
                    failedFrames=sum(r['status'] == 'failed' for r in records),
                    limitedFrames=sum(r['status'] == 'limited' for r in records),
                    reasons=sorted({reason for r in records for reason in r['reasons']}),
                    maxima={k: max(r['metrics'][k] for r in records) for k in records[0]['metrics']},
                    minArea=min(r['geometry']['minTriangleAreaRatio'] for r in subset),
                    maxFlips=max(r['geometry']['flippedTriangles'] for r in subset),
                    maxPairing=max(r['pairing'] for r in subset),
                    maxRestore=max(r['geometry']['restoredMeshError'] for r in subset)))
    unity.write_spec(root/'results.json', summary)
    print(root/'results.json', flush=True)


if __name__ == '__main__':
    run(Path(sys.argv[1]), physics='--physics' in sys.argv, context='--context' in sys.argv,
        only_side='patch' if '--patch-only' in sys.argv else None, visible_lift='--visible-lift' in sys.argv,
        profile=sys.argv[sys.argv.index('--profile')+1] if '--profile' in sys.argv else DEFAULT_PROFILE,
        speed_trial='--speed-trial' in sys.argv, point_trial='--point-trial' in sys.argv,
        controller=sys.argv[sys.argv.index('--controller')+1] if '--controller' in sys.argv else 'timed', suite='--suite' in sys.argv)
