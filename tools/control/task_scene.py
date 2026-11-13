"""Few explicit scene goals. Existing calibration, no model-wide scan or new hand warp."""
import json
from pathlib import Path
import numpy as np
from tools.control.pilot import Plant
from tools.control.intent_expression_trial import build_markers,read_markers,POINT,POINT_HEAD
from tools.control.native_target import NATIVE_KEYS,NativeTargetController,NATIVE_ROUTE_LIMITS
from tools.control.body_session import HEAD_BODY_IDS,nodes_goal
from tools.control.gesture_exploration import POSES
from tools.control.finger_control import FINGER_IDS
from tools.control.stateful_arm import Target,patch_forward
from tools.control.timed_arm import heading
from tools.control.validate_stateful_arm import APPROVED
from tools.control.mmd_timed_controller import make_stream


def build_scene(source):
    plant=Plant();m=plant.model;markers=build_markers(plant);m.set({});a=plant.anchors.read()
    home=np.array([a[k] for k in ('shoulder','elbow','wrist','palm')])
    patch=np.array(json.loads((APPROVED/'protocol.json').read_text(encoding='utf-8'))['newNodes'])
    src=json.loads(Path(source).read_text(encoding='utf-8-sig'))
    high=make_stream(src,dict(native=home,patch=patch))['frames'][58]
    records=[]

    def fit(name,xy,node_goal,hb,gaze):
        # xy is fixed before fitting: never move the object onto the resulting ray.
        context=dict(zip(HEAD_BODY_IDS,hb))|dict(zip(FINGER_IDS,POSES['index-emphasis']))
        def forward(q):
            m.set(context|dict(zip(NATIVE_KEYS,q)));p=plant.anchors.read()
            return np.array([p[k] for k in ('shoulder','elbow','wrist','palm')])
        ctrl=NativeTargetController(forward,m,plant.anchors,np.zeros(5),home,**NATIVE_ROUTE_LIMITS)
        target={k:v for k,v in node_goal.items() if k in ('wrist','elbow','palm_heading','branch')}
        q=np.zeros(5)
        # Bounded local orientation correction reuses the existing five-axis fit.
        # It yields node commands, not an actuator bypass for the live executor.
        for _ in range(6):
            q,nodes,cost=ctrl._fit_endpoint(Target(**target,time=2.),forward,q)
            forward(q);p=read_markers(m,markers)
            axis=p['indexTip']-p['indexBase'];toward=np.array(xy)-p['indexBase']
            error=(np.degrees(np.arctan2(toward[1],toward[0])-np.arctan2(axis[1],axis[0]))+180)%360-180
            if abs(error)<.5:break
            target['palm_heading']+=float(error)
        records.append(dict(id=name,objectFixedBeforeFit=xy,orientationErrorDegrees=float(error),
                            predictedWristError=float(np.linalg.norm(nodes[2]-target['wrist'])),
                            fittedParameters=q.tolist(),context=hb,scope='Core preflight only; live SDK must independently pass'))
        # Keep the independently requested wrist/elbow; only the fitted palm heading
        # is submitted. Preflight failure is retained rather than called unavailable.
        return dict(label=name,xy=xy,gaze=gaze,pointing=dict(nodes=target,headBody=hb,reachSeconds=2.2),
                    criterion='native-index-material-axis',visualAccepted=False)

    m.set(dict(zip(NATIVE_KEYS,POINT))|dict(zip(HEAD_BODY_IDS,POINT_HEAD)));a=plant.anchors.read()
    low=nodes_goal([a[k] for k in ('shoulder','elbow','wrist','palm')])
    scene={}
    scene['A']=fit('右侧位置',[.59,.28],low,POINT_HEAD.tolist(),[.7,-.15])
    scene['C']=fit('上方位置',[-.12,.65],high['targets']['native'],high['headBody'],[-.2,.55])
    # Explicit bilateral expansion goal. Patch has no finger target or palm-normal
    # promise; completion checks its actual nodes and final mesh guards.
    expansion=nodes_goal(patch_forward(patch,[-38.,18.,-4.]))
    scene['E']=dict(label='双侧对照',xy=[0.,.3],gaze=[0.,0.],criterion='bilateral-node-hold',
        pointing=dict(arms=dict(native=scene['A']['pointing']['nodes'],patch=expansion),
                      headBody=POINT_HEAD.tolist(),gesture='neutral',reachSeconds=2.4),visualAccepted=False)
    unreachable=dict(wrist=[.12,.95],elbow=[.13,.65],palm_heading=90.)
    scene['U']=dict(label='超范围测试',xy=[.12,1.05],gaze=[.15,.7],criterion='native-index-material-axis',
        pointing=dict(nodes=unreachable,headBody=[0.,0.,0.,0.,0.,0.],reachSeconds=1.3),expectedLimited=True)
    return scene,markers,records
