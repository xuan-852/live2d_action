"""Explicit upper-request acceptance fixture, not behavior generation/learning."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import sys,json,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from tools.control.body_session import BodyExecutionSession,nodes_goal
from tools.control.mmd_timed_controller import make_stream
from tools.control.pilot import Plant
from tools.control.native_target import NATIVE_KEYS,NATIVE_ROUTE_LIMITS
from tools.control.stateful_arm import DT,patch_forward
from tools.control.validate_stateful_arm import run,APPROVED
from tools.expression import unity


class TrialProvider:
    implementation_files=['tools/control/body_session.py',__file__]
    def __init__(self,samples,high,low):
        self.session=BodyExecutionSession();self.samples=samples;self.high=high;self.low=low

    def initialize(self,feedback):self.session.initialize(feedback)

    def next(self,now):
        i=round(now/DT);s=self.session
        if i==1:s.submit_route('mmd-full',self.samples,1.,now)
        elif i==121:s.submit_pose('pose-high',self.high,[8.,-4.,3.,1.,.5,-.5],now+1.,now)
        elif i==211:s.submit_route('mmd-interrupted',self.samples,.8,now)
        elif i==250:s.freeze('freeze-actual',now,now+.5)
        elif i==280:s.submit_pose('pose-low',self.low,[-6.,2.,-2.,-1.,-.5,.3],now+.8,now)
        elif i==340:s.release('release',now,now+1.2)
        return s.packet(now)

    def observe(self,row,context):return self.session.observe(row,context)


def execute(root,source,patch_projection=False,patch_rooted=False):
    root.mkdir(parents=True,exist_ok=False)
    plant=Plant();model=plant.model;model.set({});a=plant.anchors.read()
    homes=dict(native=np.array([a[k] for k in ('shoulder','elbow','wrist','palm')]),
        patch=np.array(json.loads((APPROVED/'protocol.json').read_text())['newNodes']))
    doc=json.loads(source.read_text(encoding='utf-8-sig'));stream=make_stream(doc,homes,patch_projection=patch_projection)
    samples=[dict(time=p['sourceTime'],arms=p['targets'],headBody=p['headBody']) for p in stream['frames'] if p['stage']=='mmd-source']
    def native_pose(q):
        model.set(dict(zip(NATIVE_KEYS,q)));a=plant.anchors.read()
        return nodes_goal([a[k] for k in ('shoulder','elbow','wrist','palm')])
    native_high=nodes_goal(np.array([homes['native'][0],samples[28]['arms']['native']['elbow'],samples[28]['arms']['native']['wrist'],
        np.array(samples[28]['arms']['native']['wrist'])+.03*np.array([np.cos(np.radians(samples[28]['arms']['native']['palm_heading'])),np.sin(np.radians(samples[28]['arms']['native']['palm_heading']))])]))
    high=dict(native=native_high,patch=nodes_goal(patch_forward(homes['patch'],[-20,15,-5])))
    low=dict(native=native_pose([12,0,7,0,0]),patch=nodes_goal(patch_forward(homes['patch'],[-10,30,10])))
    if patch_projection:
        high['patch']['projection']=[1.,1.,1.];low['patch']['projection']=[1.,1.,1.]
    provider=TrialProvider(samples,high,low)
    unity.write_spec(root/'upper-requests.json',dict(coordinateFrame='model-local-xy; native anatomicalLeft / patch anatomicalRight',
        route=samples,high=high,low=low,source=str(source),events={1:'full MMD, 1s entry',121:'replace by direct high pose',211:'MMD again, .8s entry',250:'interrupt and hold ACTUAL previous SDK pose',280:'replace by direct low pose',340:'release to initial actual neutral'},
        purpose='hand-authored protocol acceptance scenario only, not upper-level learning'))
    unity.write_spec(root/'acceptance-protocol.json',dict(frames=451,fps=30,onePersistentSDK=True,
        route='first MMD must deliver all original54 samples with exact times/targets/context; second explicitly interrupted after15 samples',
        control='same native/patch budgets, no resets; all-frame actual continuity and geometry checks',
        lifecycle='replacement and interrupt apply on that frame; freeze uses previous SDK actual nodes, not ideal nodes',
        terminal='last15 frames must settle both arms; input head/body residual<=.05; idle only after actual settle',
        headBody='exact source during playback; inherited-velocity quintic outside source, frozen input ranges; any connection clipping reported',
        failures='retain per-side actual residuals and reasons, even if route playback finishes',visualAcceptance=False))
    if patch_projection:
        protocol=json.loads((root/'acceptance-protocol.json').read_text())
        protocol['patchProjection']=dict(axes='three angles + upper/forearm/palm frontal lengths',ratioTolerance=.03,
            connection='rootedUpper' if patch_rooted else 'rigid MLS',
            guard='same-projection angular/reference guard + raw floor',minimumActualGuard=.5,minimumActualRawArea=.28,flips=0)
        unity.write_spec(root/'acceptance-protocol.json',protocol)
    shutil.copy2(source,root/'source-map.json')
    spec=dict(frames=[dict(stage='runtime-provider') for _ in range(450)],metadata=dict(
        driver='persistent BodyExecutionSession, live SDK feedback callback before next upper request',
        entry='explicit source-external connection; original source times untouched',source=doc['sourceFile']))
    run(root/'sdk',physics=True,request_stream=spec,native_five=True,native_limits=NATIVE_ROUTE_LIMITS.copy(),request_provider=provider,patch_projection=patch_projection,patch_rooted=patch_rooted)
    unity.write_spec(root/'session-history.json',dict(history=provider.session.history,events=provider.session.events))
    unity.write_spec(root/'capabilities.json',provider.session.describe())


if __name__=='__main__':execute(Path(sys.argv[1]),Path(sys.argv[2]),patch_projection='--patch-projection' in sys.argv,patch_rooted='--patch-rooted' in sys.argv)
