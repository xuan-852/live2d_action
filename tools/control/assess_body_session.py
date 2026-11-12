"""End-to-end body request/feedback gates, including retained capability failures."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from tools.control.stateful_arm import DT
from tools.control.timed_arm import heading
from tools.control.assess_stateful_arm import assess
from tools.expression.unity import write_spec


def check(root):
    sdk=root/'sdk';rows=[json.loads(x) for x in (sdk/'track.jsonl').read_text().splitlines()]
    if len(rows)!=450 or not(sdk/'capture/complete.txt').exists():raise ValueError('Incomplete session')
    upper=json.loads((root/'upper-requests.json').read_text());history=json.loads((root/'session-history.json').read_text())
    protocol=json.loads((sdk/'protocol.json').read_text());stage=assess(sdk)
    events={1:'mmd-full',121:'pose-high',211:'mmd-interrupted',250:'freeze-actual',280:'pose-low',340:'release'}
    event_gate=all(rows[i-1]['session']['requestId']==name for i,name in events.items())
    full=[r for r in rows if r['stage']=='mmd-full/source'];interrupted=[r for r in rows if r['stage']=='mmd-interrupted/source']
    full_times=[r['sourceTime'] for r in full];expected=[x['time'] for x in upper['route']]
    target_error=0.;heading_error=0.;context_error=0.
    for r,sample in zip(full,upper['route']):
        context_error=max(context_error,float(np.max(abs(np.array(r['headBodyRequest'])-sample['headBody']))))
        for side in ('native','patch'):
            req=np.array(r[side]['requested']);target=sample['arms'][side]
            target_error=max(target_error,float(np.max(abs(req[1:3]-[target['elbow'],target['wrist']]))))
            heading_error=max(heading_error,float(abs((heading(req)-target['palm_heading']+180)%360-180)))
    freeze=rows[249];prior=rows[248];freeze_error=0.;freeze_heading_error=0.
    for side in ('native','patch'):
        freeze_error=max(freeze_error,float(np.max(abs(np.array(freeze[side]['requested'])[1:3]-np.array(prior[side]['actual'])[1:3]))))
        freeze_heading_error=max(freeze_heading_error,float(abs((heading(freeze[side]['requested'])-heading(prior[side]['actual'])+180)%360-180)))
    # Validate the exported feedback against the separate final mesh JSON, not
    # merely against its controller prediction or duplicated report fields.
    readback_error=0.;applied_error=0.;physical_error=0.
    for r in rows:
        f=json.loads((sdk/'capture'/f"{r['frame']:04}.json").read_text(encoding='utf-8-sig'))
        applied={x['id']:x['value'] for x in f['applied']}
        for side in ('native','patch'):
            actual=np.array(f['wholeArms'][side+'Chain']).reshape(4,2)
            readback_error=max(readback_error,float(np.max(abs(actual-r['session']['arms'][side]['actual']))))
            q=[applied[k] for k in protocol['actuators']['native']] if side=='native' else f['wholeArms']['patchActuators']
            applied_error=max(applied_error,float(np.max(abs(np.array(q)-r['session']['arms'][side]['applied']))))
        for k,value in r['session']['physicalContext'].items():physical_error=max(physical_error,abs(value-applied[k]))
    def held(lo,hi):return all(r['session']['settled'] for r in rows if lo<=r['frame']<=hi)
    gates=dict(contiguousFrames=[r['frame'] for r in rows]==list(range(1,451)),
        requestOwnershipOnEventFrame=event_gate,
        fullOriginalRoute=len(full)==54 and np.allclose(full_times,expected,atol=1e-9),
        sourceTargetsUnchanged=target_error<=1e-8 and heading_error<=1e-6 and context_error<=1e-8,
        interruptedPrefixOnly=len(interrupted)==15 and np.allclose([r['sourceTime'] for r in interrupted],expected[:15]),
        noOldRequestAfterInterrupt=all(r['session']['requestId']!='mmd-interrupted' for r in rows if r['frame']>=250),
        freezeFromActualPreviousFrame=freeze_error<=1e-8 and freeze_heading_error<=1e-6,
        feedbackFresh=all(abs(r['time']-r['session']['observationTime']-DT)<=1e-7 for r in rows),
        exportedFinalMeshMatchesSDK=readback_error<=1e-8,exportedAppliedMatchesSDK=applied_error<=1e-8,
        exportedPhysicsMatchesSDK=physical_error<=1e-8,
        directHighHold=held(196,210),freezeHold=held(270,279),directLowHold=held(325,339),
        releasedToActualIdle=held(436,450) and all(r['session']['phase']=='idle' for r in rows[-15:]),
        allFramesSafetyAndReporting=all(x['safetyAndReporting'] for x in stage),
        contextConnectionBoundsReported=all('head-body-connection-bound' not in r['session']['requestReasons'] or r['session']['executionStatus'] in ('limited','failed') for r in rows))
    summary=[]
    for identifier in events.values():
        selected=[r for r in rows if r['session']['requestId']==identifier]
        summary.append(dict(requestId=identifier,frames=len(selected),phases=list(dict.fromkeys(r['session']['phase'] for r in selected)),
            finalExecutionStatus=selected[-1]['session']['executionStatus'],finalSettled=selected[-1]['session']['settled'],
            limitedFrames=sum(r['session']['executionStatus']=='limited' for r in selected),
            failedFrames=sum(r['session']['executionStatus']=='failed' for r in selected)))
    report=dict(gates=gates,wholeChainEngineeringPassed=all(gates.values()),
        lifecycleInterfacePassed=all(value for name,value in gates.items() if name not in ('directHighHold','freezeHold','directLowHold')),
        poseCapabilityGates={name:gates[name] for name in ('directHighHold','freezeHold','directLowHold')},
        inputPreservation=dict(maxNodeDifference=target_error,maxHeadingDifference=heading_error,maxHeadBodyDifference=context_error),
        actualFeedback=dict(maxMeshDifference=readback_error,maxAppliedDifference=applied_error,maxPhysicsDifference=physical_error),
        headBodyMaxInputStep=float(np.max(abs(np.diff(np.array([r['headBodyRequest'] for r in rows]),axis=0)))),
        requests=summary,history=history,
        limitations='MMD arm residuals remain; source playback completion is not pose success. Engineering gates do not certify naturalness.',visualAcceptance=False)
    write_spec(root/'session-assessment.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='history'},indent=2))
    return report


if __name__=='__main__':check(Path(sys.argv[1]))
