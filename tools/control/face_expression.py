"""Optional authored face realization of existing semantic affect, not lip sync.

Blink scheduling uses session action time, so replacing a plan cannot restart it.
Eight inputs are existing model controls; no texture or topology edits.
"""
import math

EXTENDED_FACE_IDS=('ParamEyeLOpen','ParamEyeROpen','ParamMouthOpenY','ParamMouthForm',
                   'ParamEyeLSmile','ParamEyeRSmile','ParamBrowLY','ParamBrowRY')
PRESETS={
    'neutral':[1.,1.,0.,0.,0.,0.,0.,0.],
    'warm':[.90,.90,0.,.65,.60,.60,.18,.18],
    'uncertain':[.92,.92,0.,-.28,0.,0.,.38,.16],
    'firm':[.82,.82,0.,-.40,-.15,-.15,-.30,-.30],
}


def face_goal(packet,now,accepted):
    intent=packet['intent']
    idle=intent['released'] or intent['kind']=='cancel'
    goal=PRESETS['neutral' if idle else intent['affect']].copy()
    # Fully closed middle, bounded motion is applied by the SDK-feedback controller.
    phase=now%4.3
    blink=math.sin(math.pi*(phase-3.5)/.44)**2 if 3.5<=phase<=3.94 else 0.
    goal[0]*=1-blink;goal[1]*=1-blink
    if intent['speaking'] and not idle:
        goal[2]=.04+.32*(.5-.5*math.cos(2*math.pi*1.35*(now-accepted)))
    return goal
