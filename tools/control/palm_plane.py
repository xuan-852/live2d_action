"""Orthographic palm-plane basis; no finger shape or hidden back-face artwork.

Longitudinal column is (length,0), transverse column is (shear,width).
normal_z is the signed projected area of a unit orthonormal palm plane.
Length + normal_z leaves two shear branches: callers must state the branch.
"""
import numpy as np


def palm_basis(length, normal_z, branch):
    if branch not in (-1,1):
        raise ValueError('explicit palm shear branch required')
    if not np.isfinite([length,normal_z]).all() or not 0 < length <= 1 or abs(normal_z)>length+1e-6:
        raise ValueError('inconsistent orthographic palm length/normal')
    width=float(np.clip(normal_z/length,-1,1))
    shear=float(branch*np.sqrt(max(0.,(1-length**2)*(1-width**2))))
    return np.array([width,shear])


def bases_from_reference(reference,source):
    """Use already recorded FK, validate against every exposed source sample.

    Axis is wrist -> mean four MCPs, transverse axis is little -> index,
    orthogonalized in 3D. Source x right/y up; camera orthographic +Z.
    No pose retiming or source length/heading changes.
    """
    frames=reference['frames'];targets=source['sides']['anatomicalRight']['frames']
    if reference.get('kind')!='mmd-fk' or len(frames)!=len(targets):
        raise ValueError('reference type/frame count mismatch')
    result=[]
    for i,(frame,row) in enumerate(zip(frames,targets)):
        t=row['target'];point=lambda name:np.asarray(frame[name],float)
        wrist=point('\u53f3\u624b\u9996')
        knuckle=np.mean([point('\u53f3'+finger+'\uff11') for finger in ('\u4eba\u6307','\u4e2d\u6307','\u85ac\u6307','\u5c0f\u6307')],axis=0)
        u=knuckle-wrist;length=np.linalg.norm(u)
        if length<1e-8:raise ValueError('degenerate source palm')
        u/=length
        v=point('\u53f3\u4eba\u6307\uff11')-point('\u53f3\u5c0f\u6307\uff11');v-=u*(v@u)
        if np.linalg.norm(v)<1e-8:raise ValueError('degenerate source transverse axis')
        v/=np.linalg.norm(v);r=np.linalg.norm(u[:2])
        if r<1e-6:raise ValueError('palm heading unobservable')
        d=u[:2]/r;e=np.array([-d[1],d[0]])
        normal=float(np.cross(u,v)[2])
        expected=t['sourceJointScreen']
        actual=[point('\u53f3\u8155')[:2],point('\u53f3\u3072\u3058')[:2],wrist[:2],knuckle[:2]]
        wanted=[expected[k] for k in ('shoulder','elbow','wrist','knuckle')]
        angle=np.degrees(np.arctan2(u[1],u[0]))
        if (abs(row['seconds']-i/30)>1e-7 or np.max(abs(np.array(actual)-wanted))>1e-7 or
            abs(r-t['palmProjectedRatio'])>1e-7 or abs(normal-t['palmNormalZ'])>1e-7 or
            abs((angle-t['palmHeadingDeg']+180)%360-180)>1e-6):
            raise ValueError('FK reference does not match source sample '+str(i))
        result.append([float(e@v[:2]),float(d@v[:2])])
    return result
