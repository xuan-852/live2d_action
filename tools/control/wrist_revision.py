"""Patch wrist-only A/B; palm-axis candidate from area-weighted mesh moments."""
import sys,json,ctypes as C,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from tools.control.pilot import Plant
from tools.expression import unity
from tools.whole.batch import markers


def area_axis(vertices,triangles):
    t=np.asarray(vertices,float)[triangles];u=t[:,1]-t[:,0];v=t[:,2]-t[:,0]
    a=abs(u[:,0]*v[:,1]-u[:,1]*v[:,0])/2
    if a.sum()<1e-10:raise ValueError('Zero mesh area')
    centre=np.sum(a[:,None]*t.mean(1),axis=0)/a.sum()
    # Uniform triangle second moment, not a vertex-density weighted PCA.
    total=t.sum(1)
    second=(np.einsum('ni,nj->nij',total,total)+np.einsum('nki,nkj->nij',t,t))/12
    cov=np.sum(a[:,None,None]*second,axis=0)/a.sum()-np.outer(centre,centre)
    eig,vec=np.linalg.eigh(cov);axis=vec[:,-1]
    if axis[1]>0:axis=-axis
    return centre,axis,float(eig[-1]/eig[0])


def run(root,source):
    root.mkdir(parents=True,exist_ok=False);plant=Plant();m=plant.model;m.set({})
    shutil.copy2(__file__,root/'frozen-wrist-revision.py')
    old=json.loads((source/'node-revision.json').read_text())['nodes'];nodes=np.array(old).reshape(4,2)
    idx=m.names.index('ArtMesh253');vertices=m.mesh[idx].copy()
    counts=m.api('GetDrawableIndexCounts',C.POINTER(C.c_int),[C.c_void_p])(m.ptr)
    ptr=m.api('GetDrawableIndices',C.POINTER(C.POINTER(C.c_ushort)),[C.c_void_p])(m.ptr)
    tri=np.ctypeslib.as_array(ptr[idx],shape=(counts[idx],)).copy().reshape(-1,3)
    centre,axis,ratio=area_axis(vertices,tri)
    # Intersection at existing palm/forearm overlap height: explicit candidate.
    seam=nodes[2,1];wrist=centre+axis*((seam-centre[1])/axis[1]);palm=wrist+axis*.03
    revised=nodes.copy();revised[2]=wrist;revised[3]=palm
    np.savez(root/'palm-geometry.npz',vertices=vertices,triangles=tri)
    unity.write_spec(root/'protocol.json',{'scope':'wrist-only candidate A/B, not anatomical-axis proof',
        'oldNodes':nodes.tolist(),'newNodes':revised.tolist(),'palmAreaCentre':centre.tolist(),'palmPrincipalAxis':axis.tolist(),
        'eigenRatio':ratio,'seamY':float(seam),'method':'area moments of palm mesh; intersect longitudinal axis with existing .020 seam height',
        'limitations':'palm principal axis is shape estimate, seam height not identified joint; do not equate rigid rotation fit with anatomical correctness',
        'changes':'only W/P nodes; shoulder/elbow nodes identical v1, native neutral; topology/layers unchanged'})
    times=np.arange(181)/30
    knots=[0,35,-35,0,35,-35,0];w=np.interp(times,np.arange(7),knots)
    values=np.zeros((181,6));values[:,5]=w
    unity.write_spec(root/'stimuli.json',{'values':values.tolist()})
    curves=unity.write_spec(root/'curves.json',{'fps':30,'duration':6,'physics':False,'ids':['Param94','Param97','Param99'],
        'channels':[{'values':values[:,j].tolist()} for j in range(3)]})
    for label,points in [('old-wrist',nodes),('axis-wrist',revised)]:
        poses=[{'shoulder':0 if i<90 else 20,'elbow':0 if i<90 else 35,'wrist':float(v),
            'projection':{'upperLength':1,'forearmLength':1,'palmLength':1,'palmWidth':1,'palmShear':0},
            'girdle':{'rootX':0,'rootY':0,'socketX':0,'socketY':0}} for i,v in enumerate(w)]
        patch=unity.write_spec(root/(label+'.json'),{'poses':poses,'nativeMarkers':markers(m,plant.anchors),'correctedPatchNodes':points.ravel().tolist()})
        unity.run('LabMotion.RunMotion',root/label,{'ACTION_LAB_MODEL':unity.model3_path(),'ACTION_LAB_PATCH':patch,'ACTION_LAB_CURVES':curves,'ACTION_LAB_LAYER_FRAME':''},root/(label+'.log'))
        fs=[json.loads((root/label/f'{i:04}.json').read_text(encoding='utf-8-sig')) for i in range(181)]
        unity.write_spec(root/(label+'-checks.json'),{'maxFlips':max(f['wholeArms']['flippedTriangles'] for f in fs),
            'minAreaRatio':min(f['wholeArms']['minTriangleAreaRatio'] for f in fs),
            'maxRestore':max(f['wholeArms']['restoredMeshError'] for f in fs),'maxPairing':max(f['pairingMaxDelta'] for f in fs)})
        print(label,'captured',flush=True)

if __name__=='__main__':run(Path(sys.argv[1]),Path(sys.argv[2]))
