"""Versioned patch-axis revision from mesh cross-sections, with identical-input A/B."""
import sys,json,ctypes as C,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from tools.control.pilot import Plant
from tools.control.node_overlay_trial import stimuli
from tools.whole.batch import markers
from tools.expression import unity


def section_midpoint(vertices,triangles,y):
    hits=[]
    for triangle in triangles:
        for a,b in ((0,1),(1,2),(2,0)):
            p,q=vertices[triangle[a]],vertices[triangle[b]]
            if abs(q[1]-p[1])>1e-8 and min(p[1],q[1])<=y<=max(p[1],q[1]):
                hits.append(float(p[0]+(q[0]-p[0])*(y-p[1])/(q[1]-p[1])))
    if not hits:raise ValueError('Section misses mesh')
    return [(min(hits)+max(hits))/2,float(y)], [min(hits),max(hits)]


def run(root):
    root.mkdir(parents=True,exist_ok=False);plant=Plant();m=plant.model;m.set({})
    before=root/'before';before.mkdir()
    for path in [Path(__file__),Path('preview/unity/RecoveredShoulderConnection.cs'),Path('preview/unity/WholeArmSession.cs')]:shutil.copy2(path,before/path.name)
    counts=np.ctypeslib.as_array(m.api('GetDrawableIndexCounts',C.POINTER(C.c_int),[C.c_void_p])(m.ptr),shape=(m.n,))
    pointers=m.api('GetDrawableIndices',C.POINTER(C.POINTER(C.c_ushort)),[C.c_void_p])(m.ptr)
    evidence=[];nodes=[]
    for mesh,y in [('ArtMesh251',.330),('ArtMesh261',.205),('ArtMesh253',.020),('ArtMesh253',.005)]:
        i=m.names.index(mesh);v=m.mesh[i].copy();tri=np.ctypeslib.as_array(pointers[i],shape=(int(counts[i]),)).copy().reshape(-1,3)
        point,span=section_midpoint(v,tri,y);nodes+=point
        evidence.append({'mesh':mesh,'sectionY':y,'xInterval':span,'point':point,
            'method':'mesh cross-section midpoint; section height is explicit design choice, not recovered native pivot'})
        np.savez(root/(mesh+'-'+str(y)+'.npz'),vertices=v,triangles=tri)
    values=stimuli();old=values.copy()
    # Original native inputs unchanged. Enlarge patch only; retain negative elbow tests.
    values[:,3]*=2.1;values[:,4]*=3.;values[:,5]*=2.5
    unity.write_spec(root/'node-revision.json',{'version':'patch-axis-cross-section-v1','nodes':nodes,'evidence':evidence,
        'scope':'candidate actuator axes AND readback revised together, not overlay-only displacement'})
    unity.write_spec(root/'protocol.json',{'seed':11809,'frames':361,'fps':30,'physics':False,
        'variants':['old-axis-large','revised-axis-large'],'native':'identical original inputs; no binding changes',
        'patchAmplification':[2.1,3.,2.5],'actualRanges':[[float(values[:,j].min()),float(values[:,j].max())] for j in range(3,6)],
        'comparison':'same amplified trajectory, only patch axis bindings differ; former small version remains previous run',
        'limitation':'candidate sectional nodes require visual approval, not guaranteed anatomical joints; topology/layers unchanged'})
    unity.write_spec(root/'stimuli.json',{'values':values.tolist(),'previousValues':old.tolist()})
    curves=unity.write_spec(root/'curves.json',{'fps':30,'duration':12,'physics':False,'ids':['Param94','Param97','Param99'],
        'channels':[{'values':values[:,j].tolist()} for j in range(3)]})
    poses=[{'shoulder':float(v[3]),'elbow':float(v[4]),'wrist':float(v[5]),
        'projection':{'upperLength':1,'forearmLength':1,'palmLength':1,'palmWidth':1,'palmShear':0},
        'girdle':{'rootX':0,'rootY':0,'socketX':0,'socketY':0}} for v in values]
    for label in ('old-axis-large','revised-axis-large'):
        patch={'poses':poses,'nativeMarkers':markers(m,plant.anchors),'layerPolicy':'legacy-z'}
        if label=='revised-axis-large':patch['correctedPatchNodes']=nodes
        path=unity.write_spec(root/(label+'.json'),patch)
        unity.run('LabMotion.RunMotion',root/label,{'ACTION_LAB_MODEL':unity.model3_path(),'ACTION_LAB_CURVES':curves,
            'ACTION_LAB_PATCH':path,'ACTION_LAB_LAYER_FRAME':''},root/(label+'.log'))
        frames=[json.loads((root/label/f'{i:04}.json').read_text(encoding='utf-8-sig')) for i in range(361)]
        unity.write_spec(root/(label+'-checks.json'),{'maxPairing':max(f['pairingMaxDelta'] for f in frames),
            'maxRestore':max(f['wholeArms']['restoredMeshError'] for f in frames),
            'maxFlips':max(f['wholeArms']['flippedTriangles'] for f in frames),
            'minAreaRatio':min(f['wholeArms']['minTriangleAreaRatio'] for f in frames)})
        print(label,'captured',flush=True)

if __name__=='__main__':run(Path(sys.argv[1]))
