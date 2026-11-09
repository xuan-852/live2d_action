import json
from pathlib import Path
import numpy as np
from .core_model import Model

class Anchors:
    def __init__(self,m):
        self.m=m;m.set({});self.base=m.vertices();self.weights={};self.evidence={}
        # Bind a hinge to the rigid child at its pivot. The upper sleeve also
        # deforms with elbow flexion, so it must not proxy the elbow landmark.
        for name,param,probeMesh,bindMesh in [('shoulder','Param94',149,161),('elbow','Param97',149,149),('wrist','Param99',147,147)]:
            x=self.base[probeMesh].astype(float);m.set({param:3});y=m.mesh[probeMesh].astype(float)
            coeff=np.linalg.lstsq(np.c_[x,np.ones(len(x))],y,rcond=None)[0]
            pivot=np.linalg.solve(np.eye(2)-coeff[:2].T,coeff[2])
            residual=float(np.sqrt(np.mean((np.c_[x,np.ones(len(x))]@coeff-y)**2)))
            b=self.base[bindMesh].astype(float)
            # Affine material landmark, not a hand centroid pretending to be a joint.
            nearest=np.argsort(np.linalg.norm(b-pivot,axis=1))[:12]
            a=np.vstack([b[nearest].T,np.ones(len(nearest))]);w=np.linalg.lstsq(a,np.r_[pivot,1],rcond=None)[0]
            self.weights[name]=(bindMesh,nearest,w)
            self.evidence[name]={'pivot':pivot.tolist(),'probeParameter':param,'probeMesh':m.names[probeMesh],'bindMesh':m.names[bindMesh],'vertexIndices':nearest.tolist(),'weights':w.tolist(),'affineResidual':residual}
        m.set({})
    def read(self):
        pts={n:self.m.mesh[mesh][i].T@w for n,(mesh,i,w) in self.weights.items()}
        pts['palm']=self.m.mesh[147].mean(0).astype(float)
        return pts
