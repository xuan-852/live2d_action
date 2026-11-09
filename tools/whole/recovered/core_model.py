"""Local read-only Cubism Core forward model. No Unity runtime, physics or memory IO."""
import ctypes as C, json, hashlib
from pathlib import Path
import numpy as np

F=C.POINTER(C.c_float); N=C.POINTER(C.c_int); S=C.POINTER(C.c_char_p)
class Model:
    def __init__(self, model_dir, core_dll):
        MODEL=Path(model_dir)
        self.dll=C.CDLL(str(core_dll)); self.buffers=[]
        def api(name,ret,args):
            fn=getattr(self.dll,'csm'+name);fn.restype=ret;fn.argtypes=args;return fn
        self.api=api
        raw=(MODEL/'符玄.moc3').read_bytes(); self.sha=hashlib.sha256(raw).hexdigest()
        mem=self.alloc(len(raw));C.memmove(mem,raw,len(raw))
        moc=api('ReviveMocInPlace',C.c_void_p,[C.c_void_p,C.c_uint])(mem,len(raw))
        if not moc:raise RuntimeError('MOC revive failed')
        size=api('GetSizeofModel',C.c_uint,[C.c_void_p])(moc)
        self.ptr=api('InitializeModelInPlace',C.c_void_p,[C.c_void_p,C.c_void_p,C.c_uint])(moc,self.alloc(size),size)
        if not self.ptr:raise RuntimeError('Model initialize failed')
        self.update=api('UpdateModel',None,[C.c_void_p])
        def get(name,ret):return api(name,ret,[C.c_void_p])(self.ptr)
        self.count=get('GetParameterCount',C.c_int)
        self.ids=[get('GetParameterIds',S)[i].decode() for i in range(self.count)]
        self.idx={n:i for i,n in enumerate(self.ids)}
        self.params=np.ctypeslib.as_array(get('GetParameterValues',F),shape=(self.count,))
        self.defaults=np.ctypeslib.as_array(get('GetParameterDefaultValues',F),shape=(self.count,)).copy()
        self.lo=np.ctypeslib.as_array(get('GetParameterMinimumValues',F),shape=(self.count,)).copy()
        self.hi=np.ctypeslib.as_array(get('GetParameterMaximumValues',F),shape=(self.count,)).copy()
        self.n=get('GetDrawableCount',C.c_int)
        self.names=[get('GetDrawableIds',S)[i].decode() for i in range(self.n)]
        self.counts=np.ctypeslib.as_array(get('GetDrawableVertexCounts',N),shape=(self.n,)).copy()
        ptrs=get('GetDrawableVertexPositions',C.POINTER(F))
        self.mesh=[np.ctypeslib.as_array(ptrs[i],shape=(int(self.counts[i])*2,)).reshape(-1,2) for i in range(self.n)]
        self.opacity=np.ctypeslib.as_array(get('GetDrawableOpacities',F),shape=(self.n,))
        self.parents=np.ctypeslib.as_array(get('GetDrawableParentPartIndices',N),shape=(self.n,)).copy()
        nparts=get('GetPartCount',C.c_int);self.parts=[get('GetPartIds',S)[i].decode() for i in range(nparts)]
        cdi=json.loads((MODEL/'符玄.cdi3.json').read_text(encoding='utf-8-sig'))
        self.labels={x['Id']:x.get('Name',x['Id']) for group in ['Parts','Parameters'] for x in cdi.get(group,[])}
        self.set({})
    def alloc(self,size):
        buf=C.create_string_buffer(size+64);self.buffers.append(buf);return (C.addressof(buf)+63)&~63
    def set(self,values):
        self.params[:]=self.defaults
        for k,v in values.items():
            i=self.idx[k]
            if not np.isfinite(v) or v<self.lo[i]-1e-5 or v>self.hi[i]+1e-5:raise ValueError((k,v))
            self.params[i]=v
        self.update(self.ptr)
    def vertices(self):return [m.copy() for m in self.mesh]
