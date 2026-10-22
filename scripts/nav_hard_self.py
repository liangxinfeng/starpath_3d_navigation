"""Fixed-pose model volumes and measured residual cells. No scene learning."""
import hashlib
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from nav_math import transform,inverse

def corners(lo,hi):
    return np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])

def cell_guards(cells):
    """Guard every masked residual cell with an additional 4 cm shell.

    Group only for collision checking; hard rejection still uses individual
    cells. These extra guards make the blind patches explicit to the planner.
    """
    centers=(cells+.5)*.02;keys=np.floor(centers/.08).astype(np.int64)
    parts=[]
    for key in np.unique(keys,axis=0):
        points=centers[np.all(keys==key,axis=1)]
        lo=points.min(0)-.08;hi=points.max(0)+.08
        parts.append([0,*((lo+hi)/2),*((hi-lo)/2)])
    if len(parts)>50:raise ValueError('残余点保护块过多，禁止扩大到整片区域')
    return np.array(parts)

def protected_union(core,extra):
    """Remove only guards entirely contained in ONE existing convex part.

    All eight corners inside a single box/sphere proves containment of the
    whole box. Corners covered by different parts do not prove containment.
    """
    kept=[];core=np.asarray(core).reshape(-1,7);extra=np.asarray(extra).reshape(-1,7)
    # Same containment predicate, vectorized in bounded chunks. The former
    # per-part Python loop was too costly for changing 100 Hz joint feedback.
    signs=corners(np.full(3,-1.),np.full(3,1.))
    for start in range(0,len(extra),32):
        batch=extra[start:start+32];verts=batch[:,None,1:4]+signs[None,:,:]*batch[:,None,4:]
        d=np.abs(verts[:,None,:,:]-core[None,:,None,1:4])
        boxes=np.all(d<=core[None,:,None,4:],axis=-1)
        spheres=np.linalg.norm(d,axis=-1)<=core[None,:,None,4]
        inside=np.where(core[None,:,None,0]==1,spheres,boxes)
        covered=np.any(np.all(inside,axis=2),axis=1)
        kept.extend(batch[~covered])
    return np.vstack([core,np.array(kept).reshape(-1,7)])

class HardSelfFilter:
    def __init__(self,path,model,calibration_hash,profile_path):
        self.error='';self.removed=0;self.mesh_removed=0;self.cells_removed=0;self.parts=[];self.extra_parts=np.empty((0,7))
        try:
            with np.load(path,allow_pickle=False) as f:
                for key,value in dict(model_hash=model.model_hash,calibration_hash=calibration_hash,
                        profile_hash=hashlib.sha256(Path(profile_path).read_bytes()).hexdigest(),
                        mode='hard_voxel_v3',frame='scan345_base').items():
                    if str(f[key])!=value:raise ValueError('模型、外参、模板或裁剪版本不一致')
                if not bool(f['authorised_empty_clear_body']):raise ValueError('缺少净空标定确认')
                c=np.asarray(f['cells'])
                if c.ndim!=2 or c.shape[1]!=3 or not 1<=len(c)<=10000 or not np.isfinite(c).all() or not np.equal(c,np.floor(c)).all():
                    raise ValueError('硬裁剪单元无效')
                p=(c+.5)*.02
                if np.any(np.abs(p[:,:2])>.81) or np.any((p[:,2]<.54)|(p[:,2]>2.12)):
                    raise ValueError('硬裁剪超出已标定范围')
                names=f['joint_names'].tolist();values=f['joint_values'].tolist()
                self.q=dict(zip(names,values));self.grippers=f['gripper_values'].astype(float)
                if len(names)!=21 or len(set(names))!=21 or not np.isfinite(values).all() or self.grippers.shape!=(2,) or not np.isfinite(self.grippers).all():
                    raise ValueError('标定姿态无效')
                self.cells=cKDTree(p);self.extra_parts=cell_guards(c)
            cache={'Link_Zero_Point':np.eye(4)}
            for name,(_,lo,hi) in model.shapes.items():
                t=model.pose(name,self.q,cache)
                ref=corners(lo,hi);inflated=corners(lo-.02,hi+.02);world=transform(inflated,t)
                self.parts.append(dict(name=name,inv=inverse(t),lo=lo-.02,hi=hi+.02,
                    aabb_lo=world.min(0),aabb_hi=world.max(0),corners=ref,ref=transform(ref,t)))
            self.motion_parts=list(self.parts)
            # Include the actual measured DM finger extents, not just the
            # smaller rigid wrist mesh, in the same 8 mm displacement guard.
            for name,tree in getattr(model,'profile_trees',{}).items():
                ref=corners(tree.data.min(0),tree.data.max(0))
                t=model.pose(name,self.q,cache)
                self.motion_parts.append(dict(name=name,corners=ref,ref=transform(ref,t)))
            self.cached_q=None;self.cached_health=None;self.guard_key=None;self.guards=None;self.reference_envelope=None
        except (OSError,ValueError,KeyError,TypeError) as e:self.error=str(e)

    def health(self,model):
        if self.error:return False,'自体硬裁剪不可用：'+self.error
        if model.q is None or set(model.q)!=set(self.q):return False,'等待硬裁剪标定姿态'
        if any(abs(model.gripper_q.get(s,float('inf'))-v)>3 for s,v in zip(('left','right'),self.grippers)):
            return False,'夹爪开口改变，硬裁剪已失效'
        key=tuple(model.q[n] for n in self.q)
        if key==self.cached_q:return self.cached_health
        # Head servo oscillation is judged by actual geometry displacement.
        # Retain the extra tight angular limit for arms, waist and legs.
        ok=not any(abs(model.q[n]-v)>.35 for n,v in self.q.items() if not n.startswith('Joint_Neck_'))
        cache={'Link_Zero_Point':np.eye(4)}
        if ok:
            for part in self.motion_parts:
                current=transform(part['corners'],model.pose(part['name'],model.q,cache))
                if np.max(np.linalg.norm(current-part['ref'],axis=1))>.008:ok=False;break
        result=(True,'分部位模型外扩20毫米＋局部硬裁剪') if ok else (False,'姿态偏离硬裁剪标定，请恢复空手固定姿态')
        self.cached_q=key;self.cached_health=result
        return result

    def mask(self,points,model):
        ok,why=self.health(model)
        if not ok:raise ValueError(why)
        mask=np.zeros(len(points),bool)
        ids=np.flatnonzero((np.linalg.norm(points[:,:2],axis=1)<1.3)&(points[:,2]<2.3)&(points[:,2]>-.2))
        p=points[ids];inside=np.zeros(len(ids),bool)
        for part in self.parts:
            candidate=np.flatnonzero((~inside)&np.all((p>=part['aabb_lo'])&(p<=part['aabb_hi']),axis=1))
            if len(candidate):
                local=transform(p[candidate],part['inv'])
                inside[candidate[np.all((local>=part['lo'])&(local<=part['hi']),axis=1)]]=True
        self.mesh_removed=int(inside.sum())
        remain=np.flatnonzero(~inside)
        if len(remain):
            d,_=self.cells.query(p[remain],p=np.inf,distance_upper_bound=.040000001)
            inside[remain[np.isfinite(d)]]=True
        mask[ids[inside]]=True;self.removed=int(mask.sum());self.cells_removed=self.removed-self.mesh_removed
        return mask

    def protected_parts(self,core):
        key=core.tobytes()
        if key!=self.guard_key:
            self.guards=protected_union(core,self.extra_parts);self.guard_key=key
        return self.guards

    def collision_parts(self,model):
        """Fixed calibrated envelope covering the permitted 8 mm vibration.

        Do not repeatedly re-quantize boxes on small feedback changes: a one
        voxel growth would otherwise cancel a valid trajectory at start-up.
        Real pose displacement still invalidates this envelope in health().
        """
        if self.reference_envelope is None:
            previous=model.primitives
            try:
                model.primitives=np.empty((0,7))
                model.update(self.q)
                core=model.primitives.copy()
                core[:,4:]+=.008
            finally:
                model.primitives=previous
            self.reference_envelope=protected_union(core,self.extra_parts)
        return self.reference_envelope
