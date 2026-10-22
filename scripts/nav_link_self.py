"""Link-attached self masks and corresponding collision protection volumes.

This module sends no motion commands. Source-time feedback and planner shape
acknowledgements are checked by the navigation runtime.
"""
import numpy as np
import time
import hashlib
from pathlib import Path
from scipy.spatial import cKDTree
from nav_math import transform,inverse
from nav_hard_self import HardSelfFilter,corners,protected_union


class LinkSelfFilter:
    def __init__(self,path,model,calibration_hash,profile_path,padding=.025,extra_path=''):
        if not .02<=padding<=.03:raise ValueError('Rigid-link padding outside verified design bounds')
        self.reference=HardSelfFilter(path,model,calibration_hash,profile_path)
        if self.reference.error:raise ValueError(self.reference.error)
        self.model=model;self.padding=padding;self.parts=[];self.cells=[];self.binding=[]
        self.extra_cells=[];self.verified_cells=[];self.transforms={};self.q=None;self.profile_enabled=False
        self.removed=0;self.mesh_removed=0;self.cells_removed=0
        cache={'Link_Zero_Point':np.eye(4)}
        for name,(_,lo,hi) in model.shapes.items():
            self.parts.append((name,lo-padding,hi+padding))
            model.pose(name,self.reference.q,cache)
        # A measured wrist template also contains stationary forearm/torso
        # returns. Those must not rotate with the wrist: remove all points
        # already explained by reference rigid-link geometry first.
        self.gripper_trees={}
        for name,tree in model.profile_trees.items():
            t=model.pose(name,self.reference.q,cache);world=transform(tree.data,t)
            known=np.zeros(len(world),bool)
            for link,lo,hi in self.parts:
                local=transform(world,inverse(model.pose(link,self.reference.q,cache)))
                known|=np.all((local>=lo)&(local<=hi),axis=1)
            independent=tree.data[~known]
            if len(independent)<20:raise ValueError('Insufficient independently measured gripper surface')
            self.gripper_trees[name]=cKDTree(independent)
        # Bind each already-authorised residual cube to a specific physical
        # link in the reference posture. Never retain it in base coordinates.
        for p in self.reference.cells.data:
            candidates=[]
            for name,(tree,_,_) in model.shapes.items():
                t=model.pose(name,self.reference.q,cache)
                d=float(tree.query(transform([p],inverse(t)))[0][0])
                candidates.append((d,name,t))
            for name,tree in self.gripper_trees.items():
                t=model.pose(name,self.reference.q,cache)
                d=float(tree.query(transform([p],inverse(t)))[0][0])
                candidates.append((d,name,t))
            d,name,t=min(candidates,key=lambda x:x[0])
            if d>.15:raise ValueError('Residual cell has no supported link assignment')
            # Preserve the ORIGINAL base-aligned cube orientation at reference.
            # The entire cube subsequently rotates with its owner link.
            local=inverse(t);self.cells.append((name,p.copy(),local))
            self.binding.append((name,d,p.tolist()))
        self.build_guard_cells()
        if extra_path:
            with np.load(extra_path,allow_pickle=False) as a:
                for key,value in dict(model_hash=model.model_hash,calibration_hash=calibration_hash,profile_hash=hashlib.sha256(Path(profile_path).read_bytes()).hexdigest()).items():
                    if str(a[key])!=value:raise ValueError('补充夹爪裁剪标定与当前模型不一致')
                if not bool(a['authorised_empty_clear_grippers']):raise ValueError('补充夹爪裁剪缺少空手净空确认')
                points=a['points'];q=dict(zip(a['joint_names'].tolist(),a['joint_values'].tolist()))
                if points.ndim!=2 or points.shape[1]!=3 or not 1<=len(points)<=100 or not np.isfinite(points).all() or set(q)!=set(self.reference.q) or not np.isfinite(list(q.values())).all():
                    raise ValueError('补充夹爪裁剪数据无效')
                self.add_verified_gripper_cells(points,q)
                if 'verified_patch_points' in a.files:
                    if not bool(a['authorised_head_right_gripper_clear']):
                        raise ValueError('局部自体补丁缺少现场净空确认')
                    patch_q=dict(zip(a['joint_names'].tolist(),a['patch_joint_values'].tolist()))
                    self.add_verified_link_patches(a['verified_patch_points'],a['verified_patch_links'].tolist(),patch_q)

    def add_verified_link_patches(self,points,links,q):
        """Operator-confirmed head/right-gripper echoes only; never learn online.

        Each measured point gets a 3.5 cm half-width cube on its actual link.
        A 7.5 cm outer guard preserves collision protection around it.
        """
        allowed={'Link_Neck_Yaw_to_Head','Link_Right_Wrist_Lower_to_Gripper'}
        points=np.asarray(points,dtype=float)
        if points.ndim!=2 or points.shape[1]!=3 or not 1<=len(points)<=8 or len(links)!=len(points) or not np.isfinite(points).all():
            raise ValueError('已确认局部回波数据无效')
        if set(q)!=set(self.reference.q) or not np.isfinite(list(q.values())).all() or any(abs(v)>360 for v in q.values()):
            raise ValueError('已确认局部回波缺少有效标定姿态')
        if not set(links)<=allowed:raise ValueError('局部回波不属于已确认的头部或右夹爪')
        cache={'Link_Zero_Point':np.eye(4)};pending=[]
        for p,name in zip(points,links):
            if name not in self.model.shapes:raise ValueError('局部回波对应连杆不存在')
            t=self.model.pose(name,q,cache);local=transform([p],inverse(t))[0]
            tree=self.gripper_trees.get(name,self.model.shapes[name][0])
            if float(tree.query(local)[0])>.12:raise ValueError('局部回波偏离已知自体表面过远')
            pending.append((name,p.copy(),inverse(t)))
        for name in set(links):
            group=points[np.array(links)==name]
            if np.max(np.ptp(group,axis=0))>.05:raise ValueError('局部回波补丁范围过大')
        self.verified_cells=getattr(self,'verified_cells',[])+pending
        self.build_guard_cells()

    def build_guard_cells(self):
        groups={}
        for cells,half in ((self.cells,.08),(self.extra_cells,.08),(getattr(self,'verified_cells',[]),.075)):
            for name,center,ref_inv in cells:
                local_center=transform([center],ref_inv)[0]
                key=(name,*np.floor(local_center/.08).astype(int))
                groups.setdefault(key,[]).append(transform(corners(center-half,center+half),ref_inv))
        self.guard_cells=[]
        for key,chunks in groups.items():
            p=np.concatenate(chunks);self.guard_cells.append((key[0],corners(p.min(0),p.max(0))))

    def prepare(self,q,grippers):
        if set(q)!=set(self.reference.q) or not np.isfinite(list(q.values())).all():
            raise ValueError('Incomplete/nonfinite joint pose')
        if any(abs(v)>360 for v in q.values()):raise ValueError('Invalid joint angles')
        cache={'Link_Zero_Point':np.eye(4)}
        for name,_,_ in self.parts:self.model.pose(name,q,cache)
        for name,_,_ in self.cells+self.extra_cells+getattr(self,'verified_cells',[]):self.model.pose(name,q,cache)
        self.transforms=cache;self.q=dict(q)
        self.profile_enabled=all(abs(grippers.get(side,float('inf'))-v)<=3
            for side,v in zip(('left','right'),self.reference.grippers))
        def bounded(t,lo,hi):
            center=(lo+hi)/2;half=(hi-lo)/2
            world=transform([center],t)[0];extent=np.abs(t[:3,:3])@half
            return inverse(t),lo,hi,world-extent,world+extent
        self.prepared_parts=[bounded(cache[name],lo,hi) for name,lo,hi in self.parts]
        self.prepared_grippers=[]
        if self.profile_enabled:
            if not hasattr(self,'gripper_bounds'):
                self.gripper_bounds={name:(tree.data.min(0)-.045,tree.data.max(0)+.045) for name,tree in self.gripper_trees.items()}
            for name,tree in self.gripper_trees.items():
                lo,hi=self.gripper_bounds[name];self.prepared_grippers.append((tree,bounded(cache[name],lo,hi)))
        self.prepared_cells=[]
        for cells,half in ((self.cells,.04),(self.extra_cells,.04),(getattr(self,'verified_cells',[]),.035)):
            for name,center,ref_inv in cells:
                if name in self.model.profile_trees and not self.profile_enabled:continue
                self.prepared_cells.append(bounded(cache[name]@ref_inv,center-half,center+half))
        bounds=self.prepared_parts+self.prepared_cells+[p for _,p in self.prepared_grippers]
        self.global_low=np.min([p[3] for p in bounds],axis=0) if bounds else np.zeros(3)
        self.global_high=np.max([p[4] for p in bounds],axis=0) if bounds else np.zeros(3)

    def health(self,model):
        if model.q is None or set(model.q)!=set(self.reference.q):return False,'动态模型关节集合不完整'
        if not np.isfinite(list(model.q.values())).all():return False,'动态模型关节值无效'
        if any(abs(model.gripper_q.get(s,float('inf'))-v)>3 for s,v in zip(('left','right'),self.reference.grippers)):
            return False,'DM夹爪开口超出已标定范围'
        return True,'连杆过滤随实时姿态更新'

    def mask_registered(self,points,model,stamp):
        # FAST-LIO's registered cloud has no reliable arm-point acquisition
        # timestamps. Only erase returns inside the masks at all three sampled
        # postures; do not expand a mask over the swept volume. Raw per-point
        # filtering upstream remains the primary remover of moving self returns.
        mask=np.ones(len(points),bool)
        for t in (stamp-.1,stamp-.05,stamp):
            self.prepare(model.at_scan(t),model.gripper_q);mask&=self.mask(points)
        self.removed=int(mask.sum());self.mesh_removed=self.removed;self.cells_removed=0
        self.prepare(model.q,model.gripper_q)
        return mask

    def add_verified_gripper_cells(self,points,q):
        """Only supplied static, empty-hand, externally clear calibration data.

        This is never called by live mask(), and must never learn obstacles.
        A cube extends 2 cm around an observed return, with an outer collision
        guard. No expansion to a whole hand-sized sphere or height band.
        """
        cache={'Link_Zero_Point':np.eye(4)};result=[]
        for p in points:
            choices=[]
            for name,tree in self.gripper_trees.items():
                t=self.model.pose(name,q,cache);local=transform([p],inverse(t))[0]
                d=float(tree.query(local)[0]);choices.append((d,name,t,local))
            d,name,t,local=min(choices,key=lambda x:x[0])
            if d>.10:continue
            # Need to be close to actual measured finger geometry, not merely
            # close to the much larger unknown-finger collision sphere.
            result.append((name,np.asarray(p).copy(),inverse(t)))
        self.extra_cells=result
        self.build_guard_cells()

    def mask(self,points):
        if self.q is None:raise ValueError('prepare() required')
        points=np.asarray(points);mask=np.zeros(len(points),bool)
        ids=np.flatnonzero(np.all((points>=self.global_low-1e-9)&(points<=self.global_high+1e-9),axis=1))
        if not len(ids):return mask
        p=points[ids];inside=np.zeros(len(ids),bool)
        for inv,lo,hi,world_lo,world_hi in self.prepared_parts+self.prepared_cells:
            use=np.flatnonzero((~inside)&np.all((p>=world_lo-1e-9)&(p<=world_hi+1e-9),axis=1))
            if not len(use):continue
            local=transform(p[use],inv);inside[use[np.all((local>=lo-1e-9)&(local<=hi+1e-9),axis=1)]]=True
        for tree,(inv,lo,hi,world_lo,world_hi) in self.prepared_grippers:
            use=np.flatnonzero((~inside)&np.all((p>=world_lo)&(p<=world_hi),axis=1))
            if len(use):
                distance,_=tree.query(transform(p[use],inv),distance_upper_bound=.045)
                inside[use[np.isfinite(distance)]]=True
        mask[ids[inside]]=True
        return mask

    def collision_parts(self):
        if self.q is None:raise ValueError('prepare() required')
        old=self.model.primitives
        try:
            self.model.primitives=np.empty((0,7));self.model.update(self.q)
            core=self.model.primitives.copy()
        finally:self.model.primitives=old
        extra=[]
        if hasattr(self,'guard_cells'):
            for name,verts in self.guard_cells:
                if name in self.model.profile_trees and not self.profile_enabled:continue
                p=transform(verts,self.transforms[name])
                lo=np.floor(p.min(0)/.01)*.01;hi=np.ceil(p.max(0)/.01)*.01
                extra.append([0,*((lo+hi)/2),*((hi-lo)/2)])
        else:
            for cells,half in ((self.cells,.08),(self.extra_cells,.08),(getattr(self,'verified_cells',[]),.075)):
                for name,center,ref_inv in cells:
                    if name in self.model.profile_trees and not self.profile_enabled:continue
                    p=transform(corners(center-half,center+half),self.transforms[name]@ref_inv)
                    lo=p.min(0);hi=p.max(0);extra.append([0,*((lo+hi)/2),*((hi-lo)/2)])
        # Stable per-link ordering lets the handoff bound the intervening
        # motion. Pose-dependent containment pruning would scramble identities.
        return np.vstack([core,np.array(extra).reshape(-1,7)])
