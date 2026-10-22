"""Narrow, explicitly calibrated self-return residuals, never scene denoising.

The capture must be made with empty hands, a fixed posture, and the user's
confirmed 20 cm clear space around the upper body. No online learning.
Unmatched returns, even a single point, remain collision obstacles.
"""
import hashlib
from collections import deque
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from nav_math import transform

RADIUS = .015

def repeated_surface(frames, radius=.012, min_frames=3):
    """Keep only measurements supported by distinct capture frames."""
    if not frames:return np.empty((0,3))
    points=np.concatenate(frames)
    if not len(points):return points
    labels=np.concatenate([np.full(len(p),i) for i,p in enumerate(frames)])
    tree=cKDTree(points)
    keep=np.array([len(np.unique(labels[ids]))>=min_frames
                   for ids in tree.query_ball_point(points,radius)],dtype=bool)
    points=points[keep]
    if not len(points):return points
    # Choose actual measured points; never fill gaps or build a convex hull.
    _,ids=np.unique(np.floor(points/.004).astype(np.int64),axis=0,return_index=True)
    return points[np.sort(ids)]

class ResidualFilter:
    def __init__(self,path,model_hash,calibration_hash,profile_path):
        self.error='';self.tree=None;self.removed=0;self.speckles=0;self.history=deque(maxlen=2);self.last_stamp=None
        try:
            with np.load(path,allow_pickle=False) as f:
                for key,value in dict(model_hash=model_hash,calibration_hash=calibration_hash,
                        profile_hash=hashlib.sha256(Path(profile_path).read_bytes()).hexdigest()).items():
                    if str(f[key])!=value:raise ValueError('模型、雷达外参或夹爪模板已变化')
                if not bool(f['authorised_empty_clear_body']):raise ValueError('缺少空手净空确认')
                if str(f['frame'])!='scan345_base':raise ValueError('残余点模板坐标系不符')
                p=np.asarray(f['points'],float)
                if p.ndim!=2 or p.shape[1]!=3 or not 3<=len(p)<=10000 or not np.isfinite(p).all():
                    raise ValueError('残余点模板无效')
                if np.any(np.abs(p[:,:2])>.8) or np.any((p[:,2]<.55)|(p[:,2]>2.1)):
                    raise ValueError('残余点模板超出已确认上半身范围')
                names=f['joint_names'].tolist();values=f['joint_values'].tolist()
                self.q=dict(zip(names,values));self.grippers=f['gripper_values'].astype(float)
                if len(names)!=21 or len(set(names))!=21 or not np.isfinite(values).all() or self.grippers.shape!=(2,) or not np.isfinite(self.grippers).all():
                    raise ValueError('残余点模板姿态不完整')
                self.tree=cKDTree(p)
                z=np.asarray(f['noise_zone'],float) if 'noise_zone' in f else p
                if z.ndim!=2 or z.shape[1]!=3 or not 3<=len(z)<=10000 or not np.isfinite(z).all() or np.any(np.abs(z[:,:2])>.8) or np.any((z[:,2]<.55)|(z[:,2]>2.1)):
                    raise ValueError('杂点区域无效')
                self.zone=cKDTree(z)
        except (OSError,ValueError,KeyError,TypeError) as e:self.error=str(e)

    def health(self,model):
        if self.error or self.tree is None:return False,'自体残余点标定不可用：'+self.error
        if model.q is None or set(model.q)!=set(self.q):return False,'等待残余点标定姿态'
        if any(abs(model.q[n]-v)>.35 for n,v in self.q.items()):
            return False,'姿态与残余点标定不同，请恢复空手标定姿态'
        if any(abs(model.gripper_q.get(s,float('inf'))-v)>3 for s,v in zip(('left','right'),self.grippers)):
            return False,'夹爪开口与残余点标定不同'
        return True,'空手自体残余点标定有效'

    def mask(self,points,model):
        ok,why=self.health(model)
        if not ok:raise ValueError(why)
        mask=np.zeros(len(points),bool)
        if not len(points):self.removed=0;return mask
        # A tight measured-surface band; no rejection by point count, range,
        # obstacle height, or temporal disappearance in the environment.
        lo=self.tree.data.min(axis=0)-RADIUS;hi=self.tree.data.max(axis=0)+RADIUS
        ids=np.flatnonzero(np.all((points>=lo)&(points<=hi),axis=1))
        if len(ids):
            d,_=self.tree.query(points[ids],distance_upper_bound=RADIUS)
            mask[ids[np.isfinite(d)]]=True
        self.removed=int(mask.sum())
        return mask

    def speckle_mask(self,points,excluded,stamp,body):
        """Radius outliers ONLY in verified empty calibration patches.

        Raw, pre-decimation neighbors preserve thin structures. World-frame
        persistence rescues a sparse return in three consecutive scans. A gap,
        startup or backwards stamp disables rejection. Runtime must slow the
        reference AND commands while such points are being disambiguated.
        Removed rays are not used as free-space evidence.
        """
        mask=np.zeros(len(points),bool);self.speckles=0
        valid=np.flatnonzero(~excluded)
        close=valid[np.all(np.abs(points[valid,:2])<.85,axis=1)&
                    (points[valid,2]>.5)&(points[valid,2]<2.15)]
        world=transform(points[close],body)
        if self.last_stamp is None or not 0<stamp-self.last_stamp<=.25:self.history.clear()
        if len(self.history)==2 and len(close):
            dist,_=self.zone.query(points[close],distance_upper_bound=.06)
            ids=close[np.isfinite(dist)]
            if len(ids):
                tree=cKDTree(points[valid])
                d,_=tree.query(points[ids],k=3,distance_upper_bound=.035)
                isolated=np.sum(np.isfinite(d),axis=1)<3
                current_world=transform(points[ids],body)
                persistent=np.ones(len(ids),bool)
                for previous in self.history:
                    d,_=previous.query(current_world,distance_upper_bound=.015)
                    persistent&=np.isfinite(d)
                mask[ids[isolated&~persistent]]=True
        self.history.append(cKDTree(world));self.last_stamp=stamp
        self.speckles=int(mask.sum())
        return mask
