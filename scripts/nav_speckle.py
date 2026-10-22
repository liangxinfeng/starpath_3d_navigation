"""Bounded spatial/temporal filtering for transient isolated obstacle returns.

Input coordinates must be continuous odometry, never the moving body frame.
Only current points are output; previous scans confirm evidence but cannot
leave historical obstacles behind. Startup and timestamp gaps retain all data.
"""
import time
import numpy as np
from scipy.spatial import cKDTree


class TransientSpeckleFilter:
    def __init__(self):
        self.previous=None;self.stamp=None;self.stats={}

    def filter(self,points,stamp):
        start=time.monotonic();p=np.asarray(points,float).reshape(-1,3)
        if not np.isfinite(p).all() or not np.isfinite(stamp):raise ValueError('孤点过滤输入无效')
        tree=cKDTree(p) if len(p) else None
        keep=np.ones(len(p),bool);spatial=np.zeros(len(p),bool);temporal=np.zeros(len(p),bool)
        dt=None if self.stamp is None else float(stamp-self.stamp)
        valid=dt is not None and .04<=dt<=.25
        if valid and len(p):
            # Five returns in a 10 cm sphere are immediately supported.
            # Bounded five-neighbour query avoids enumerating dense surfaces.
            distances,_=tree.query(p,k=5,distance_upper_bound=.10)
            spatial=np.isfinite(distances[:,-1])
            sparse=np.flatnonzero(~spatial)
            if len(sparse) and self.previous is not None:
                distances,_=self.previous.query(p[sparse],distance_upper_bound=.04)
                temporal[sparse]=np.isfinite(distances)
            keep=spatial|temporal
        self.previous=tree;self.stamp=float(stamp)
        self.stats=dict(mode='spatial_or_consecutive_scan' if valid else 'retain_all_warmup_or_gap',
            input_points=len(p),kept_points=int(keep.sum()),removed_points=int((~keep).sum()),
            spatial_points=int(spatial.sum()),persistent_sparse_points=int(temporal.sum()),
            radius_m=.10,min_points=5,temporal_match_m=.04,scan_gap_s=dt,
            processing_ms=round((time.monotonic()-start)*1000,2))
        return keep
