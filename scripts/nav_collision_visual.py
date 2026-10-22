"""Read-only snapshot of the exact current collision primitives and scan.

This is diagnostic data only; it never changes collision or filtering rules.
"""
import numpy as np
from nav_math import transform,inverse
from nav_body_model import primitive_distances

def collision_visual(points,body,parts,fresh=True,limit=1500):
    out=dict(fresh=bool(fresh),count=0,points=[],parts=[],xyz=None,gap=None,display_count=0)
    if not fresh or body is None or not len(parts):return out
    parts=np.asarray(parts);local=transform(np.asarray(points).reshape(-1,3),inverse(body))
    extents=parts[:,4:7].copy();sphere=parts[:,0]==1
    extents[sphere]=parts[sphere,4,None]
    lo=(parts[:,1:4]-extents).min(axis=0);hi=(parts[:,1:4]+extents).max(axis=0)
    ids=np.flatnonzero(np.isfinite(local).all(axis=1)&(local>=lo).all(axis=1)&(local<=hi).all(axis=1))
    if not len(ids):return out
    gaps=primitive_distances(local[ids],parts);ids=ids[gaps<=0];gaps=gaps[gaps<=0]
    if not len(ids):return out
    worst=int(ids[np.argmin(gaps)])
    active=[i for i,p in enumerate(parts) if np.any(primitive_distances(local[ids],[p])<=0)]
    selected=ids[::max(1,int(np.ceil(len(ids)/limit)))];selected=np.unique(np.r_[selected,worst])
    out.update(count=len(ids),points=np.asarray(points)[selected].tolist(),parts=active,
               xyz=local[worst].tolist(),gap=float(gaps.min()),display_count=len(selected))
    return out
