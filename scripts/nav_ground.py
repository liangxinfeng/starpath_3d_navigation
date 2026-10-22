"""Current-scan floor removal constrained by the saved flat-floor map.

Only a thin band around a spatially supported plane is removed. A coherent
raised surface cannot replace the floor merely because it has many returns.
No temporal obstacle memory is maintained here.
"""
import numpy as np
from scipy.spatial import cKDTree, ConvexHull, Delaunay, QhullError


def supported_low_geometry(xy):
    """Retain a supported lowered patch or extended edge, not three speckles.

    Sample density cannot substitute for physical extent: one representative
    per 5 cm cell, minimum-area hull and width for surfaces. Long narrow edges
    stay protected even when their area degenerates to a line.
    """
    xy=np.asarray(xy,float).reshape(-1,2)
    _,indices=np.unique(np.floor(xy/.05),axis=0,return_index=True)
    p=xy[indices]
    if len(p)<3:return False
    # Translation/rotation invariant dimensions, measured in metres.
    span=float(np.max(np.linalg.norm(p[:,None]-p[None,:],axis=2)))
    if span>=.14:return True
    try:
        hull=ConvexHull(p);v=p[hull.vertices]
        edges=np.roll(v,-1,axis=0)-v;normals=np.column_stack([-edges[:,1],edges[:,0]])
        normals/=np.linalg.norm(normals,axis=1)[:,None]
        width=float(np.min(np.ptp(v@normals.T,axis=0)))
        return hull.volume>=.0025 and width>=.04
    except QhullError:return False


def low_return_outliers(points, heights, ground):
    """Reject sparse below-floor echoes only beneath observed surrounding floor.

    A coherent lowered surface, absent floor, or edge of observed floor retains
    the old obstacle decision. This is an outlier filter, not a cliff override.
    """
    points=np.asarray(points,float).reshape(-1,3);heights=np.asarray(heights,float)
    ground=np.asarray(ground,float).reshape(-1,3);reject=np.zeros(len(points),bool)
    ids=np.flatnonzero((heights<-.08)&(heights>-.35))
    if not len(ids) or len(ground)<3:return reject
    low=points[ids];tree=cKDTree(low[:,:2]);floor=cKDTree(ground[:,:2])
    for i,neighbors in enumerate(tree.query_ball_point(low[:,:2],.15)):
        coherent=np.asarray(neighbors)[np.abs(heights[ids[neighbors]]-heights[ids[i]])<=.04]
        if supported_low_geometry(low[coherent,:2]):continue
        p=low[i,:2]
        if floor.query(p)[0]>.08:continue
        # Use the same maximum 45 cm support span as the saved floor grid.
        # A small nearest-neighbour disk can incorrectly reject a valid sparse
        # ground triangle and leave an obstacle inside an already free cell.
        support=ground[floor.query_ball_point(p,.45),:2]
        if len(np.unique(np.floor(support/.05),axis=0))<3:continue
        try:
            tri=Delaunay(support-p);simplex=int(tri.find_simplex([[0.,0.]])[0])
            if simplex<0:continue
            v=tri.points[tri.simplices[simplex]]
            if np.max(np.linalg.norm(v-np.roll(v,-1,axis=0),axis=1))<=.45:
                reject[ids[i]]=True
        except QhullError:continue
    return reject


def separate_ground(points, prior, center, tolerance=.045):
    points=np.asarray(points,float);prior=np.asarray(prior,float)
    if points.ndim!=2 or points.shape[1]!=3 or prior.shape!=(3,) or not np.isfinite(prior).all():
        raise ValueError('地面参数无效')
    plane=prior.copy();mode='saved_plane';support=0
    delta=points[:,:2]-np.asarray(center)[:2]
    residual=points[:,2]-points[:,:2]@prior[:2]-prior[2]
    candidate=points[(np.linalg.norm(delta,axis=1)<6.)&(np.abs(residual)<.10)]
    if len(candidate)>=100:
        # Equal XY-cell weighting prevents a dense vertical face dominating fit.
        cells=np.floor(candidate[:,:2]/.20).astype(np.int64)
        _,idx=np.unique(cells,axis=0,return_index=True)
        candidate=candidate[idx]
        # Seed from points close to the saved ground, not an arbitrary largest plane.
        d=candidate[:,2]-candidate[:,:2]@prior[:2]-prior[2]
        seed=np.abs(d)<=.045
        design=np.column_stack([candidate[:,:2]-np.asarray(center)[:2],np.ones(len(candidate))])
        reference=np.r_[prior[:2],prior[2]+np.asarray(center)[:2]@prior[:2]]
        coef=reference.copy();good=seed
        if np.count_nonzero(seed)>=25:
            for _ in range(3):
                if np.count_nonzero(good)<25:break
                coef,_,rank,_=np.linalg.lstsq(design[good],candidate[good,2],rcond=None)
                if rank<3:break
                good=np.abs(candidate[:,2]-design@coef)<.025
            xy=candidate[good,:2];support=len(xy)
            spread=np.linalg.eigvalsh(np.cov(xy.T)) if len(xy)>2 else np.zeros(2)
            if (support>=25 and support>=.55*len(candidate) and spread[0]>.06
                    and np.linalg.norm(coef[:2])<np.tan(np.deg2rad(8))
                    and np.linalg.norm(coef[:2]-prior[:2])<np.tan(np.deg2rad(2))
                    and abs(coef[2]-reference[2])<=.04):
                plane=np.r_[coef[:2],coef[2]-np.asarray(center)[:2]@coef[:2]];mode='current_supported_plane'
    h=points[:,2]-points[:,:2]@plane[:2]-plane[2]
    ground=np.abs(h)<=tolerance
    return ground,h,dict(mode=mode,plane=plane.tolist(),support_cells=support,tolerance_m=tolerance,
                         removed_points=int(ground.sum()),input_points=len(points))
