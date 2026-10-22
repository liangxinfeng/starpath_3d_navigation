"""Saved-map segmentation plus current-scan-only live occupancy.

For the existing flat-floor S3 test mode. A supported floor plane is estimated
from mapping-trajectory neighbourhoods, then applied to the WHOLE saved cloud.
Saved walls and unknown areas are preserved; live scans do not accumulate.
"""
import hashlib,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree,Delaunay,QhullError
from scipy.ndimage import distance_transform_edt


def load_bundle(path,imu_base):
    path=Path(path).resolve();m=json.loads(path.read_text())
    if m.get('format')!='airy-keyframes-v1' or m.get('frame')!='airy_global':
        raise ValueError('需要 airy_global 坐标的完整关键帧地图')
    cloud=(path.parent/m['cloud']).resolve()
    if cloud.parent!=path.parent or cloud.stat().st_size>32_000_000:
        raise ValueError('地图文件位置或大小异常')
    raw=cloud.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=m.get('cloud_sha256'):raise ValueError('地图点云校验失败')
    marker=b'DATA binary\n';end=raw.find(marker)
    if end<0 or end>4096:raise ValueError('仅支持完整二进制 XYZ PCD')
    header=dict(line.split(' ',1) for line in raw[:end].decode('ascii').splitlines() if ' ' in line)
    if any(header.get(k)!=v for k,v in {'FIELDS':'x y z','SIZE':'4 4 4','TYPE':'F F F','COUNT':'1 1 1','HEIGHT':'1'}.items()):
        raise ValueError('地图 PCD 字段不匹配')
    points=np.frombuffer(raw[end+len(marker):],dtype='<f4').reshape(-1,3).copy()
    if len(points)!=int(header['POINTS']) or len(points)!=int(m['point_count']) or len(points)<100:
        raise ValueError('地图点数不完整')
    if not np.isfinite(points).all():raise ValueError('地图含无效坐标')
    poses=np.asarray([k['pose'] for k in m['keyframes']],float)
    if poses.ndim!=3 or poses.shape[1:]!=(4,4) or not len(poses) or not np.isfinite(poses).all():
        raise ValueError('地图缺少有效关键帧位姿')
    # Keyframe poses are T_map_level, NOT T_map_imu. Ignoring the persisted
    # gravity-alignment rotation changes base height by ~0.49 m on this AIRY.
    levels=[]
    for k in m['keyframes']:
        f=(path.parent/k['file']).resolve()
        if not f.is_relative_to(path.parent/'keyframes') or f.stat().st_size>16_000_000:
            raise ValueError('关键帧文件位置或大小异常')
        if hashlib.sha256(f.read_bytes()).hexdigest()!=k.get('sha256'):
            raise ValueError('关键帧坐标变换校验失败')
        with np.load(f,allow_pickle=False) as data:level=np.array(data['imu_to_level'],float)
        if level.shape!=(4,4) or not np.isfinite(level).all() or not np.allclose(level[3],[0,0,0,1],atol=1e-5):
            raise ValueError('关键帧重力对齐变换无效')
        levels.append(level)
    bases=(poses@np.array(levels)@imu_base)[:,:3,3]
    return points,bases,dict(map=str(path),map_id=m.get('map_id',''),cloud_sha256=m['cloud_sha256'])


def fit_floor(points,bases):
    dist,idx=cKDTree(bases[:,:2]).query(points[:,:2])
    rel=points[:,2]-bases[idx,2]
    # Software base reference is not necessarily the physical upper chassis face.
    # Find the measured floor rather than assuming the user-measured deck height.
    near=(dist<.65)&(rel>-.8)&(rel<.2)
    hist,edges=np.histogram(rel[near],np.arange(-.8,.221,.02))
    if hist.max(initial=0)<15:raise ValueError('关键帧附近没有足够地面证据')
    offset=float(edges[np.argmax(hist)]+.01)
    seed=points[near&(np.abs(rel-offset)<.07)].astype(float)
    if len(seed)<80:raise ValueError('地面拟合点不足')
    # Equal-area support prevents a dense close return dominating the plane.
    _,cell,inverse_cell=np.unique(np.floor(seed[:,:2]/.1).astype(np.int64),axis=0,return_index=True,return_inverse=True)
    weights=1./np.bincount(inverse_cell)[inverse_cell]
    # Robust height regression; seed selection prevents a desktop winning RANSAC.
    design=np.column_stack([seed[:,:2],np.ones(len(seed))]);keep=np.ones(len(seed),bool)
    for _ in range(5):
        if np.count_nonzero(keep)<60:raise ValueError('地面拟合不稳定')
        w=np.sqrt(weights[keep])
        coef,_,rank,_=np.linalg.lstsq(design[keep]*w[:,None],seed[keep,2]*w,rcond=None)
        if rank<3:raise ValueError('地面证据退化，无法拟合平面')
        keep=np.abs(seed[:,2]-design@coef)<.045
    if np.linalg.norm(coef[:2])>np.tan(np.deg2rad(2)):
        raise ValueError('地面坡度超过本平地导航测试的支持范围')
    if keep.mean()<.7:raise ValueError('地面不满足单层平面假设，需要分层地形建模')
    return coef,offset,float(np.quantile(np.abs(seed[keep,2]-design[keep]@coef),.95))


class StaticMap:
    def __init__(self,points,bases,resolution=.05):
        self.res=float(resolution)
        self.plane,self.base_offset,self.floor_error=fit_floor(points,bases)
        self.top=-self.base_offset+.37  # matches the original chassis-height slice
        height=self.heights(points)
        self.ground=points[np.abs(height)<=.045]
        # Flat-floor mode: below-floor returns are not negative obstacles.
        # Excluding them does not create free cells: only supported ground does.
        self.rejected_low_points=points[height<-.045]
        self.obstacles=points[self.obstacle_mask(height)]
        if len(self.ground)<100:raise ValueError('完整地图内有效地面不足')
        relevant=np.concatenate([self.ground,self.obstacles])
        self.origin=np.floor((relevant[:,:2].min(axis=0)-.5)/self.res)*self.res
        size=np.ceil((relevant[:,:2].max(axis=0)+.5-self.origin)/self.res).astype(int)
        self.width,self.height=map(int,size)
        if np.any(size<=0) or np.any(size>4000) or size.prod()>2_000_000:raise ValueError('全图栅格超出资源范围')
        self.grid=np.full((self.height,self.width),-1,dtype=np.int8)
        # Interpolate only short supported ground triangles, never the whole hull.
        _,indices=np.unique(np.floor(self.ground[:,:2]/.04).astype(int),axis=0,return_index=True)
        xy=self.ground[indices,:2]
        try:tri=Delaunay(xy)
        except QhullError as e:raise ValueError('地面覆盖不足以生成导航地图') from e
        vertices=xy[tri.simplices]
        edges=np.stack([np.linalg.norm(vertices[:,i]-vertices[:,(i+1)%3],axis=1) for i in range(3)])
        valid=edges.max(axis=0)<=.45
        yy,xx=np.indices(self.grid.shape);centers=np.column_stack([xx.ravel()+.5,yy.ravel()+.5])*self.res+self.origin
        simplex=tri.find_simplex(centers)
        supported=(simplex>=0)&valid[np.maximum(simplex,0)]
        self.grid.ravel()[supported]=0
        # An observed point also supports its own cell; no radial expansion.
        self.grid.ravel()[self.cells(self.ground)]=0
        self.grid.ravel()[self.cells(self.obstacles)]=100
        self.grid.setflags(write=False)

    def heights(self,points):
        a=np.asarray(points)
        return a[:,2]-(a[:,:2]@self.plane[:2]+self.plane[2])

    def cell_indices(self,points):
        q=np.floor((np.asarray(points)[:,:2]-self.origin)/self.res).astype(int)
        valid=(q[:,0]>=0)&(q[:,0]<self.width)&(q[:,1]>=0)&(q[:,1]<self.height)
        return q[:,1]*self.width+q[:,0],valid

    def cells(self,points):
        ids,valid=self.cell_indices(points)
        return np.unique(ids[valid])

    def obstacle_mask(self,height):
        # Share the positive-height slice between saved and unclassified scans.
        h=np.asarray(height)
        return (h>.06)&(h<self.top)

    def classify(self,points):
        h=self.heights(points)
        return np.abs(h)<=.045,self.obstacle_mask(h)


class LiveLayer:
    def __init__(self,static):
        self.static=static;self.hits={};self.removed_previous_cells=0

    def update(self,points,sensor=None,preclassified=False):
        """Replace live occupancy with this valid scan, including an empty scan.

        The runtime already excludes ground and self returns for preclassified
        input. Saved obstacles and unknown cells retain their original values.
        Sensor loss is handled by the map/runtime freshness guard, not by
        treating missing messages as clear space.
        """
        points=np.asarray(points,float).reshape(-1,3)
        if not np.isfinite(points).all():raise ValueError('实时障碍点无效')
        if preclassified:ground=np.empty((0,3));obstacles=points
        else:
            g,o=self.static.classify(points);ground=points[g];obstacles=points[o]
        ids,valid=self.static.cell_indices(obstacles);seen={}
        for cell,p in zip(ids[valid],obstacles[valid]):
            if cell not in seen or p[2]>seen[cell][2]:seen[int(cell)]=p.copy()
        if len(seen)>50000:raise ValueError('当前扫描障碍层超限')
        self.removed_previous_cells=len(set(self.hits)-set(seen));self.hits=seen
        return ground,obstacles

    def combined(self):
        out=self.static.grid.copy()
        if self.hits:out.ravel()[list(self.hits)]=100
        return out


def inflated(grid,resolution):
    out=grid.copy()
    if np.any(grid==100):
        distance=distance_transform_edt(grid!=100)*resolution
        # Whole chassis footprint is checked separately; retain original 2-cell halo.
        out[(grid==0)&(distance<=2*resolution+1e-6)]=50
        out[(grid==0)&(distance<=resolution+1e-6)]=80
    return out
