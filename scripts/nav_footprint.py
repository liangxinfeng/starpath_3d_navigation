"""Planar chassis collision outline derived from the installed S3 URDF.

The hull encloses collision meshes rigidly attached to Link_Zero_Point;
articulated body links remain the responsibility of the 3-D body checker.
No metric padding or simplified rectangular chassis is added.
"""
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial import ConvexHull
from nav_body_model import origin, triangles
from nav_math import transform

DEFAULT_URDF = '/home/ubuntu/miniconda3/envs/robot_env/lib/python3.12/site-packages/autolife_robot_sdk/descriptions/autolife_s1/urdfs/robot_v3_1.urdf'

@lru_cache(maxsize=4)
def chassis_footprint(urdf=DEFAULT_URDF):
    path=Path(urdf);root=ET.parse(path).getroot()
    poses={'Link_Zero_Point':np.eye(4)}
    for _ in range(len(root.findall('joint'))+1):
        before=len(poses)
        for j in root.findall('joint'):
            parent=j.find('parent').get('link');child=j.find('child').get('link')
            if j.get('type')=='fixed' and parent in poses:
                poses[child]=poses[parent]@origin(j.find('origin'))
        if len(poses)==before:break
    points=[]
    for link in root.findall('link'):
        if link.get('name') not in poses:continue
        for c in link.findall('collision'):
            mesh=c.find('geometry/mesh')
            if mesh is None:raise ValueError('底盘碰撞几何非网格，需显式支持后才能规划')
            filename=mesh.get('filename')
            if filename.startswith('package://'):raise ValueError('底盘网格路径尚未解析')
            v=triangles(path.parent/filename).reshape(-1,3)
            v=v*np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
            points.append(transform(v,poses[link.get('name')]@origin(c.find('origin'))))
    if not points:raise ValueError('URDF缺少底盘碰撞几何')
    xy=np.concatenate(points)[:,:2]
    if not np.isfinite(xy).all():raise ValueError('URDF底盘碰撞坐标无效')
    hull=xy[ConvexHull(xy).vertices].copy();hull.setflags(write=False)
    return hull

def intersecting_cells(centers, polygon, resolution):
    """Exact convex polygon / square overlap via separating axes.

    The half cell size represents the occupied cell itself, not footprint
    padding. Interior cells and rotated edge intersections are both tested.
    Coordinates are expressed in the occupancy grid's axis-aligned frame.
    """
    edges=np.roll(polygon,-1,axis=0)-polygon
    axes=np.vstack((np.eye(2),np.column_stack((-edges[:,1],edges[:,0]))))
    axes=axes[np.linalg.norm(axes,axis=1)>1e-12]
    axes/=np.linalg.norm(axes,axis=1)[:,None]
    projected=polygon@axes.T;middle=centers@axes.T
    extent=.5*resolution*np.abs(axes).sum(axis=1)
    return ((middle+extent>=projected.min(axis=0)-1e-10)&
            (middle-extent<=projected.max(axis=0)+1e-10)).all(axis=1)
