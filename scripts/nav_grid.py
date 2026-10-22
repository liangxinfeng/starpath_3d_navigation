"""Fail-closed occupancy checks, independent of ROS and of the ESDF optimizer."""
import math
import numpy as np
from nav_footprint import chassis_footprint, intersecting_cells


class ObservedGrid:
    def __init__(self, data, width, height, resolution, origin, angle=0., footprint=None):
        if width <= 0 or height <= 0 or not .001 <= resolution <= 1.:
            raise ValueError('无效规划栅格尺寸')
        self.data = np.asarray(data, dtype=np.int16).reshape(height, width)
        if not np.isin(self.data, np.arange(-1, 101)).all():
            raise ValueError('无效规划栅格代价')
        self.width, self.height, self.resolution = width, height, float(resolution)
        self.origin = np.asarray(origin, dtype=float)
        self.angle = float(angle)
        self.footprint = np.array(chassis_footprint() if footprint is None else footprint, dtype=float)
        if self.footprint.ndim != 2 or self.footprint.shape[1] != 2 or len(self.footprint)<3 or not np.isfinite(self.footprint).all():
            raise ValueError('无效URDF底盘轮廓')
        if not np.isfinite([*self.origin, self.angle]).all():
            raise ValueError('无效规划栅格坐标')

    def cells(self, xy):
        c, s = math.cos(self.angle), math.sin(self.angle)
        local = (np.asarray(xy) - self.origin) @ np.array([[c, -s], [s, c]])
        return np.floor(local / self.resolution).astype(np.int64)

    def points_reason(self, xy):
        xy = np.asarray(xy).reshape(-1, 2)
        if not len(xy) or not np.isfinite(xy).all():
            return '坐标无效'
        cells = self.cells(xy)
        if ((cells < 0).any() or (cells[:, 0] >= self.width).any()
                or (cells[:, 1] >= self.height).any()):
            return '超出全局导航地图'
        v = self.data[cells[:, 1], cells[:, 0]]
        if (v < 0).any():
            return '进入未知区域，尚未确认可通行'
        if (v >= 50).any():
            return '进入障碍物或安全膨胀区'
        return ''

    def pose_reason(self, x, y, heading):
        """Check occupied squares intersecting the unpadded URDF footprint."""
        if not np.isfinite([x, y, heading]).all():
            return '底盘位姿无效'
        c, s = math.cos(heading), math.sin(heading)
        rot = np.array([[c, -s], [s, c]])
        corners = self.footprint @ rot.T + [x, y]
        ids = self.cells(corners)
        lo, hi = ids.min(axis=0), ids.max(axis=0)
        if (lo < 0).any() or hi[0] >= self.width or hi[1] >= self.height:
            return '底盘轮廓超出全局导航地图'
        xx, yy = np.meshgrid(np.arange(lo[0], hi[0]+1), np.arange(lo[1], hi[1]+1))
        centers = np.column_stack([xx.ravel()+.5, yy.ravel()+.5]) * self.resolution
        cg, sg = math.cos(self.angle), math.sin(self.angle)
        polygon = (corners-self.origin) @ np.array([[cg, -sg], [sg, cg]])
        keep = intersecting_cells(centers,polygon,self.resolution)
        v = self.data[yy.ravel()[keep], xx.ravel()[keep]]
        if (v < 0).any():
            return '底盘轮廓进入未知区域'
        if (v >= 50).any():
            return '底盘轮廓进入障碍物或安全膨胀区'
        return ''

    def path_reason(self, xy, start_heading, final_heading=None):
        xy = np.asarray(xy, dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) < 2 or not np.isfinite(xy).all():
            return '路径数据无效'
        # Validate interpolation, not just the sparse waypoints. Rotation at
        # corners is conservatively checked as an in-place rotation.
        angle = float(start_heading)
        for a, b in zip(xy[:-1], xy[1:]):
            distance = float(np.linalg.norm(b-a))
            if distance < 1e-8:
                continue
            target = math.atan2(b[1]-a[1], b[0]-a[0])
            turn = math.remainder(target-angle, 2*math.pi)
            for h in np.linspace(angle, angle+turn, max(2, math.ceil(abs(turn)/.04)+1)):
                why = self.pose_reason(*a, h)
                if why:
                    return why
            angle = target
            for t in np.linspace(0., 1., max(2, math.ceil(distance/(self.resolution/3))+1)):
                why = self.pose_reason(*(a+(b-a)*t), angle)
                if why:
                    return why
        if final_heading is not None:
            turn = math.remainder(final_heading-angle, 2*math.pi)
            for h in np.linspace(angle, angle+turn, max(2, math.ceil(abs(turn)/.04)+1)):
                why = self.pose_reason(*xy[-1], h)
                if why:
                    return why
        return ''
