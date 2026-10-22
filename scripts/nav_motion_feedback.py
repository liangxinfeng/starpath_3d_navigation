"""Align delayed map observations with continuous original chassis odometry."""
import math
from collections import deque
import numpy as np
from nav_math import inverse,yaw,wrap,Rotation
from nav_nav2_math import planar

class MotionFeedback:
    def __init__(self):
        self.history=deque(maxlen=400);self.correction=None;self.reference=None;self.aligned_stamp=-1e9
    def push(self,stamp,pose,velocity):
        if not math.isfinite(stamp) or not np.isfinite(pose).all() or not all(math.isfinite(v) for v in velocity):raise ValueError('底盘里程计含无效数值')
        if math.hypot(*velocity[:2])>2. or abs(velocity[2])>4.:raise ValueError('底盘速度反馈异常')
        if self.history and stamp<=self.history[-1][0]:
            if stamp<self.history[-1][0]-.05:raise ValueError('底盘里程计时间倒退，请重新连接')
            return False
        self.history.append((float(stamp),planar(pose),tuple(velocity)));return True
    def at(self,stamp):
        if not self.history or stamp<self.history[0][0] or stamp>self.history[-1][0]:return None
        for t,p,v in reversed(self.history):
            if t<=stamp:
                if abs(t-stamp)<1e-7:return p.copy()
                t1,p1,_=next(x for x in self.history if x[0]>stamp)
                if t1-t>.1+1e-9:return None
                ratio=(stamp-t)/(t1-t);out=p.copy();out[:2,3]=p[:2,3]+ratio*(p1[:2,3]-p[:2,3])
                out[:3,:3]=Rotation.from_euler('z',yaw(p)+ratio*wrap(yaw(p1)-yaw(p))).as_matrix();return out
        return None
    def align(self,stamp,map_body):
        if stamp<self.aligned_stamp:return False
        local=self.at(stamp)
        if local is None:return False
        self.correction=planar(map_body)@inverse(local);self.reference=map_body.copy();self.aligned_stamp=stamp;return True
    def body(self):
        if self.correction is None or not self.history:return None
        current=self.correction@self.history[-1][1];out=self.reference.copy();out[:2,3]=current[:2,3]
        out[:3,:3]=Rotation.from_euler('z',wrap(yaw(current)-yaw(out))).as_matrix()@out[:3,:3];return out
