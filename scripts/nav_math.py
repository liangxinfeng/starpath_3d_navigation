"""Frame conversions and trajectory validation; ROS-independent."""
import math
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import BSpline

def inverse(t):
    out=np.eye(4);out[:3,:3]=t[:3,:3].T;out[:3,3]=-out[:3,:3]@t[:3,3];return out
def transform(points,t):return np.asarray(points)@t[:3,:3].T+t[:3,3]
def matrix(p):
    q=p.orientation;t=np.eye(4);t[:3,:3]=Rotation.from_quat([q.x,q.y,q.z,q.w]).as_matrix()
    t[:3,3]=[p.position.x,p.position.y,p.position.z]
    if not np.isfinite(t).all():raise ValueError('invalid pose')
    return t
def fill_pose(p,t):
    p.position.x,p.position.y,p.position.z=map(float,t[:3,3])
    p.orientation.x,p.orientation.y,p.orientation.z,p.orientation.w=map(float,Rotation.from_matrix(t[:3,:3]).as_quat())
def seconds(header):return header.stamp.sec+header.stamp.nanosec*1e-9
def yaw(t):return math.atan2(t[1,0],t[0,0])
def wrap(a):return math.atan2(math.sin(a),math.cos(a))
def planar_velocity(previous_local,current_local,world_rotation,dt):
    """Physical motion comes from continuous odometry, never map correction."""
    if not .01<dt<.5:return np.zeros(3)
    v=np.asarray(world_rotation)@((current_local[:3,3]-previous_local[:3,3])/dt)
    v[2]=0.
    return v if np.isfinite(v).all() and np.linalg.norm(v)<1. else np.zeros(3)
def extrinsics(xyz,rpy,calibration):
    base_lidar=np.eye(4);base_lidar[:3,:3]=Rotation.from_euler('xyz',rpy).as_matrix();base_lidar[:3,3]=xyz
    imu_lidar=np.eye(4);imu_lidar[:3,:3]=np.array(calibration['extrinsic_R']).reshape(3,3);imu_lidar[:3,3]=calibration['extrinsic_T']
    return base_lidar@inverse(imu_lidar),imu_lidar

class Trajectory:
    def __init__(self,points,knots,order):
        p=np.asarray(points,float);k=np.asarray(knots,float)
        if p.ndim!=2 or p.shape[1]!=3 or not 4<=len(p)<=10000 or order!=3:raise ValueError('invalid spline control points')
        if len(k)!=len(p)+order+1 or not np.isfinite(k).all() or not np.isfinite(p).all() or np.any(np.diff(k)<0):raise ValueError('invalid spline knots')
        self.start=float(k[order]);self.duration=float(k[-order-1]-self.start)
        if not .01<self.duration<600:raise ValueError('invalid spline duration')
        self.curve=BSpline(k,p,order,extrapolate=False);self.derivative=self.curve.derivative()
    def at(self,t):return self.curve(self.start+np.clip(t,0,self.duration))
    def velocity(self,t):return self.derivative(self.start+np.clip(t,0,self.duration))
    def samples(self):return self.at(np.linspace(0,self.duration,min(800,max(2,int(self.duration/.1)+1))))

def clearance(points,body,radius=.36,offset=.12,top=1.85):
    """Distance to conservative two-cylinder footprint, ignoring floor and overhead."""
    local=transform(points,inverse(body))
    local=local[(local[:,2]>.08)&(local[:,2]<top)]
    if not len(local):return 10.
    d=np.minimum(np.linalg.norm(local[:,:2]-[offset,0],axis=1),np.linalg.norm(local[:,:2]+[offset,0],axis=1))-radius
    return float(np.min(d))
