"""Validated differential-drive primitives and a pose-feedback follower."""
import math
import numpy as np
from nav_math import wrap,inverse,yaw,transform

def advance(p,v,w,t):
    x,y,a=map(float,p)
    if abs(w)<1e-10:x+=v*t*math.cos(a);y+=v*t*math.sin(a)
    else:
        x+=v/w*(math.sin(a+w*t)-math.sin(a))
        y-=v/w*(math.cos(a+w*t)-math.cos(a))
    return np.array([x,y,wrap(a+w*t)])

class ChassisTrajectory:
    def __init__(self,poses,speeds,rates,durations,final,z):
        self.poses=np.asarray(poses,dtype=float)
        self.v=np.asarray(speeds,dtype=float);self.w=np.asarray(rates,dtype=float);self.dt=np.asarray(durations,dtype=float)
        n=len(self.v)
        if not 0<=n<=1000 or self.poses.shape!=(n+1,3) or len(self.w)!=n or len(self.dt)!=n:raise ValueError('底盘轨迹数组不一致')
        if not all(np.isfinite(a).all() for a in (self.poses,self.v,self.w,self.dt)):raise ValueError('轨迹包含非有限值')
        if np.any(self.v<0) or np.any(self.v>.15+1e-8) or np.any(np.abs(self.w)>.30+1e-8) or np.any(self.dt<=0) or np.sum(self.dt)>600:
            raise ValueError('底盘动作超出速度或时间范围')
        for i in range(n):
            expected=advance(self.poses[i],self.v[i],self.w[i],self.dt[i])
            if np.linalg.norm(expected[:2]-self.poses[i+1,:2])>1e-5 or abs(wrap(expected[2]-self.poses[i+1,2]))>1e-5:
                raise ValueError('动作与位姿不连续')
        self.ends=np.cumsum(self.dt);self.duration=float(self.ends[-1]) if n else 0.
        self.final=bool(final);self.z=float(z)
    def reference(self,t):
        if t>=self.duration:return self.poses[-1],0.,0.
        i=int(np.searchsorted(self.ends,max(0.,t),side='right'))
        start=0. if i==0 else self.ends[i-1]
        return advance(self.poses[i],self.v[i],self.w[i],max(0.,t-start)),float(self.v[i]),float(self.w[i])
    def samples(self):
        result=[self.poses[0]]
        for i in range(len(self.v)):
            count=max(1,int(math.ceil(max(self.v[i]*self.dt[i]/.03,abs(self.w[i]*self.dt[i])/.04))))
            result.extend(advance(self.poses[i],self.v[i],self.w[i],t) for t in np.linspace(0,self.dt[i],count+1)[1:])
        return np.asarray(result)
    def control(self,p,t,dt):
        ref,v,w=self.reference(t);a=float(p[2]);delta=ref[:2]-np.asarray(p[:2])
        ex=math.cos(a)*delta[0]+math.sin(a)*delta[1]
        ey=-math.sin(a)*delta[0]+math.cos(a)*delta[1]
        ea=wrap(ref[2]-a)
        if np.linalg.norm(delta)>.15 or abs(ea)>.40:raise ValueError('底盘实际跟踪偏差超限：%.1f厘米 / %.1f度，已停车'%(np.linalg.norm(delta)*100,math.degrees(abs(ea))))
        # Preserve arc feedforward; do not replace it with rotate-to-lookahead.
        x=max(0.,min(.15,v*math.cos(ea)+1.0*ex))
        turning=v==0 and t<self.duration
        rate=max(-.30,min(.30,w+1.5*ea+(0. if turning else 2.0*ey)))
        if v==0 and t<self.duration:x=0. # an in-place turn stays in place
        # Slow the reference if the chassis lags, instead of letting the
        # virtual robot advance independently of measured motion.
        # An in-place turn cannot correct XY drift; never freeze its clock on
        # longitudinal position error, nor steer it with lateral error.
        # Slow only when lagging the signed turn, not when ahead of it.
        angular_lag=max(0.,ea*math.copysign(1.,w)) if abs(w)>1e-5 else abs(ea)
        scale=max(0.,min(1.,1.-(0. if turning else max(0.,ex-.02)*15.)-max(0.,angular_lag-.06)/.14))
        t=min(self.duration,t+dt*scale)
        end=self.poses[-1]
        done=t>=self.duration and np.linalg.norm(end[:2]-p[:2])<.045 and abs(wrap(end[2]-p[2]))<.06
        return x,rate,t,done


class TrackingFrame:
    """Keep one segment in continuous odometry despite map corrections.

    The original global destination is NOT moved. Only the already accepted
    segment is anchored; completion must be checked against the global goal.
    """
    def __init__(self,correction):
        self.anchor=np.asarray(correction,dtype=float).copy()
    def body(self,current_body,current_correction):
        return self.anchor@inverse(current_correction)@current_body
    def display(self,samples,z,current_correction):
        delta=current_correction@inverse(self.anchor)
        points=transform(np.column_stack((samples[:,:2],np.full(len(samples),z))),delta)
        angles=np.array([wrap(a+yaw(delta)) for a in samples[:,2]])
        return points,angles

def at_global_goal(body,goal):
    position=np.linalg.norm(body[:2,3]-np.asarray(goal['point'])[:2])
    angle=0. if goal.get('yaw') is None else abs(wrap(yaw(body)-float(goal['yaw'])))
    return position<.045 and angle<.06
