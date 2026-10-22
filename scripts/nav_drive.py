"""Fail-closed motion gate shared by the ROS watchdog and isolated tests."""
import math
from control_center_chassis_guard.velocity_ramp import VelocityRamp

def valid_chassis_odom(m,now):
    stamp=m.header.stamp.sec+m.header.stamp.nanosec*1e-9
    v=m.twist.twist;p=m.pose.pose.position
    return (m.header.frame_id=='odom' and m.child_frame_id=='base_link'
            and -.05<=now-stamp<.5
            and all(math.isfinite(x) for x in [p.x,p.y,p.z,v.linear.x,v.linear.y,v.angular.z]))

class MotionGate:
    def __init__(self,lease,publish,session,clock,max_linear,max_angular):
        self.max_linear=float(max_linear);self.max_angular=float(max_angular);self.mode="manual"
        self.lease=lease;self.publish=publish;self.session=session;self.clock=clock
        self.last_input=-1e9;self.last_odom=-1e9;self.seq=-1;self.enabled=False
        self.target=(0.,0.,0.);self.current=(0.,0.,0.);self.moving=False;self.blocked=False;self.reason='键盘控制未启用'
        self.ramp=VelocityRamp(clock=clock)
    def stop(self,reason,block=False):
        # Never send zeros into another controller's ownership interval.
        if self.lease.held:
            try:self.publish((0.,0.,0.))
            finally:self.lease.release()
        self.moving=False;self.current=(0.,0.,0.);self.target=(0.,0.,0.)
        self.ramp.reset()
        self.reason=reason;self.blocked=block;self.enabled=False if block else self.enabled
    def receive(self,d):
        now=self.clock()
        if d.get('session')!=self.session:return
        try:
            stamp=float(d['stamp']);seq=int(d['seq']);v=tuple(float(d[k]) for k in ['x','y','yaw'])
            if not all(math.isfinite(a) for a in (*v,stamp)) or not -.05<=now-stamp<=.2:raise ValueError('stale/nonfinite')
            if seq<=self.seq:return
            self.seq=seq;self.last_input=now
            if not bool(d.get('enabled')):
                self.enabled=False;self.stop('键盘控制未启用');return
            mode=d.get('mode','manual')
            if mode not in ('manual','auto'):raise ValueError('invalid control mode')
            if mode!=self.mode and self.enabled:self.stop('控制模式切换');self.ramp.reset()
            self.mode=mode;self.enabled=True
            if self.blocked:return
            if math.hypot(v[0],v[1])>self.max_linear+.001 or abs(v[2])>self.max_angular+.001:raise ValueError('speed limit')
            self.target=v
            self.ramp.set_target(*v)
            if not any(v):self.reason='松键平滑减速'
        except (ValueError,KeyError,TypeError):self.stop('无效或过期控制指令，已停车',True)
    def tick(self):
        now=self.clock()
        if self.moving and not self.lease.held:self.stop('底盘控制权已丢失',True);return
        if not self.enabled or self.blocked:return
        if now-self.last_input>.30:self.stop('界面心跳中断，已停车',True);return
        if now-self.last_odom>.6:self.stop('底盘里程计中断，已停车',True);return
        if self.mode=='auto':
            if not any(self.target):self.stop('导航已停车');return
            if not self.lease.held and not self.lease.acquire(timeout_sec=0.):
                self.stop(self.lease.last_error or '底盘控制权不可用',True);return
            self.current=self.target;self.publish(self.current);self.moving=True;self.reason='原导航速度控制中';return
        if self.ramp.idle:return
        if not self.lease.held and not self.lease.acquire(timeout_sec=0.):
            self.stop(self.lease.last_error or '底盘控制权不可用',True);return
        self.current=self.ramp.step()
        if self.ramp.idle:self.current=(0.,0.,0.)
        self.publish(self.current);self.moving=not self.ramp.idle
        if self.ramp.idle:
            self.lease.release();self.ramp.reset();self.reason='已平滑停车'
        else:self.reason='键盘控制中' if any(self.target) else '松键平滑减速'
    def close(self):
        try:self.stop('控制已关闭')
        finally:self.lease.shutdown()
