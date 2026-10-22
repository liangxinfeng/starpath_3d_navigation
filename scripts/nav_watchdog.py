#!/usr/bin/python3
"""Separate watchdog, so adapter/UI stalls cannot refresh chassis commands."""
import os,sys,json,time,fcntl,math
from pathlib import Path
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_prefix
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from std_msgs.msg import String
sys.path.insert(0,str(Path(get_package_prefix('starmap_3d_mapping'))/'lib/starmap_3d_mapping'))
from nav_drive import MotionGate
from nav_native_limits import native_limits,chassis_yaw_scale,chassis_command
from control_center_chassis_guard.chassis_control_lease import ChassisControlLease

class NoHardware:
    held=False;last_error='当前会话仅允许规划预览'
    def acquire(self,**kwargs):return False
    def release(self):pass
    def shutdown(self):pass

class Watchdog(Node):
    def __init__(self):
        super().__init__('scan345_watchdog')
        self.declare_parameter('session','');self.declare_parameter('allow_hardware',False)
        session=self.get_parameter('session').value
        if not session:raise ValueError('session required')
        self.lock=(Path('/tmp')/('scan345_watchdog_'+os.environ.get('ROS_DOMAIN_ID','0')+'.lock')).open('w')
        fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.yaw_scale=chassis_yaw_scale()
        hardware=self.get_parameter('allow_hardware').value
        if hardware and os.environ.get('ROS_DOMAIN_ID','0')!='0':raise ValueError('Hardware output requires robot domain 0')
        self.pub=self.create_publisher(Twist,'/topic_gv_target_cmd_vel_0_345',1) if hardware else None
        lease=ChassisControlLease(domain_id='0',robot_id='345',logger=self.get_logger()) if hardware else NoHardware()
        limits=native_limits()
        self.gate=MotionGate(lease,self.publish,session,time.monotonic,math.hypot(max(limits['vx_max'],-limits['vx_min']),limits['vy_max']),limits['wz_max'])
        self.status=self.create_publisher(String,'drive_status',1)
        self.create_subscription(String,'drive_intent',self.intent,1)
        self.create_subscription(Odometry,'body_pose',self.odom,1)
        self.create_subscription(Odometry,'nav2_odom',self.local_odom,1)
        self.create_timer(.02,self.gate.tick);self.create_timer(.1,self.report)
    def publish(self,v):
        if self.pub and self.context.ok():
            m=Twist();m.linear.x,m.linear.y,m.angular.z=chassis_command(v,self.yaw_scale);self.pub.publish(m)
    def intent(self,m):
        try:
            d=json.loads(m.data)
            if isinstance(d,dict):self.gate.receive(d)
        except (ValueError,TypeError):pass
    def odom(self,m):
        stamp=m.header.stamp.sec+m.header.stamp.nanosec*1e-9
        if m.header.frame_id=='airy_global' and -.05<=time.time()-stamp<.5:self.gate.last_odom=time.monotonic()
    def local_odom(self,m):
        stamp=m.header.stamp.sec+m.header.stamp.nanosec*1e-9
        if m.header.frame_id=='scan345_odom' and m.child_frame_id=='scan345_base' and -.05<=time.time()-stamp<.5:self.gate.last_odom=time.monotonic()
    def report(self):
        g=self.gate;now=time.monotonic()
        self.status.publish(String(data=json.dumps(dict(session=g.session,seq=g.seq,stamp=now,blocked=g.blocked,reason=g.reason,held=g.lease.held,yaw_command_scale=self.yaw_scale,physical_velocity=list(g.current),ready=not g.blocked and not g.enabled and now-g.last_odom<.5 and now-g.last_input<.3))))
    def close(self):self.gate.close();self.destroy_node()
def main():
    rclpy.init();n=None
    try:n=Watchdog();rclpy.spin(n)
    except (KeyboardInterrupt,rclpy.executors.ExternalShutdownException):pass
    finally:
        if n:n.close()
        rclpy.try_shutdown()
if __name__=='__main__':main()
