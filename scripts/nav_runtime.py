#!/usr/bin/python3
"""AIRY -> synchronized SCAN observations, guarded goal/trajectory execution."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,time,math,fcntl,hashlib,sys
from collections import deque
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from ament_index_python.packages import get_package_share_directory,get_package_prefix
sys.path.insert(0,str(Path(get_package_prefix("starmap_3d_mapping"))/"lib/starmap_3d_mapping"))
from airy_fixed_sector import FixedSectorFilter
from nav_msgs.msg import Odometry,Path as RosPath
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String,Bool,Header,Float64MultiArray
from starpath_3d_navigation.msg import ChassisPath
from starpath_3d_navigation.srv import NavCommand
from nav_ground import separate_ground,low_return_outliers
from nav_speckle import TransientSpeckleFilter
from nav_math import *
from nav_safety import Safety
from nav_body_model import BodyModel,swept_poses,primitive_distances
from nav_chassis import ChassisTrajectory,advance,TrackingFrame,at_global_goal
from nav_hard_self import HardSelfFilter
from nav_link_self import LinkSelfFilter
from nav_envelope import EnvelopeHandoff

class Runtime(Node):
    def __init__(self):
        super().__init__('scan345_runtime')
        for k,v in dict(session='',allow_hardware=False,max_speed=.15,max_yaw_rate=.3,body_radius=.36,body_offset=.12,body_top=1.85,ground_clearance=.08,base_lidar_xyz=[.2634,0.,.2],base_lidar_rpy=[0.,.436332,0.],body_model_enabled=True,dynamic_geometry=False,fixed_sector_filter=True,
            body_urdf='/home/ubuntu/miniconda3/envs/robot_env/lib/python3.12/site-packages/autolife_robot_sdk/descriptions/autolife_s1/urdfs/robot_v3_1.urdf',
            self_profile=str(Path.home()/'.config/starpath_3d_navigation/empty_gripper_profile.npz'),
            self_hard_profile=str(Path.home()/'.config/starpath_3d_navigation/hard_profile.npz'),self_extra_profile='').items():self.declare_parameter(k,v)
        self.session=self.get_parameter('session').value
        if not self.session:raise ValueError('session required')
        self.lock=(Path('/tmp')/('scan345_runtime_'+os.environ.get('ROS_DOMAIN_ID','0')+'.lock')).open('w')
        fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        calpath=Path(get_package_share_directory('starmap_3d_mapping'))/'config/factory_calibration_345.json'
        calib=json.loads(calpath.read_text())
        self.base_imu,self.imu_lidar=extrinsics(self.value('base_lidar_xyz'),self.value('base_lidar_rpy'),calib)
        self.imu_base=inverse(self.base_imu)
        self.model=None;self.residual=None;self.envelope=[];self.envelope_ack=[];self.envelope_time=0.;self.profile_revision=0;self.model_reason='等待关节模型'
        self.sector=FixedSectorFilter(calpath.parent/'fixed_sector_345.json') if self.value('fixed_sector_filter') else None
        self.dynamic_geometry=bool(self.value('dynamic_geometry')) or self.sector is not None;self.handoff=EnvelopeHandoff(time.monotonic,padding=0. if self.sector else .01)
        self.sensor_filter={};self.sensor_filter_time=-1e9
        if self.value('body_model_enabled'):
            if self.value('base_lidar_xyz')!=[.2634,0.,.2] or self.value('base_lidar_rpy')!=[0.,.436332,0.]:raise ValueError('自体模板的底盘雷达外参不匹配')
            self.model=BodyModel(self.value('body_urdf'),nominal_geometry=self.sector is not None,estimated_grippers=False)
            if self.sector:
                # Joints are used only for collision shape, never for filtering.
                self.model.pose_guard=lambda model:(True,'关节反馈有效；固定扇形不随关节改变')
            else:
                self.model.load_profile(self.value('self_profile'),hashlib.sha256(calpath.read_bytes()).hexdigest())
                self.residual=(LinkSelfFilter if self.dynamic_geometry else HardSelfFilter)(self.value('self_hard_profile'),self.model,
                    hashlib.sha256(calpath.read_bytes()).hexdigest(),self.value('self_profile'),**(dict(extra_path=self.value('self_extra_profile')) if self.dynamic_geometry else {}))
                self.model.pose_guard=self.residual.health
            self.model.dynamic_geometry=self.dynamic_geometry
        elif self.value('allow_hardware') or os.environ.get('ROS_DOMAIN_ID','0')=='0':
            raise ValueError('真机域不允许关闭整机自体模型')
        self.safety=Safety(time.monotonic);self.odoms={};self.clouds={};self.global_poses={}
        self.correction=None;self.body=None;self.local_body=None;self.last_body=None;self.points=np.empty((0,3));self.global_stamp=0.;self.backend_time=0.;self.backend={}
        self.tracking_frame=None;self.tracking_info={};self.goal_refinements=0
        self.motion_history=deque(maxlen=30);self.waiting_for_stop=False;self.replan_stop_deadline=0.;self.obstacle_replans=0
        self.goal=None;self.pending_goal=0.;self.command_id=0;self.trajectory=None;self.traj_time=0.;self.progress=0.;self.ui_seq=-1;self.drive_seq=0;self.plan_retries=0;self.retry_scan_index=-1
        self.drive_status={};self.drive_status_time=0.;self.last_tick=time.monotonic();self.state='等待三维定位和实时扫描';self.last_pair=0.;self.scans=0;self.last_collision=None
        self.body_pub=self.create_publisher(Odometry,'body_pose',1)
        self.sensor_pub=self.create_publisher(Odometry,'sensor_pose',1)
        self.cloud_pub=self.create_publisher(PointCloud2,'cloud',1)
        self.clean_pub=self.create_publisher(PointCloud2,'points_cropped',1)
        self.live_pub=self.create_publisher(PointCloud2,'live_scan',1)
        self.ground_pub=self.create_publisher(PointCloud2,'removed_ground_points',1)
        self.chassis_cloud_pub=self.create_publisher(PointCloud2,'chassis_obstacle_points',1)
        self.ground_filter_status={}
        self.speckle_filter=TransientSpeckleFilter()
        self.status_pub=self.create_publisher(String,'status',1)
        self.goal_pub=self.create_publisher(PoseStamped,'move_base_simple/goal',1)
        self.path_pub=self.create_publisher(RosPath,'trajectory',1)
        self.ready_pub=self.create_publisher(Bool,'input_ready',1)
        self.reset_pub=self.create_publisher(Bool,'reset',1)
        self.freeze_pub=self.create_publisher(Bool,'planning/go2_execution_frozen',1)
        self.drive_pub=self.create_publisher(String,'drive_intent',1)
        self.envelope_pub=self.create_publisher(Float64MultiArray,'body_envelope',1)
        self.self_pub=self.create_publisher(PointCloud2,'self_returns',1)
        self.create_subscription(Float64MultiArray,'body_envelope_applied',self.envelope_applied,1)
        if self.model:self.create_subscription(String,'/topic_arm_whole_body_and_gripper_current_joints_status_0_345',self.joints,qos_profile_sensor_data)
        self.create_subscription(Odometry,'/airy_fast_lio/Odometry',self.odom,qos_profile_sensor_data)
        self.create_subscription(PointCloud2,'/airy_fast_lio/cloud_registered_body',self.cloud,qos_profile_sensor_data)
        self.create_subscription(PoseStamped,'/airy_fast_lio/global_pose',self.global_pose,1)
        self.create_subscription(String,'/airy_fast_lio/backend_status',self.backend_status,1)
        self.create_subscription(String,'/airy_fast_lio/sensor_status',self.sensor_filter_status,1)
        self.create_subscription(String,'ui_heartbeat',self.heartbeat,1)
        self.create_subscription(String,'drive_status',self.watchdog_status,1)
        self.create_subscription(ChassisPath,'planning/chassis_path',self.chassis_path,1)
        self.create_subscription(String,'planning_error',self.planning_error,1)
        self.create_subscription(Bool,'planner_alive',lambda _:setattr(self.safety,'planner',time.monotonic()),1)
        self.create_service(NavCommand,'command',self.command)
        self.create_timer(.02,self.tick);self.create_timer(.2,self.report)
    def value(self,k):return self.get_parameter(k).value
    def sensor_filter_status(self,m):
        try:self.sensor_filter=json.loads(m.data);self.sensor_filter_time=time.monotonic()
        except (ValueError,TypeError):self.sensor_filter={};self.sensor_filter_time=-1e9
    def joints(self,m,info):
        try:
            self.model.feedback(json.loads(m.data),source_time=float(info['source_timestamp'])/1e9)
        except (ValueError,TypeError,KeyError):self.model.error='关节反馈格式错误';self.model.last=-1e9
    def update_dynamic_envelope(self):
        if self.sector:
            self.model.update();parts=self.model.primitives
        else:parts=self.residual.collision_parts()
        self.handoff.update(parts)
        self.envelope=self.handoff.sent.copy();self.model.protected_parts=self.handoff.guard()
    def envelope_applied(self,m):
        if self.dynamic_geometry:
            if self.handoff.acknowledge(m.data):self.model.protected_parts=self.handoff.guard()
            return
        if list(m.data)==self.envelope and self.envelope:self.envelope_ack=list(m.data);self.envelope_time=time.monotonic()
    def geometry_health(self):
        if self.model:
            ok,why=self.model.health()
            if not ok:return False,why
            if not self.sector:
                ok,why=self.model.profile_health()
                if not ok:return False,why
            if self.residual:
                ok,why=self.residual.health(self.model)
                if not ok:return False,why
            if self.dynamic_geometry:
                s=self.sensor_filter
                valid=(s.get('filter_mode')=='fixed_sector_volume' and s.get('sector_hash')==self.sector.digest) if self.sector else (s.get('filter_mode')=='dynamic_link_volumes' and s.get('self_filter_source_time')=='dds_source' and s.get('self_filter_model_hash')==self.model.model_hash)
                if time.monotonic()-self.sensor_filter_time>1.5 or s.get('last_valid_scan_age',float('inf'))>.5 or not valid:
                    return False,'等待匹配参数的固定扇形原始点云过滤' if self.sector else '等待匹配模型的动态原始点云过滤'
                ok,why=self.handoff.health()
                if not ok:return False,why
            elif not self.envelope or self.envelope_ack!=self.envelope or time.monotonic()-self.envelope_time>.8:return False,'等待规划器应用整机碰撞外形'
        return True,'实时碰撞模型正常'
    def health(self):
        ok,why=self.geometry_health()
        if not ok:return ok,why
        ok,why=self.safety.health()
        if not ok:return ok,why
        return True,'定位、实时扫描与前伸姿态正常'
    def gap(self,points,body):
        if self.model:return self.model.clearance(points,body)
        return clearance(points,body,self.value('body_radius'),self.value('body_offset'),self.value('body_top'))
    def current(self,header,limit=.6):return -.05<=time.time()-seconds(header)<limit
    def backend_status(self,m):
        try:d=json.loads(m.data)
        except ValueError:return
        self.backend=d;self.backend_time=time.monotonic()
        self.safety.locked=d.get('mode')=='localization' and bool(d.get('locked'))
    def odom(self,m):
        if m.header.frame_id!='airy_map' or m.child_frame_id!='airy_imu' or not self.current(m.header,.5):return
        self.local_body=matrix(m.pose.pose)@self.imu_base
        self.motion_history.append((time.monotonic(),self.local_body.copy()))
        key=round(seconds(m.header)*1e6);self.odoms[key]=m
        while len(self.odoms)>40:self.odoms.pop(next(iter(self.odoms)))
        self.safety.odom=time.monotonic()-(time.time()-seconds(m.header))
        if key in self.global_poses:self.update_correction(key,self.global_poses.pop(key))
        if self.correction is not None and self.safety.locked and time.monotonic()-self.safety.local<1.2:
            local_body=matrix(m.pose.pose)@self.imu_base
            body=self.correction@local_body
            out=Odometry();out.header=m.header;out.header.frame_id='airy_global';out.child_frame_id='scan345_base'
            fill_pose(out.pose.pose,body)
            if self.last_body:
                stamp,old=self.last_body;dt=seconds(m.header)-stamp
                v=planar_velocity(old,local_body,self.correction[:3,:3],dt)
                # SCAN expects world-frame velocity; a localization correction
                # is not chassis motion, and a flat-floor base has no Z speed.
                out.twist.twist.linear.x,out.twist.twist.linear.y,out.twist.twist.linear.z=map(float,v)
            self.last_body=(seconds(m.header),local_body.copy());self.body=body;self.body_pub.publish(out)
        self.pair(key)
    def global_pose(self,m):
        if m.header.frame_id!='airy_global' or not self.current(m.header,1.2):return
        key=round(seconds(m.header)*1e6)
        if key in self.odoms:self.update_correction(key,m)
        else:
            self.global_poses[key]=m
            while len(self.global_poses)>8:self.global_poses.pop(next(iter(self.global_poses)))
    def update_correction(self,key,m):
        new=matrix(m.pose)@inverse(matrix(self.odoms[key].pose.pose))
        if self.correction is not None:
            delta=inverse(self.correction)@new
            if np.linalg.norm(delta[:3,3])>.25 or abs(wrap(yaw(delta)))>math.radians(10):
                self.abort('定位发生明显校正，已取消目标，请重新规划',True)
        # Cached collision points and body must share the NEW correction.
        # Otherwise an update between scans moves the body into old-frame points.
        if self.correction is not None:
            reframe=new@inverse(self.correction)
            if self.body is not None:self.body=reframe@self.body
            if len(self.points):self.points=transform(self.points,reframe)
        self.correction=new;self.global_stamp=seconds(m.header)
        if self.trajectory is not None and self.tracking_frame is not None:self.publish_trajectory()
        self.safety.local=time.monotonic()-(time.time()-self.global_stamp)
    def cloud(self,m):
        if m.header.frame_id!='airy_imu' or not self.current(m.header):return
        key=round(seconds(m.header)*1e6);self.clouds[key]=m
        while len(self.clouds)>5:self.clouds.pop(next(iter(self.clouds)))
        self.pair(key)
    def pair(self,key):
        if key not in self.clouds or key not in self.odoms or self.correction is None:return
        cloud=self.clouds.pop(key)
        if not self.current(cloud.header) or not self.safety.locked or time.monotonic()-self.safety.local>1.2:return
        if time.monotonic()-self.last_pair<.09:return
        a=point_cloud2.read_points(cloud,field_names=('x','y','z'),skip_nans=True)
        if len(a)<100:return
        points=np.column_stack([a['x'],a['y'],a['z']]);imu=self.correction@matrix(self.odoms[key].pose.pose)
        world=transform(points,imu);body=imu@self.imu_base
        body_points=transform(world,inverse(body))
        self_mask=np.zeros(len(world),dtype=bool)
        if self.model:
            good,why=self.model.health()
            if not good:self.model_reason=why;return
            if not self.sector:
                good,why=self.model.profile_health()
                if not good:self.model_reason=why;return
            try:
                # This mode explicitly requires a calibrated, stationary pose.
                # Receipt-clock interpolation is not hardware synchronisation:
                # a cloud callback can delay joint receipt and create false gaps.
                # Fresh feedback, measured stationarity and residual pose limits
                # are checked separately; moving-arm operation is not enabled.
                if self.sector:
                    self_mask=self.sector.mask(body_points)
                    self.update_dynamic_envelope();self.model.filtered=int(self_mask.sum())
                elif self.dynamic_geometry:
                    self_mask=self.residual.mask_registered(body_points,self.model,seconds(cloud.header))
                    self.update_dynamic_envelope();self.model.filtered=int(self_mask.sum())
                else:
                    protected=self.residual.collision_parts(self.model) if self.residual else None
                    self.model.update();self_mask=self.model.filter_with_profile(body_points,self.value('base_lidar_xyz'))
                    if self.residual:
                        self_mask|=self.residual.mask(body_points,self.model)
                        self.model.protected_parts=protected
                    parts=getattr(self.model,'protected_parts',self.model.primitives).ravel().tolist()
                    if not self.envelope or parts!=self.envelope[1:]:
                        self.profile_revision+=1
                        self.envelope=[float(self.profile_revision)]+parts
                        had_goal=self.goal is not None
                        self.envelope_ack=[];self.abort('整机碰撞外形更新，请重新规划',True)
                        if not had_goal:self.state='部件轮廓已更新，可设置目标'
                self.model_reason=why
            except ValueError as e:self.model_reason=str(e);self.abort(self.model_reason,True);return
        ground_mask=np.zeros(len(world),dtype=bool);low_noise=np.zeros(len(world),dtype=bool);height=np.full(len(world),np.nan)
        hint=getattr(self,'global_map',{})
        if hint.get('floor_plane') is not None:
            available=~self_mask
            g,h,detail=separate_ground(world[available],hint['floor_plane'],body[:3,3])
            ground_mask[available]=g;height[available]=h;self.ground_filter_status=detail
            low_noise[available]=low_return_outliers(world[available],h,world[available][g])
            self.ground_filter_status['rejected_low_points']=int(low_noise.sum())
        else:self.ground_filter_status={'mode':'waiting_map_floor','removed_points':0}
        # Use the same floor decision for full-body collision and global planning.
        mask=(body_points[:,2]>self.value('ground_clearance'))&(body_points[:,2]<(2.6 if self.model else self.value('body_top')+.2))&~self_mask
        mask &= ~ground_mask
        header=Header();header.stamp=cloud.header.stamp;header.frame_id='airy_global'
        sensor=Odometry();sensor.header=header;sensor.child_frame_id='scan345_lidar'
        fill_pose(sensor.pose.pose,imu@self.imu_lidar)
        self.sensor_pub.publish(sensor)
        clean=(~self_mask)&(body_points[:,2]>-1.)&(body_points[:,2]<1.)&(np.linalg.norm(body_points[:,:2],axis=1)<12.)
        clean &= ~ground_mask
        clean &= ~low_noise
        candidates=mask|clean
        candidate_ids=np.flatnonzero(candidates)
        # Remove map correction so a localization adjustment cannot invalidate
        # association; motion of the chassis must not align its own echoes.
        stable=transform(world[candidates],inverse(self.correction))
        supported=self.speckle_filter.filter(stable,seconds(cloud.header))
        rejected=np.zeros(len(world),bool);rejected[candidate_ids[~supported]]=True
        mask &= ~rejected;clean &= ~rejected
        obstacles=world[mask]
        self.ground_filter_status['speckle_filter']=self.speckle_filter.stats
        self.clean_pub.publish(point_cloud2.create_cloud_xyz32(header,world[clean].astype(np.float32)))
        nav_mask=clean&(((height>.045)&(height<float(hint.get('chassis_obstacle_height',.37))))|((height<-.08)&(height>-.35)))
        self.chassis_cloud_pub.publish(point_cloud2.create_cloud_xyz32(header,world[nav_mask].astype(np.float32)))
        self.ground_pub.publish(point_cloud2.create_cloud_xyz32(header,world[ground_mask].astype(np.float32)))
        self.ground_filter_status['planner_points']=int(nav_mask.sum())
        self.ground_filter_status['collision_points']=len(obstacles)
        self.cloud_pub.publish(point_cloud2.create_cloud_xyz32(header,obstacles.astype(np.float32)))
        self.live_pub.publish(point_cloud2.create_cloud_xyz32(header,world.astype(np.float32)))
        self.self_pub.publish(point_cloud2.create_cloud_xyz32(header,world[self_mask].astype(np.float32)))
        self.points=obstacles;self.last_pair=time.monotonic();self.safety.cloud=self.last_pair-(time.time()-seconds(header));self.scans+=1
    def heartbeat(self,m):
        try:
            d=json.loads(m.data);seq=int(d['seq']);age=time.monotonic()-float(d['stamp'])
            if d.get('session')!=self.session or seq<=self.ui_seq or not -.05<=age<.25:return
            self.ui_seq=seq;self.safety.ui=time.monotonic()-age
            if d.get('stop'):self.abort('界面请求停车')
        except (ValueError,TypeError,KeyError):pass
    def watchdog_status(self,m):
        try:d=json.loads(m.data)
        except ValueError:return
        if d.get('session')!=self.session or not -.05<=time.monotonic()-d.get('stamp',0)<.5:return
        self.drive_status=d;self.drive_status_time=time.monotonic()
        if d.get('blocked') and self.safety.armed:self.abort(d.get('reason','底盘保护停车'))
    def abort(self,why,clear_map=False):
        self.tracking_frame=None
        self.waiting_for_stop=False
        self.safety.stop(why);self.goal=None;self.pending_goal=0;self.trajectory=None;self.command_id=0;self.state=why
        self.reset_pub.publish(Bool(data=clear_map));self.send_drive(0.,0.,False)
        p=RosPath();p.header.frame_id='airy_global';self.path_pub.publish(p)
    def record_collision(self,future,horizon,speed,rate):
        # Diagnostic only, called AFTER stop has been sent. Never changes masks
        # or decisions. Overwrite one bounded snapshot on each collision stop.
        try:
            if self.model is None or self.body is None or not len(self.points):return
            points=transform(self.points,inverse(self.body));local=transform(self.points,inverse(future))
            parts=self.model.protected_parts;gaps=primitive_distances(local,parts)
            ids=np.flatnonzero(gaps<=0.)
            if not len(ids):return
            i=int(ids[np.argmin(gaps[ids])]);part=int(np.argmin([primitive_distances(local[i:i+1],[p])[0] for p in parts]))
            event=dict(count=len(ids),xyz=np.round(points[i],4).tolist(),gap=round(float(gaps[i]),4),part=part,horizon=float(horizon),speed=float(speed),yaw_rate=float(rate),stamp=time.time(),neighbours=int(np.sum(np.linalg.norm(points-points[i],axis=1)<.1)-1))
            self.last_collision=event
            np.savez_compressed('/tmp/scan345_last_collision.npz',points=self.points,body=self.body,future=future,parts=parts,metadata=json.dumps(event),joints=json.dumps(self.model.q),grippers=json.dumps(self.model.gripper_q))
            self.get_logger().warning('COLLISION_EVENT '+json.dumps(event,ensure_ascii=False))
        except Exception as e:self.get_logger().warning('Collision diagnostic unavailable: '+str(e))
    def command(self,req,res):
        try:
            d=json.loads(req.payload or '{}')
            if d.get('session')!=self.session:raise ValueError('界面会话不匹配')
            if req.command=='cancel':self.abort('已取消目标并停车');res.success=True;res.message=self.state;return res
            if req.command!='goal':raise ValueError('未知命令')
            ok,why=self.health()
            if not ok or self.body is None:raise ValueError(why)
            if time.monotonic()-self.safety.ui>.3:raise ValueError('界面心跳未就绪')
            point=np.array([float(d['x']),float(d['y']),self.body[2,3]])
            if not np.isfinite(point).all():raise ValueError('目标坐标必须是有限数值')
            execute=bool(d.get('execute',False));heading=d.get('yaw')
            if heading is not None and not math.isfinite(float(heading)):raise ValueError('无效到点朝向')
            self.abort('正在生成新目标轨迹')
            self.last_collision=None
            if execute:
                if not self.value('allow_hardware'):raise ValueError('当前启动为仅规划模式')
                if time.monotonic()-self.drive_status_time>.5 or not self.drive_status.get('ready'):raise ValueError('底盘看门狗尚未就绪，请等状态正常后执行')
                self.safety.arm()
            self.goal_refinements=0;self.obstacle_replans=0;self.tracking_info={}
            self.goal=dict(point=point,execute=execute,yaw=heading);self.pending_goal=time.monotonic()+.35;self.plan_retries=0;self.retry_scan_index=-1
            self.state='等待规划；真机低速执行已启用' if execute else '等待规划；仅预览'
            res.success=True;res.message=self.state
        except Exception as e:res.success=False;res.message=str(e)
        return res
    def chassis_path(self,m):
        if self.goal is None or m.command_id!=self.command_id or not self.command_id:return
        try:
            if m.header.frame_id!='airy_global' or not self.current(m.header,1.):raise ValueError('底盘轨迹坐标系或时间无效')
            poses=[[p.position.x,p.position.y,2*math.atan2(p.orientation.z,p.orientation.w)] for p in m.poses]
            if not poses:raise ValueError('底盘轨迹为空')
            if any(abs(p.position.z-self.goal['point'][2])>.12 for p in m.poses):raise ValueError('轨迹高度超出平地测试范围')
            traj=ChassisTrajectory(poses,m.speeds,m.yaw_rates,m.durations,m.final_segment,m.poses[0].position.z)
            if self.body is None or np.linalg.norm(traj.poses[0,:2]-self.body[:2,3])>.08 or abs(wrap(traj.poses[0,2]-yaw(self.body)))>.10:
                raise ValueError('规划期间底盘位姿已变化，请重新规划')
            if traj.final and np.linalg.norm(traj.poses[-1,:2]-self.goal['point'][:2])>.01:raise ValueError('轨迹没有到达原目标')
            if traj.final and self.goal['yaw'] is not None and abs(wrap(traj.poses[-1,2]-float(self.goal['yaw'])))>.01:raise ValueError('轨迹终点朝向不符')
            # Search runs on an immutable scan in another process/thread. Check
            # the immediate motion against the newest scan before accepting it;
            # the full future sweep is rechecked continuously during execution.
            for t in np.linspace(0,min(.7,traj.duration),12):
                ref,_,_=traj.reference(t);body=self.body.copy();body[:2,3]=ref[:2]
                body[:3,:3]=Rotation.from_euler('z',wrap(ref[2]-yaw(self.body))).as_matrix()@self.body[:3,:3]
                if self.gap(self.points,body)<=0.:raise ValueError('起步动作被最新点云阻挡')
            self.trajectory=traj;self.tracking_frame=TrackingFrame(self.correction);self.progress=0.;self.traj_time=time.monotonic()
            self.last_collision=None
            self.publish_trajectory()
            self.state='正在按底盘动作低速导航' if self.safety.armed else ('底盘轨迹已生成 · 仅预览' if traj.final else '底盘局部轨迹已生成 · 仅预览，执行时继续规划至原目标')
        except Exception as e:
            if str(e)=='起步动作被最新点云阻挡' and self.retry_scan(str(e)):return
            if str(e)=='起步动作被最新点云阻挡' and self.queue_obstacle_replan(str(e)):return
            self.abort('拒绝无效轨迹：'+str(e))
    def publish_trajectory(self):
        if self.trajectory is None or self.tracking_frame is None:return
        points,angles=self.tracking_frame.display(self.trajectory.samples(),self.trajectory.z,self.correction)
        p=RosPath();p.header.frame_id='airy_global';p.header.stamp=self.get_clock().now().to_msg()
        for point,angle in zip(points,angles):
            v=PoseStamped();v.header=p.header;v.pose.position.x,v.pose.position.y,v.pose.position.z=map(float,point)
            v.pose.orientation.z=math.sin(angle/2);v.pose.orientation.w=math.cos(angle/2);p.poses.append(v)
        self.path_pub.publish(p)
    def next_segment(self,now,reason):
        # Stop, then plan and collision-check again from measured pose to the
        # unchanged original destination. This never resumes an unchecked edge.
        self.trajectory=None;self.tracking_frame=None;self.pending_goal=now+.45;self.command_id=0;self.plan_retries=0;self.retry_scan_index=self.scans
        self.reset_pub.publish(Bool(data=False));self.state=reason
    def settled_for_replan(self,now):
        recent=[(t,b) for t,b in self.motion_history if now-t<.4]
        if not recent or now-recent[-1][0]>.15 or recent[-1][0]-recent[0][0]<.25:return False
        last=recent[-1][1]
        return all(np.linalg.norm(b[:2,3]-last[:2,3])<.01 and abs(wrap(yaw(b)-yaw(last)))<.02 for _,b in recent)
    def queue_obstacle_replan(self,why):
        if self.goal is None or not self.goal.get('execute') or not self.safety.armed:return False
        # Disabled intent issues a zero and releases control before planning.
        # Never carry the previous motion through the planner's search time.
        self.send_drive(0.,0.,False)
        if self.obstacle_replans>=3:
            self.abort('障碍持续阻挡，已尝试3次重新规划；保持停车，请调整目标或等待通道恢复')
            return True
        self.obstacle_replans+=1;now=time.monotonic()
        self.next_segment(now,'发现障碍，已停车；等待新扫描重新规划（%d/3）'%self.obstacle_replans)
        path=RosPath();path.header.frame_id='airy_global';self.path_pub.publish(path)
        self.pending_goal=now+.8;self.waiting_for_stop=True;self.replan_stop_deadline=now+5.
        self.get_logger().info('OBSTACLE_REPLAN '+str(self.obstacle_replans)+' '+why)
        return True
    def record_tracking(self,current):
        ref,_,_=self.trajectory.reference(self.progress)
        self.tracking_info=dict(position_error_m=float(np.linalg.norm(ref[:2]-current[:2])),yaw_error_rad=float(wrap(ref[2]-current[2])),progress_s=float(self.progress),frame='continuous_local_odometry')
    def tracking_fault(self,reason):
        event=dict(self.tracking_info,reason=reason,stamp=time.time())
        self.abort(reason)
        self.get_logger().warning('TRACKING_EVENT '+json.dumps(event,ensure_ascii=False))
        try:Path('/tmp/scan345_last_tracking.json').write_text(json.dumps(event,ensure_ascii=False))
        except OSError as e:self.get_logger().warning('Tracking diagnostic unavailable: '+str(e))
    def retry_scan(self,reason):
        # Before accepting any trajectory, a transient occupancy may be
        # rechecked on a NEW scan. Never discard the return, enlarge a self
        # mask, retry an executing path, or recover from a health/stop fault.
        if self.goal is None or self.trajectory is not None or self.plan_retries>=2 or not self.health()[0]:return False
        self.plan_retries+=1;self.command_id=0;self.pending_goal=time.monotonic()+.40;self.retry_scan_index=self.scans
        self.reset_pub.publish(Bool(data=False));self.send_drive(0.,0.,False)
        self.state=f'保持停车，等待新扫描复核（{self.plan_retries}/2）：{reason}'
        return True
    def send_drive(self,x,yaw_rate,enabled):
        self.drive_seq+=1
        self.drive_pub.publish(String(data=json.dumps(dict(session=self.session,seq=self.drive_seq,stamp=time.monotonic(),enabled=enabled,x=float(x),y=0.,yaw=float(yaw_rate)))))
    def planning_error(self,m):
        try:d=json.loads(m.data)
        except (ValueError,TypeError):return
        if not isinstance(d,dict) or self.goal is None or not self.command_id or d.get('command_id')!=self.command_id:return
        reasons={'reference_path_unavailable':'规划失败：参考路径被阻挡或无法生成，目标已取消',
                 'search_failed':'规划失败：局部搜索未找到可行轨迹，目标已取消',
                 'trajectory_collision':'规划失败：轨迹与局部障碍冲突',
                 'start_occupied':'规划失败：当前URDF部件轮廓与障碍相交',
                 'goal_occupied':'规划失败：目标位置或朝向与部件轮廓存在碰撞',
                 'inputs_stale':'规划失败：定位、点云或部件轮廓未就绪',
                 'search_budget':'规划失败：限定时间内未找到可行底盘动作，请选择更开阔方向',
                 'no_chassis_route':'规划失败：局部范围内没有满足朝向和碰撞约束的底盘路线'}
        reason=reasons.get(d.get('code'))
        if reason:
            if d.get('code') in ('start_occupied','goal_occupied','trajectory_collision','search_budget','no_chassis_route') and self.retry_scan(reason):return
            if d.get('code') in ('start_occupied','goal_occupied','trajectory_collision','search_budget','no_chassis_route') and self.queue_obstacle_replan(reason):return
            self.get_logger().warning(reason);self.abort(reason)
    def tick(self):
        now=time.monotonic();dt=min(.05,max(0.,now-self.last_tick));self.last_tick=now
        if self.dynamic_geometry and self.model.health()[0]:
            try:
                if self.residual:self.residual.prepare(self.model.q,self.model.gripper_q)
                self.update_dynamic_envelope()
            except ValueError as e:self.abort(str(e),True);return
        if now-self.backend_time>1.:self.safety.locked=False
        ok,why=self.health();self.ready_pub.publish(Bool(data=ok))
        was_armed=self.safety.armed;self.safety.check()
        if self.goal is not None and (not ok or was_armed and not self.safety.armed):
            self.abort(self.safety.fault or why,True)
        if getattr(self,'waiting_for_stop',False):
            if self.settled_for_replan(now):self.waiting_for_stop=False
            elif now>self.replan_stop_deadline:self.abort('未确认底盘停稳，保持停车；本次重规划已取消')
        if self.goal and self.pending_goal and not getattr(self,'waiting_for_stop',False) and now>=self.pending_goal and self.scans>self.retry_scan_index:
            p=PoseStamped();p.header.frame_id='airy_global';p.header.stamp=self.get_clock().now().to_msg()
            self.command_id=p.header.stamp.sec*10**9+p.header.stamp.nanosec
            p.pose.position.x,p.pose.position.y,p.pose.position.z=map(float,self.goal['point'])
            if self.goal['yaw'] is not None:
                p.pose.orientation.z=math.sin(float(self.goal['yaw'])/2);p.pose.orientation.w=math.cos(float(self.goal['yaw'])/2)
            else:p.pose.orientation.w=0. # explicitly unspecified final heading
            self.goal_pub.publish(p);self.pending_goal=0.;self.traj_time=now
        if self.goal and not self.pending_goal and self.trajectory is None and now-self.traj_time>8:self.abort('规划超时，目标已取消')
        x=rate=0.;frozen=True
        if self.safety.armed:
            if now-self.drive_status_time>.5:self.abort('底盘看门狗状态中断')
            elif self.trajectory and self.body is not None:
                control_body=self.tracking_frame.anchor@self.local_body
                current=np.array([control_body[0,3],control_body[1,3],yaw(control_body)])
                physical=np.array([self.body[0,3],self.body[1,3],yaw(self.body)])
                try:
                    if now-self.traj_time>max(15.,self.trajectory.duration*2.5+5.):
                        raise ValueError('底盘未能在限定时间内跟随当前段，已停车，请重新规划')
                    self.record_tracking(current)
                    x,rate,self.progress,done=self.trajectory.control(current,self.progress,dt)
                    frozen=False
                    if done:
                        x=rate=0.
                        if self.trajectory.final and at_global_goal(self.body,self.goal):self.abort('已到达目标')
                        elif self.trajectory.final:
                            self.goal_refinements+=1
                            if self.goal_refinements>3:self.abort('定位后的终点位置仍未收敛，已停车，请检查定位')
                            else:self.next_segment(now,'当前段完成，按全局定位重新检查原目标并规划')
                        else:self.next_segment(now,'局部段完成，停车规划下一段至原目标')
                    else:
                        for t in np.linspace(0,.5,9):
                            p=advance(physical,x,rate,float(t));future=self.body.copy();future[:2,3]=p[:2]
                            future[:3,:3]=Rotation.from_euler('z',wrap(p[2]-physical[2])).as_matrix()@self.body[:3,:3]
                            if self.gap(self.points,future)<=(0. if self.model else .03):
                                self.send_drive(0.,0.,False);self.record_collision(future,t,x,rate)
                                if not self.queue_obstacle_replan('下一步动作与最新点云相交'):self.abort('前进或转向经过的部件轮廓附近有障碍，已停车')
                                x=rate=0.;frozen=True;break
                except ValueError as e:self.tracking_fault(str(e));x=rate=0.
        self.freeze_pub.publish(Bool(data=frozen or not self.safety.armed))
        x,rate=Safety.bounded(x,rate,self.value('max_speed'),self.value('max_yaw_rate'))
        self.send_drive(x,rate,self.safety.armed and self.trajectory is not None)
    def report(self):
        if self.envelope:self.envelope_pub.publish(Float64MultiArray(data=self.envelope))
        ok,why=self.health()
        gap=self.gap(self.points,self.body) if self.body is not None else None
        d=dict(obstacle_replans=self.obstacle_replans,waiting_for_stop=self.waiting_for_stop,tracking=self.tracking_info,session=self.session,ready=ok,reason=why,state=self.state,planner='底盘 SE(2) 动作搜索',armed=self.safety.armed,hardware=self.value('allow_hardware'),scans=self.scans,clearance=gap,cloud_age=round(max(0,time.monotonic()-self.safety.cloud),2),global_age=round(max(0,time.monotonic()-self.safety.local),2),watchdog=self.drive_status.get('reason','等待'),goal=self.goal is not None,trajectory=self.trajectory is not None,body=self.body.tolist() if self.body is not None else None,map=self.backend.get('map',''))
        if self.model:d.update(collision_padding_m=0.,collision_geometry='实时URDF网格紧包围盒；夹爪估算球已取消',collision_part_names=self.model.primitive_names,self_filtered=self.model.filtered,body_radius=self.model.radius,body_parts=getattr(self.model,'protected_parts',self.model.primitives).tolist(),posture='固定扇形过滤' if self.sector else self.model.profile_health()[1],joints_ok=self.model.health()[0])
        if self.model and self.body is not None and len(self.points):
            p=transform(self.points,inverse(self.body))
            p=p[np.linalg.norm(p[:,:2],axis=1)<1.4]
            if len(p):
                gaps=primitive_distances(p,self.model.protected_parts)
                hit=np.flatnonzero(gaps<=0.)
                if len(hit):
                    i=int(hit[np.argmin(gaps[hit])])
                    d['blocker']=dict(count=len(hit),xyz=np.round(p[i],3).tolist(),gap=round(float(gaps[i]),3))
        if self.residual:d.update(hard_filtered=self.residual.removed,hard_mesh=self.residual.mesh_removed,hard_cells=self.residual.cells_removed,filter_mode='dynamic_link_model' if self.dynamic_geometry else 'hard_model_cells',filter_status=self.residual.health(self.model)[1])
        if self.sector:d.update(hard_filtered=int(self.sector.last_mask.sum())+int(self.sensor_filter.get('masked',0)),filter_mode='fixed_sector_volume',filter_status='固定扇形过滤 · 不随关节变化',sector_config=self.sector.config,sector_hash=self.sector.digest)
        if self.last_collision:d['last_collision']=self.last_collision
        self.status_pub.publish(String(data=json.dumps(d,ensure_ascii=False,allow_nan=False)))
    def close(self):
        if self.context.ok():self.abort('导航已关闭')
        self.destroy_node()
def main():
    rclpy.init();n=None
    try:n=Runtime();rclpy.spin(n)
    except (KeyboardInterrupt,rclpy.executors.ExternalShutdownException):pass
    finally:
        if n:n.close()
        rclpy.try_shutdown()
if __name__=='__main__':main()
