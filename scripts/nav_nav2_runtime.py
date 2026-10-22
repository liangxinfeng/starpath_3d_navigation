#!/usr/bin/python3
"""Existing AIRY/URDF safety input bridge with native original-navigation plugins.

Nav2 commands are proposals only. The separate existing watchdog owns hardware.
Action callbacks are fenced by goal generation so cancelled goals cannot revive.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,time,math
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile,DurabilityPolicy
from lifecycle_msgs.srv import GetState
from nav2_msgs.srv import ManageLifecycleNodes
from sensor_msgs_py import point_cloud2
from nav2_msgs.action import ComputePathToPose,FollowPath
from geometry_msgs.msg import PoseStamped,TransformStamped,TwistStamped
from nav_msgs.msg import Odometry,OccupancyGrid,Path as RosPath
from std_msgs.msg import String,Bool
from tf2_ros import TransformBroadcaster
from nav_runtime import Runtime
from nav_math import matrix,inverse,fill_pose,yaw,wrap,seconds,Rotation,transform
from nav_grid import ObservedGrid
from nav_footprint import chassis_footprint
from nav_obstacle_wait import ObstacleWait
from nav_collision_visual import collision_visual
from nav_nav2_math import advance_omni,planar
from nav_native_limits import native_limits,constrain_native
from nav_motion_feedback import MotionFeedback
from rclpy.qos import qos_profile_sensor_data
import copy


class NativeRuntime(Runtime):
    def __init__(self):
        self.approach_state={};self.approach_at=-1e9
        self.motion_feedback=MotionFeedback();self.chassis_at=-1e9;self.chassis_error="等待原底盘轮速与IMU反馈"
        self.epoch=0;self.follow_serial=0;self.plan_handle=None;self.follow_handle=None
        self.plan_future=None;self.follow_future=None;self.follow_pending=False
        self.native_command=(0.,0.,0.);self.native_at=0.;self.follow_started=0.;self.next_plan=0.;self.retry_at=0.
        self.global_map={};self.map_at=0.;self.grid=None;self.grid_at=0.;self.native_path=None
        self.node_states={};self.state_futures={};self.plan_started=0.;self.plan_ms=0.;self.path_length=0.
        self.manual_points=np.empty((0,3));self.manual_scan_at=-1e9;self.manual_pair_key=None
        self.startup_future=None;self.startup_at=-1e9;self.startup_done=False
        self.manual_enabled=False;self.manual_command=(0.,0.,0.);self.manual_at=0.;self.manual_seq=-1
        self.obstacle_wait=ObstacleWait()
        super().__init__()
        self.chassis_footprint=chassis_footprint(self.value('body_urdf'))
        self.native_limits=native_limits();self.speed_clamps=0
        self.tf=TransformBroadcaster(self);self.last_native_odom=None
        self.nav_odom_pub=self.create_publisher(Odometry,'nav2_odom',1)
        self.plan_client=ActionClient(self,ComputePathToPose,'compute_path_to_pose')
        self.follow_client=ActionClient(self,FollowPath,'follow_path')
        self.state_clients={name:self.create_client(GetState,name+'/get_state') for name in ['planner_server','controller_server']}
        latched=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(String,'global_map_status',self.map_status,latched)
        self.create_subscription(OccupancyGrid,'occupancy_static_map',self.map_grid,latched)
        self.create_subscription(TwistStamped,'nav2_command',self.velocity,1)
        self.create_subscription(String,'manual_intent',self.manual,1)
        self.create_subscription(String,'approach_status',self.approach_status,1)
        self.create_subscription(RosPath,'approach_path',self.approach_path,1)
        self.create_subscription(Odometry,'chassis_odom',self.chassis_odom,qos_profile_sensor_data)
        self.lifecycle_client=self.create_client(ManageLifecycleNodes,'navigation_lifecycle/manage_nodes')
        self.create_timer(.25,self.probe_nodes)

    def approach_path(self,m):
        if self.goal and self.follow_handle and m.header.frame_id=='airy_global':
            if m.poses:self.path_pub.publish(m)
            elif self.native_path:self.path_pub.publish(self.native_path)

    def approach_status(self,m):
        try:self.approach_state=json.loads(m.data);self.approach_at=time.monotonic()
        except (ValueError,TypeError):pass

    def probe_nodes(self):
        # Never invent airy_global from local odometry. Activate map-dependent
        # Nav2 nodes only after a real localization correction and map exist.
        now=time.monotonic()
        inputs=(self.safety.locked and now-self.safety.local<1.2 and now-self.safety.odom<.5
                and self.motion_feedback.correction is not None and now-self.chassis_at<.2 and self.body is not None and self.global_map.get('ready') and now-self.map_at<1.2)
        if inputs and not self.startup_done and self.startup_future is None and now-self.startup_at>5. and self.lifecycle_client.service_is_ready():
            req=ManageLifecycleNodes.Request();req.command=ManageLifecycleNodes.Request.STARTUP
            self.startup_at=now;self.startup_future=self.lifecycle_client.call_async(req)
            def started(f):
                self.startup_future=None
                try:self.startup_done=bool(f.result().success)
                except Exception:self.startup_done=False
            self.startup_future.add_done_callback(started)
        for name,client in self.state_clients.items():
            if client.service_is_ready() and name not in self.state_futures:
                f=client.call_async(GetState.Request());self.state_futures[name]=f
                def done(f,name=name):
                    self.state_futures.pop(name,None)
                    try:self.node_states[name]=(f.result().current_state.id,time.monotonic())
                    except Exception:self.node_states[name]=(0,0.)
                f.add_done_callback(done)

    def update_dynamic_envelope(self):
        super().update_dynamic_envelope()
        # The runtime is now the sole consumer of the full 3-D envelope. Apply
        # and acknowledge it here; Nav2 keeps the same independent chassis map.
        if self.handoff.acknowledge(self.envelope):self.model.protected_parts=self.handoff.guard()

    def odom(self,m):
        if m.header.frame_id!='airy_map' or m.child_frame_id!='airy_imu' or not self.current(m.header,.5):return
        super().odom(m)
        self.align_chassis(seconds(m.header),m)

    def update_correction(self,key,m):
        super().update_correction(key,m)
        if key in self.odoms:self.align_chassis(seconds(m.header),self.odoms[key])

    def align_chassis(self,stamp,odom):
        if self.correction is None or not self.safety.locked:return
        measured=self.correction@matrix(odom.pose.pose)@self.imu_base
        if self.motion_feedback.align(stamp,measured):
            current=self.motion_feedback.body()
            if current is not None:self.body=current

    def chassis_odom(self,m):
        if m.header.frame_id!='odom_s3' or m.child_frame_id!='base_link' or not self.current(m.header,.2):return
        v=m.twist.twist;velocity=(float(v.linear.x),float(v.linear.y),float(v.angular.z))
        try:
            if not self.motion_feedback.push(seconds(m.header),matrix(m.pose.pose),velocity):return
        except ValueError as e:
            self.chassis_at=-1e9;self.chassis_error=str(e);self.abort(self.chassis_error);return
        self.chassis_at=time.monotonic()-(time.time()-seconds(m.header));self.chassis_error=''
        local=self.motion_feedback.history[-1][1];links=[('scan345_odom','scan345_base',local)]
        current=self.motion_feedback.body()
        valid=(current is not None and self.safety.locked and time.time()-self.motion_feedback.aligned_stamp<1.2 and time.monotonic()-self.safety.local<1.2)
        if valid:
            self.body=current;links.insert(0,('airy_global','scan345_odom',self.motion_feedback.correction))
            b=Odometry();b.header=copy.deepcopy(m.header);b.header.frame_id='airy_global';b.child_frame_id='scan345_base';fill_pose(b.pose.pose,current);b.twist=copy.deepcopy(m.twist);self.body_pub.publish(b)
        transforms=[]
        for frame,child,pose in links:
            q=TransformStamped();q.header.stamp=m.header.stamp;q.header.frame_id=frame;q.child_frame_id=child
            q.transform.translation.x,q.transform.translation.y,q.transform.translation.z=map(float,pose[:3,3])
            values=Rotation.from_matrix(pose[:3,:3]).as_quat();q.transform.rotation.x,q.transform.rotation.y,q.transform.rotation.z,q.transform.rotation.w=map(float,values);transforms.append(q)
        self.tf.sendTransform(transforms)
        o=copy.deepcopy(m);o.header.frame_id='scan345_odom';o.child_frame_id='scan345_base';fill_pose(o.pose.pose,local);self.nav_odom_pub.publish(o)

    def pair(self,key):
        # A separate continuous-odometry cloud supports guarded manual control
        # even when saved-map localization has no unique match yet.
        if key!=self.manual_pair_key and key in self.clouds and key in self.odoms:
            cloud=self.clouds[key];odom=self.odoms[key]
            if self.current(cloud.header,.5) and self.current(odom.header,.5):
                good,why=self.geometry_health()
                if good:
                    a=point_cloud2.read_points(cloud,field_names=('x','y','z'),skip_nans=True)
                    if len(a)>=100:
                        points=np.column_stack([a['x'],a['y'],a['z']]);bp=transform(points,self.base_imu)
                        mask=self.sector.mask(bp) if self.sector else np.zeros(len(bp),dtype=bool)
                        # Non-sector production operation must retain its calibrated
                        # self filter; this path is not used to bypass that mode.
                        if self.model and not self.sector:
                            self.model.update();mask=self.model.filter_with_profile(bp,self.value('base_lidar_xyz'))
                            if self.residual:mask|=self.residual.mask(bp,self.model)
                        keep=(~mask)&(bp[:,2]>self.value('ground_clearance'))&(bp[:,2]<2.6)
                        self.manual_points=transform(points[keep],matrix(odom.pose.pose))
                        self.manual_scan_at=time.monotonic()-(time.time()-seconds(cloud.header));self.manual_pair_key=key
        super().pair(key)

    def manual_health(self):
        now=time.monotonic();ok,why=self.geometry_health()
        if not ok:return ok,why
        if self.local_body is None or now-self.safety.odom>=.5:return False,'手动控制等待实时里程计'
        if now-self.manual_scan_at>=.6:return False,'手动控制等待实时避障点云'
        return True,'手动控制就绪（不依赖地图定位）'

    def map_status(self,m):
        try:d=json.loads(m.data)
        except (ValueError,TypeError):return
        changed=d.get('generation')!=self.global_map.get('generation') or d.get('loaded_stamp')!=self.global_map.get('loaded_stamp')
        if changed:
            self.grid=None;self.grid_at=0.
            if self.goal:self.abort('导航底图已切换，请重新选择目标')
        self.global_map=d;self.map_at=time.monotonic()

    def map_grid(self,m):
        if m.header.frame_id!='airy_global' or not self.current(m.header,.8) or seconds(m.header)<self.global_map.get('loaded_stamp',float('inf')):return
        try:
            self.grid=ObservedGrid(m.data,m.info.width,m.info.height,m.info.resolution,[m.info.origin.position.x,m.info.origin.position.y],footprint=self.chassis_footprint)
            self.grid_at=time.monotonic()
        except ValueError:self.grid=None

    def health(self):
        now=time.monotonic()
        if now-self.backend_time>2.:return False,'等待三维定位状态'
        if self.backend.get('mode')!='localization':return False,'请先启动已有地图的三维定位'
        if not self.backend.get('locked'):return False,'三维定位尚未锁定：'+str(self.backend.get('state','等待重定位'))
        if now-self.chassis_at>.2:return False,self.chassis_error or '原底盘运动反馈中断'
        if self.motion_feedback.correction is None or time.time()-self.motion_feedback.aligned_stamp>1.2:return False,'等待三维定位与底盘里程计时间对齐'
        if not all(self.node_states.get(name,(0,0))[0]==3 and now-self.node_states[name][1]<.8 for name in ['planner_server','controller_server']):
            return False,'等待原导航Smac / MPPI就绪'
        self.safety.planner=now
        if now-self.map_at>1.2 or not self.global_map.get('ready'):return False,self.global_map.get('reason','等待全局导航底图')
        if self.global_map.get('map')!=self.backend.get('map'):return False,'定位地图与导航底图不一致'
        if self.grid is None or now-self.grid_at>1.2:return False,'全局地图更新中断'
        return super().health()

    def cancel_actions(self):
        self.follow_serial+=1;self.native_at=0.;self.native_command=(0.,0.,0.);self.follow_pending=False
        for handle in [self.plan_handle,self.follow_handle]:
            if handle:
                try:handle.cancel_goal_async()
                except Exception:pass
        self.plan_handle=self.follow_handle=None;self.plan_future=self.follow_future=None

    def abort(self,why,clear_map=False):
        self.obstacle_wait.clear()
        self.manual_enabled=False;self.manual_command=(0.,0.,0.)
        self.epoch+=1;self.cancel_actions();self.native_path=None;self.retry_at=0.
        super().abort(why,clear_map)

    def manual(self,m):
        try:
            d=json.loads(m.data);now=time.monotonic();seq=int(d['seq'])
            if d.get('session')!=self.session or seq<=self.manual_seq or not -.05<=now-float(d['stamp'])<.25:return
            self.manual_seq=seq;enabled=bool(d.get('enabled'));v=tuple(float(d[k]) for k in ['x','y','yaw'])
            if not all(math.isfinite(a) for a in v) or math.hypot(v[0],v[1])>.151 or abs(v[2])>.301:raise ValueError('invalid manual speed')
            if enabled and not self.manual_enabled:self.abort('手动键盘控制已启用，自动导航已取消')
            self.manual_enabled=enabled;self.manual_at=now;self.manual_command=v if enabled else (0.,0.,0.)
            if not enabled:self.send_drive(0.,0.,False)
        except (ValueError,TypeError,KeyError):
            self.manual_enabled=False;self.send_drive(0.,0.,False)

    def send_drive(self,x,rate,enabled,side=0.):
        self.drive_seq+=1
        self.drive_pub.publish(String(data=json.dumps(dict(session=self.session,seq=self.drive_seq,stamp=time.monotonic(),enabled=bool(enabled),mode='manual' if self.manual_enabled else 'auto',x=float(x),y=float(side),yaw=float(rate)))))

    def command(self,req,res):
        try:
            d=json.loads(req.payload or '{}')
            if d.get('session')!=self.session:raise ValueError('界面会话不匹配')
            if req.command=='cancel':self.abort('已取消目标并停车');res.success=True;res.message=self.state;return res
            if req.command!='goal':raise ValueError('未知命令')
            if self.manual_enabled:raise ValueError('请先关闭键盘手动控制，再下发导航目标')
            ok,why=self.health()
            if not ok or self.body is None:raise ValueError(why)
            if time.monotonic()-self.safety.ui>.3:raise ValueError('界面心跳未就绪')
            xy=np.array([float(d['x']),float(d['y'])]);heading=d.get('yaw')
            if not np.isfinite(xy).all() or heading is not None and not math.isfinite(float(heading)):raise ValueError('目标坐标或方向无效')
            why=self.grid.points_reason([xy])
            if why:raise ValueError('目标不可规划：'+why)
            why=self.grid.pose_reason(*self.body[:2,3],yaw(self.body))
            if why:raise ValueError('起点不可规划：'+why)
            execute=bool(d.get('execute',False))
            if execute and (not self.value('allow_hardware') or time.monotonic()-self.drive_status_time>.5 or not self.drive_status.get('ready')):raise ValueError('底盘执行或看门狗尚未就绪')
            self.abort('正在使用原导航Smac规划')
            delta=xy-self.body[:2,3]
            resolved=float(heading) if heading is not None else (math.atan2(delta[1],delta[0]) if np.linalg.norm(delta)>.05 else yaw(self.body))
            self.goal=dict(point=np.r_[xy,self.body[2,3]],yaw=resolved,requested_yaw=heading,execute=execute)
            self.obstacle_replans=0;self.next_plan=time.monotonic();self.plan_started=0.
            if execute:self.safety.arm()
            self.state='等待Smac规划；仅预览' if not execute else '等待Smac规划；执行已启用'
            res.success=True;res.message=self.state
        except (ValueError,TypeError,KeyError) as e:
            res.success=False;res.message=str(e);self.state='目标被拒绝：'+str(e)
        return res

    def request_plan(self):
        if not self.goal or self.plan_future is not None or self.plan_handle is not None:return
        if not self.plan_client.server_is_ready():return
        epoch=self.epoch;g=ComputePathToPose.Goal();g.planner_id='GridBased';g.use_start=False
        g.goal=PoseStamped();g.goal.header.frame_id='airy_global';g.goal.header.stamp=self.get_clock().now().to_msg()
        g.goal.pose.position.x,g.goal.pose.position.y=map(float,self.goal['point'][:2]);a=self.goal['yaw']
        g.goal.pose.orientation.z=math.sin(a/2);g.goal.pose.orientation.w=math.cos(a/2)
        self.plan_started=time.monotonic();self.plan_future=self.plan_client.send_goal_async(g)
        def accepted(f):
            try:h=f.result()
            except Exception as e:
                if epoch==self.epoch:self.abort('Smac请求异常：'+str(e))
                return
            if epoch!=self.epoch:
                if h.accepted:h.cancel_goal_async()
                return
            self.plan_future=None
            if not h.accepted:self.retry('Smac未接受目标');return
            self.plan_handle=h;h.get_result_async().add_done_callback(lambda f:self.planned(f,epoch))
        self.plan_future.add_done_callback(accepted)

    def planned(self,f,epoch):
        if epoch!=self.epoch or not self.goal:return
        self.plan_handle=None
        try:
            result=f.result();path=result.result.path
            if result.status!=4 or result.result.error_code!=0 or len(path.poses)<2:
                reason=result.result.error_msg or ('规划器返回错误码 %s'%result.result.error_code)
                if 'goal is occupied' in reason:reason='目标落位轮廓被占用或进入未知区域，请调整目标位置或朝向'
                self.retry('原导航未找到可行路径：'+reason);return
            if path.header.frame_id!='airy_global':raise ValueError('规划路径坐标系错误')
            xy=np.array([[p.pose.position.x,p.pose.position.y] for p in path.poses])
            if not np.isfinite(xy).all() or np.linalg.norm(xy[-1]-self.goal['point'][:2])>.02:raise ValueError('路径未到原目标')
            if self.goal['execute'] and self.obstacle_wait.active and not self.obstacle_wait.ready(time.monotonic()):
                self.retry('已有候选路径，等待当前位置连续3帧无冲突');return
            if self.goal['requested_yaw'] is None and self.native_path is None:
                delta=xy[-1]-xy[max(0,len(xy)-4)]
                if np.linalg.norm(delta)>.01:self.goal['yaw']=math.atan2(delta[1],delta[0])
            a=self.goal['yaw'];path.poses[-1].pose.orientation.z=math.sin(a/2);path.poses[-1].pose.orientation.w=math.cos(a/2)
            self.native_path=path;self.path_pub.publish(path)
            self.plan_ms=(time.monotonic()-self.plan_started)*1000;self.path_length=float(np.linalg.norm(np.diff(xy,axis=0),axis=1).sum());self.next_plan=time.monotonic()+2.
            if self.goal['execute'] and self.safety.armed:self.follow(path)
            else:
                self.obstacle_wait.clear();self.state='Smac全局路径已生成 · 仅预览'
        except Exception as e:self.abort('原导航路径校验失败：'+str(e))

    def follow(self,path):
        if not self.follow_client.server_is_ready():self.retry('MPPI服务未就绪');return
        if not path.poses:self.retry('MPPI收到空路径');return
        # A saved-map goal is fixed in airy_global, not in historical odometry.
        # Nav2's final checker must transform it using the latest map->odom TF,
        # consistently with the controller's current-time arrival measurement.
        path.poses[-1].header=copy.deepcopy(path.poses[-1].header)
        path.poses[-1].header.stamp.sec=0
        path.poses[-1].header.stamp.nanosec=0
        self.follow_serial+=1;serial=self.follow_serial;epoch=self.epoch
        self.follow_pending=True
        if self.follow_handle is None:self.native_at=0.
        self.follow_started=time.monotonic()
        g=FollowPath.Goal();g.path=path;g.controller_id='FollowPath';g.goal_checker_id='goal_checker';g.progress_checker_id='progress_checker'
        self.follow_future=self.follow_client.send_goal_async(g)
        def accepted(f):
            try:h=f.result()
            except Exception as e:
                if epoch==self.epoch and serial==self.follow_serial:self.abort('MPPI请求异常：'+str(e))
                return
            if epoch!=self.epoch or serial!=self.follow_serial:
                if h.accepted:h.cancel_goal_async()
                return
            self.follow_pending=False;self.follow_future=None
            if not h.accepted:self.retry('MPPI拒绝路径');return
            self.obstacle_wait.clear()
            self.follow_handle=h;self.accept_stamp=time.time();self.state='自动导航执行中'
            h.get_result_async().add_done_callback(lambda f:self.followed(f,epoch,serial))
        self.follow_future.add_done_callback(accepted)

    def followed(self,f,epoch,serial):
        if epoch!=self.epoch or serial!=self.follow_serial or not self.goal:return
        self.follow_handle=None
        try:r=f.result()
        except Exception as e:self.abort('MPPI结果异常：'+str(e));return
        if r.status==4 and r.result.error_code==0:self.abort('已到达目标并停稳')
        elif r.status!=5:self.retry('MPPI暂停：'+r.result.error_msg)

    def retry(self,reason):
        if not self.goal:return
        self.epoch+=1
        self.cancel_actions();self.send_drive(0.,0.,False);self.native_path=None
        self.obstacle_replans+=1
        self.obstacle_wait.begin(self.scans)
        self.retry_at=time.monotonic()+1.;self.next_plan=self.retry_at
        self.state=reason+'；已停车，保留目标持续重规划（第%d次；取消目标可终止）'%self.obstacle_replans

    def velocity(self,m):
        if not self.goal or not self.safety.armed or not self.follow_handle:return
        if not self.current(m.header,.25) or seconds(m.header)<getattr(self,'accept_stamp',0.):return
        v=(m.twist.linear.x,m.twist.linear.y,m.twist.angular.z)
        try:bounded=constrain_native(v,self.native_limits)
        except ValueError as e:self.abort(str(e)+'，已停车');return
        if bounded!=v:self.speed_clamps+=1
        self.native_command=bounded;self.native_at=time.monotonic()

    def tick(self):
        now=time.monotonic()
        if self.dynamic_geometry and self.model.health()[0]:
            try:
                if self.residual:self.residual.prepare(self.model.q,self.model.gripper_q)
                self.update_dynamic_envelope()
            except ValueError as e:self.abort(str(e));return
        if now-self.backend_time>1.:self.safety.locked=False
        ok,why=self.health();self.ready_pub.publish(Bool(data=ok));was_armed=self.safety.armed;self.safety.check()
        if self.goal and (not ok or was_armed and not self.safety.armed):self.abort(self.safety.fault or why)
        if self.goal and self.obstacle_wait.active and self.scans!=self.obstacle_wait.last_scan:
            clear=(self.body is not None and self.grid is not None
                   and not self.grid.pose_reason(*self.body[:2,3],yaw(self.body))
                   and self.gap(self.points,self.body)>0.)
            self.obstacle_wait.observe(self.scans,clear,now)
        if self.goal and self.retry_at and now>=self.retry_at:
            if self.settled_for_replan(now):self.retry_at=0.
            elif now-self.retry_at>3.:self.abort('重规划前未确认底盘停稳，已取消目标')
        if self.goal and not self.retry_at:
            distance=np.linalg.norm(self.body[:2,3]-self.goal['point'][:2])
            if now>=self.next_plan and (self.native_path is None or self.goal['execute'] and distance>.45):self.request_plan()
            if (self.plan_future is not None or self.plan_handle is not None) and now-self.plan_started>5:self.abort('Smac规划响应超时')
            if self.follow_pending and now-self.follow_started>3:self.abort('MPPI路径接收超时')
        vx=vy=w=0.;enabled=False
        manual_ok,manual_why=self.manual_health()
        manual=bool(self.manual_enabled and now-self.manual_at<.25 and now-self.safety.ui<.3 and manual_ok and self.value('allow_hardware'))
        if self.manual_enabled and not manual_ok:self.state=manual_why
        if self.manual_enabled:
            self.native_command=self.manual_command if manual else (0.,0.,0.);self.native_at=now
        if manual or self.goal and self.safety.armed and self.follow_handle and not self.retry_at:
            if now-self.drive_status_time>.5:self.abort('底盘看门狗状态中断')
            elif now-self.native_at>.35:
                if now-self.follow_started>1.:self.abort('MPPI控制输出中断，已停车')
            elif (self.local_body if manual else self.body) is not None:
                vx,vy,w=self.native_command;enabled=True
                collision_body=self.local_body if manual else self.body
                collision_points=self.manual_points if manual else self.points
                physical=np.array([*collision_body[:2,3],yaw(collision_body)])
                for t in np.linspace(0,.6,13):
                    p=advance_omni(physical,vx,vy,w,float(t));future=collision_body.copy();future[:2,3]=p[:2]
                    future[:3,:3]=Rotation.from_euler('z',wrap(p[2]-physical[2])).as_matrix()@collision_body[:3,:3]
                    reason='' if manual else self.grid.pose_reason(*p)
                    if reason or self.gap(collision_points,future)<=0.:
                        self.send_drive(0.,0.,False)
                        if not manual:self.record_collision(future,t,vx,w)
                        if manual:self.state='手动方向与实时障碍冲突，已停车'
                        else:self.retry(reason or '整机轮廓前方存在实时障碍')
                        vx=vy=w=0.;enabled=False;break
        self.send_drive(vx,w,enabled,vy)

    def report(self):
        ok,why=self.health()
        if ok and self.state=='等待三维定位和实时扫描':self.state='导航就绪，可预览或执行目标'
        d=dict(session=self.session,ready=ok,reason=why,state=self.state,planner='Smac＋MPPI连续位姿轨迹',armed=self.safety.armed,hardware=self.value('allow_hardware'),scans=self.scans,goal=self.goal is not None,trajectory=self.native_path is not None,body=self.body.tolist() if self.body is not None else None,map=self.backend.get('map',''),global_map=self.global_map,plan_ms=round(self.plan_ms,1),path_length=round(self.path_length,3),obstacle_replans=self.obstacle_replans,watchdog=self.drive_status.get('reason','等待'),tracking={})
        manual_ok,manual_why=self.manual_health()
        d.update(ground_filter=self.ground_filter_status,velocity_source='original_s3_wheel_imu',chassis_feedback_age=round(max(0,time.monotonic()-self.chassis_at),3),speed_policy='original_navigation',speed_clamps=self.speed_clamps,manual_linear=.15,manual_angular=.30,manual_ready=manual_ok,manual_reason=manual_why,manual_enabled=self.manual_enabled,max_speed=self.value('max_speed'),max_yaw_rate=self.value('max_yaw_rate'))
        d.update(obstacle_waiting=self.obstacle_wait.active,obstacle_clear_scans=self.obstacle_wait.clear_scans,replan_policy='until_goal_or_explicit_stop',replan_interval_s=1.)
        d['approach_control']=self.approach_state if self.goal and time.monotonic()-self.approach_at<.8 else {}
        if self.model:
            parts=getattr(self.model,'protected_parts',self.model.primitives)
            d['current_collision']=collision_visual(self.points,self.body,parts,time.monotonic()-self.safety.cloud<.6)
            d['collision_part_names']=self.model.primitive_names
            d['estimated_gripper_collision']=self.model.estimated_grippers
            d.update(body_parts=getattr(self.model,'protected_parts',self.model.primitives).tolist(),self_filtered=self.model.filtered,collision_padding_m=0.,posture='固定扇形过滤',joints_ok=self.model.health()[0])
        if self.sector:d.update(hard_filtered=int(self.sector.last_mask.sum())+int(self.sensor_filter.get('masked',0)),sector_config=self.sector.config,filter_mode='fixed_sector_volume',filter_status='固定扇形过滤 · 原导航规划控制')
        if self.last_collision:d['last_collision']=self.last_collision
        self.status_pub.publish(String(data=json.dumps(d,ensure_ascii=False,allow_nan=False)))


def main():
    rclpy.init();n=None
    try:n=NativeRuntime();rclpy.spin(n)
    except (KeyboardInterrupt,rclpy.executors.ExternalShutdownException):pass
    finally:
        if n:n.close()
        rclpy.try_shutdown()
if __name__=='__main__':main()
