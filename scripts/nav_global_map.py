#!/usr/bin/python3
"""Full saved-map floor segmentation with a separate live obstacle layer."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json,time,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,DurabilityPolicy,qos_profile_sensor_data
from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import String,Header
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from nav_msgs.msg import OccupancyGrid,Odometry
from nav_math import extrinsics,inverse,seconds
from nav_static_map import load_bundle,StaticMap,LiveLayer,inflated


class GlobalMap(Node):
    def __init__(self):
        super().__init__('scan345_global_map')
        cal=Path(get_package_share_directory('starmap_3d_mapping'))/'config/factory_calibration_345.json'
        base,_=extrinsics([.2634,0.,.2],[0.,.436332,0.],json.loads(cal.read_text()))
        self.imu_base=inverse(base)
        self.backend={};self.backend_at=0.;self.signature=None;self.requested=None
        self.static=None;self.live=None;self.meta={};self.error='等待三维定位选择完整地图'
        self.loaded_stamp=0.;self.generation='';self.live_at=0.;self.last_check=0.
        self.sensors={};self.scans={};self.worker=ThreadPoolExecutor(max_workers=1);self.job=None
        self.scan_ms=0.;self.build_ms=0.;self.combined=None;self.inflation=None
        latched=QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub=self.create_publisher(OccupancyGrid,'occupancy_static_map',latched)
        self.inflated_pub=self.create_publisher(OccupancyGrid,'inflation_occupancy_map',latched)
        self.static_pub=self.create_publisher(OccupancyGrid,'saved_occupancy_map',latched)
        self.sg=self.create_publisher(PointCloud2,'saved_ground_points',latched)
        self.so=self.create_publisher(PointCloud2,'saved_obstacle_points',latched)
        self.ground=self.create_publisher(PointCloud2,'ground_points',qos_profile_sensor_data)
        self.obstacles=self.create_publisher(PointCloud2,'non_ground_points',qos_profile_sensor_data)
        self.status=self.create_publisher(String,'global_map_status',latched)
        self.create_subscription(String,'/airy_fast_lio/backend_status',self.backend_status,1)
        self.create_subscription(Odometry,'sensor_pose',self.sensor,qos_profile_sensor_data)
        self.create_subscription(PointCloud2,'chassis_obstacle_points',self.scan,qos_profile_sensor_data)
        self.create_timer(.2,self.tick)

    def backend_status(self,m):
        try:self.backend=json.loads(m.data);self.backend_at=time.monotonic()
        except (ValueError,TypeError):pass

    def sensor(self,m):
        if m.header.frame_id!='airy_global' or not -.05<=time.time()-seconds(m.header)<.8:return
        key=round(seconds(m.header)*1e6);p=m.pose.pose.position
        self.sensors[key]=np.array([p.x,p.y,p.z]);self.pair(key)
        while len(self.sensors)>12:self.sensors.pop(next(iter(self.sensors)))

    def scan(self,m):
        if m.header.frame_id!='airy_global' or not -.05<=time.time()-seconds(m.header)<.8:return
        key=round(seconds(m.header)*1e6);self.scans[key]=m;self.pair(key)
        while len(self.scans)>5:self.scans.pop(next(iter(self.scans)))

    def pair(self,key):
        if self.live is None or key not in self.scans or key not in self.sensors:return
        m=self.scans.pop(key);sensor=self.sensors.pop(key)
        if not -.05<=time.time()-seconds(m.header)<.8:return
        start=time.monotonic()
        try:
            a=point_cloud2.read_points(m,field_names=('x','y','z'),skip_nans=True)
            points=np.column_stack([a['x'],a['y'],a['z']])
            ground,obstacles=self.live.update(points,sensor,preclassified=True)
            self.ground.publish(point_cloud2.create_cloud_xyz32(m.header,ground.astype(np.float32)))
            self.obstacles.publish(point_cloud2.create_cloud_xyz32(m.header,obstacles.astype(np.float32)))
            self.combined=self.live.combined();self.inflation=inflated(self.combined,self.static.res)
            self.live_at=time.monotonic()-(time.time()-seconds(m.header));self.scan_ms=(time.monotonic()-start)*1000
            self.error=''
        except Exception as e:
            self.error='实时地图更新失败：'+str(e);self.live_at=0.

    def build(self,path):
        start=time.monotonic();points,bases,meta=load_bundle(path,self.imu_base)
        static=StaticMap(points,bases);meta['source_points']=len(points)
        return static,meta,(time.monotonic()-start)*1000

    def message(self,data):
        m=OccupancyGrid();m.header.frame_id='airy_global';m.header.stamp=self.get_clock().now().to_msg()
        m.info.map_load_time=rclpy.time.Time(seconds=self.loaded_stamp).to_msg()
        m.info.resolution=self.static.res;m.info.width=self.static.width;m.info.height=self.static.height
        m.info.origin.position.x=float(self.static.origin[0]);m.info.origin.position.y=float(self.static.origin[1]);m.info.origin.orientation.w=1.
        m.data=data.ravel().tolist();return m

    def tick(self):
        now=time.monotonic();backend_ok=now-self.backend_at<2 and self.backend.get('mode')=='localization'
        path=str(self.backend.get('map','')) if backend_ok else ''
        if now-self.last_check>1.:
            self.last_check=now
            try:
                p=Path(path).resolve();s=p.stat()
                if p.name!='manifest.json':raise ValueError('需要完整地图 manifest.json')
                sig=(str(p),s.st_mtime_ns,s.st_size)
            except (OSError,ValueError):sig=None
            if sig!=self.signature:
                self.signature=sig;self.static=self.live=None;self.live_at=0.;self.meta={};self.generation=''
                if self.job is None:self.requested=None
                self.error='正在分割完整保存地图' if sig else '等待有效的完整保存地图'
                self.scans.clear();self.sensors.clear()
            if sig is not None and self.static is None and self.job is None and self.requested!=sig:
                self.requested=sig;self.job=self.worker.submit(self.build,sig[0])
        if self.job and self.job.done():
            try:
                static,meta,ms=self.job.result()
                if self.requested==self.signature:
                    self.static=static;self.live=LiveLayer(static);self.meta=meta;self.build_ms=ms
                    self.loaded_stamp=time.time();self.generation=hashlib.sha256((str(self.signature)+meta['cloud_sha256']).encode()).hexdigest()[:20]
                    self.error='等待实时扫描叠加';self.combined=static.grid.copy();self.inflation=inflated(self.combined,static.res)
                    self.static_pub.publish(self.message(static.grid))
                    header=Header(frame_id='airy_global',stamp=self.get_clock().now().to_msg())
                    # Decimation is display-only. Segmentation uses every saved point.
                    for pub,pts in [(self.sg,static.ground),(self.so,static.obstacles)]:
                        pts=pts[::max(1,int(np.ceil(len(pts)/30000)))]
                        pub.publish(point_cloud2.create_cloud_xyz32(header,pts.astype(np.float32)))
            except Exception as e:self.error='完整地图处理失败：'+str(e)
            finally:
                self.job=None
                if self.requested!=self.signature:self.requested=None
        ready=bool(self.static is not None and backend_ok and path==self.meta.get('map') and now-self.live_at<1.2 and not self.error)
        if ready:
            self.pub.publish(self.message(self.combined));self.inflated_pub.publish(self.message(self.inflation))
        d=dict(ready=ready,reason=self.error or ('全图静态障碍＋当前帧有效障碍（不保留历史）' if ready else '等待定位或实时扫描更新'),generation=self.generation,loaded_stamp=self.loaded_stamp,**self.meta)
        if self.static is not None:
            s=self.static
            d.update(floor_policy='平地约束拟合＋地面下方回波不作为障碍',negative_obstacles_enabled=False,rejected_low_points=len(s.rejected_low_points),floor_max_slope_deg=2.0)
            d.update(width=s.width,height=s.height,resolution=s.res,ground_points=len(s.ground),static_obstacle_points=len(s.obstacles),free_cells=int(np.count_nonzero(s.grid==0)),unknown_cells=int(np.count_nonzero(s.grid<0)),static_obstacle_cells=int(np.count_nonzero(s.grid==100)),live_cells=len(self.live.hits),live_policy='current_scan_only',history_obstacle_cells=0,removed_previous_cells=self.live.removed_previous_cells,chassis_obstacle_height=s.top,floor_plane=s.plane.tolist(),floor_below_base=round(-s.base_offset,3),floor_error_p95=s.floor_error,build_ms=round(self.build_ms,1),scan_ms=round(self.scan_ms,1))
        self.status.publish(String(data=json.dumps(d,ensure_ascii=False)))

    def close(self):
        self.worker.shutdown(wait=True,cancel_futures=True);self.destroy_node()


def main():
    rclpy.init();node=None
    try:node=GlobalMap();rclpy.spin(node)
    except (KeyboardInterrupt,rclpy.executors.ExternalShutdownException):pass
    finally:
        if node:node.close()
        rclpy.try_shutdown()

if __name__=='__main__':main()
