#!/usr/bin/python3
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys,json,time,math,uuid,signal,subprocess,fcntl
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,DurabilityPolicy,qos_profile_sensor_data
from PyQt5 import QtWidgets as W,QtCore as C,QtGui as G
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from nav_msgs.msg import Path as RosPath
from std_msgs.msg import String
from starpath_3d_navigation.srv import NavCommand
from nav_view import NavView

class Console(W.QMainWindow):
    def __init__(self):
        super().__init__();self.setWindowTitle('星途三维导航');self.resize(1320,940)
        self.session=uuid.uuid4().hex;self.seq=0;self.node=Node('scan345_navigation_ui')
        self.proc=None;self.loc_proc=None;self.logfile=None;self.logread=None;self.stop_time=None;self.future=None
        self.state={};self.state_time=0.;self.backend={};self.backend_time=0.;self.last_live=0.;self.body=None;self.stopping=False;self.closed=False
        self.heart=self.node.create_publisher(String,'/scan345/ui_heartbeat',1)
        self.manual_pub=self.node.create_publisher(String,'/scan345/manual_intent',1);self.manual_seq=0;self.pressed=set();self.restart_after_stop=False
        self.client=self.node.create_client(NavCommand,'/scan345/command')
        self.node.create_subscription(String,'/scan345/status',self.status,1)
        self.node.create_subscription(String,'/airy_fast_lio/backend_status',self.backend_status,1)
        self.node.create_subscription(PointCloud2,'/airy_fast_lio/optimized_map',self.map_cloud,QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self.node.create_subscription(PointCloud2,'/scan345/cloud',self.live_cloud,1)
        self.node.create_subscription(PointCloud2,'/scan345/self_returns',self.self_cloud,1)
        self.node.create_subscription(PointCloud2,'/scan345/removed_ground_points',lambda m:self.filtered_layer(m,'ground_returns'),1)
        self.node.create_subscription(PointCloud2,'/scan345/non_ground_points',lambda m:self.filtered_layer(m,'planner_returns'),qos_profile_sensor_data)
        self.node.create_subscription(RosPath,'/scan345/trajectory',self.path,1)
        root=W.QWidget();self.setCentralWidget(root);layout=W.QVBoxLayout(root);layout.setContentsMargins(22,16,22,16);layout.setSpacing(10)
        title=W.QLabel('星途三维导航');title.setObjectName('title');layout.addWidget(title)
        subtitle=W.QLabel('StarPath  ·  星图三维定位 + 原导航Smac全局规划 + MPPI控制 · 整图规划、动态避障、停稳到点');subtitle.setObjectName('muted');layout.addWidget(subtitle)
        row=W.QHBoxLayout();self.map_path=W.QLineEdit();self.map_path.setReadOnly(True);self.map_path.setPlaceholderText('选择已有三维地图；若三维定位已启动，可直接连接')
        row.addWidget(self.map_path,1);choose=W.QPushButton('选择地图');choose.clicked.connect(self.choose_map);row.addWidget(choose)
        self.loc_btn=W.QPushButton('启动三维定位');self.loc_btn.clicked.connect(self.start_localization);row.addWidget(self.loc_btn);layout.addLayout(row)
        row=W.QHBoxLayout();self.start_btn=W.QPushButton('连接并启动导航测试');self.start_btn.clicked.connect(self.start);row.addWidget(self.start_btn)
        self.stop_btn=W.QPushButton('停止测试');self.stop_btn.clicked.connect(self.stop);row.addWidget(self.stop_btn)
        self.cancel_btn=W.QPushButton('取消目标 / 停车');self.cancel_btn.setObjectName('stop');self.cancel_btn.clicked.connect(self.cancel);row.addWidget(self.cancel_btn)
        self.status_label=W.QLabel('未启动 · 默认仅规划预览');row.addWidget(self.status_label,1);layout.addLayout(row)
        self.status_label.setWordWrap(True)
        self.detail=W.QLabel('等待三维定位；不会自动下发导航目标。');self.detail.setWordWrap(True);layout.addWidget(self.detail)
        speed_row=W.QHBoxLayout()
        speed_row.addWidget(W.QLabel('自动导航：末段曲线接近 / 行驶中同步转向'))
        self.manual_check=W.QCheckBox('键盘控制底盘 WASDQE');self.manual_check.toggled.connect(self.manual_toggle);speed_row.addWidget(self.manual_check)
        self.manual_hint=W.QLabel('W/S前后 · A/D平移 · Q/E旋转 · 空格停车');speed_row.addWidget(self.manual_hint,1);layout.addLayout(speed_row)
        self.view=NavView();self.view.setFocusPolicy(C.Qt.StrongFocus);self.view.goalSelected.connect(self.picked);layout.addWidget(self.view,1)
        row=W.QHBoxLayout()
        self.collision_check=W.QCheckBox('显示实际碰撞轮廓与当前冲突点');self.collision_check.setChecked(True)
        self.collision_check.toggled.connect(lambda v:(setattr(self.view,'collision_on',v),self.view.update()));row.addWidget(self.collision_check)
        self.filter_check=W.QCheckBox('显示过滤体积（黄虚线，非碰撞范围）')
        self.filter_check.toggled.connect(lambda v:(setattr(self.view,'filter_on',v),self.view.update()));row.addWidget(self.filter_check)
        b=W.QPushButton('查看碰撞轮廓');b.clicked.connect(self.focus_collision);row.addWidget(b);row.addStretch();layout.addLayout(row)
        row=W.QHBoxLayout();self.live_check=W.QCheckBox('有效障碍：紫红碰撞 / 红色规划');self.live_check.setChecked(True);self.live_check.toggled.connect(self.toggle_live);row.addWidget(self.live_check)
        self.ground_check=W.QCheckBox('对照已剔除地面（灰，不参与避障）');self.ground_check.toggled.connect(self.toggle_ground);row.addWidget(self.ground_check)
        for name,fn in [('俯视',lambda:self.view.reset_view(True)),('三维视角',self.view.reset_view),('跟随机器人',self.center_robot)]:
            b=W.QPushButton(name);b.clicked.connect(fn);row.addWidget(b)
        self.pick_btn=W.QPushButton('拖选目标位置 / 朝向');self.pick_btn.clicked.connect(self.pick_goal);row.addWidget(self.pick_btn)
        row.addStretch();layout.addLayout(row)
        row=W.QHBoxLayout();self.x=W.QDoubleSpinBox();self.y=W.QDoubleSpinBox()
        for name,b in [('目标 X',self.x),('Y',self.y)]:b.setRange(-1000,1000);b.setDecimals(2);row.addWidget(W.QLabel(name));row.addWidget(b)
        self.heading_check=W.QCheckBox('到点朝向');row.addWidget(self.heading_check);self.heading=W.QDoubleSpinBox();self.heading.setRange(-180,180);self.heading.setSuffix('°');row.addWidget(self.heading)
        self.plan_btn=W.QPushButton('生成轨迹预览');self.plan_btn.clicked.connect(lambda:self.goal(False));row.addWidget(self.plan_btn)
        self.execute_btn=W.QPushButton('执行此目标');self.execute_btn.setObjectName('execute');self.execute_btn.clicked.connect(lambda:self.goal(True));row.addWidget(self.execute_btn)
        layout.addLayout(row)
        self.goal_hint=W.QLabel('拖选：位置＋到点朝向；单击：沿进场方向停车。随后预览或执行。');layout.addWidget(self.goal_hint)
        for b in (self.x,self.y,self.heading):b.valueChanged.connect(self.sync_goal)
        self.heading_check.toggled.connect(self.sync_goal)
        notice=W.QLabel('自动导航采用原速度与到点减速逻辑；键盘手动控制独立于地图定位 · 地面默认隐藏；勾选灰色对照可查看剔除效果 / 自体青色 / 路径橙色\n黄色线框：上身固定扇形＋底盘内部22厘米圆柱，不随关节变化；区内所有点均剔除。\n蓝色：当前实际检测包围盒；橙红：当前冲突部件；红色大点：实时冲突点（不受高度裁剪）。模型额外外扩0厘米；夹爪估算球已取消，腕部连接件仍保留。非地面有效点参与避障，实时规划障碍只取当前扫描；反馈或定位异常、窗口失焦仍会停车。');notice.setObjectName('muted');layout.addWidget(notice)
        self.logs=W.QPlainTextEdit();self.logs.setReadOnly(True);self.logs.setMaximumHeight(120);self.logs.setMaximumBlockCount(180);layout.addWidget(self.logs)
        self.setStyleSheet('QMainWindow,QWidget{background:#f1f5fa;color:#17324d;font-family:"Noto Sans CJK SC";font-size:14px} QLabel#title{font-size:28px;font-weight:700;color:#102d4b} QLabel#muted{color:#61758a;font-size:12px} QPushButton{background:white;border:1px solid #c5d3e1;border-radius:7px;padding:9px 12px} QPushButton:hover{background:#e3effb} QPushButton:disabled{color:#9ba9b6} QPushButton#stop{color:#b13135;background:#fff0ef} QPushButton#execute{color:white;background:#246b69} QLineEdit,QDoubleSpinBox,QPlainTextEdit{background:white;border:1px solid #c5d3e1;border-radius:5px;padding:5px}')
        self.timer=C.QTimer(self);self.timer.timeout.connect(self.tick);self.timer.start(40)
        self.heartbeat_timer=C.QTimer(self);self.heartbeat_timer.timeout.connect(self.send_heartbeat);self.heartbeat_timer.start(80)
        self.manual_timer=C.QTimer(self);self.manual_timer.timeout.connect(self.manual_tick);self.manual_timer.start(50)
        W.QApplication.instance().installEventFilter(self)
    def log(self,text):self.logs.appendPlainText(str(text))
    def send_heartbeat(self,stop=False):
        if self.closed or not self.node.context.ok():return
        self.seq+=1;self.heart.publish(String(data=json.dumps(dict(session=self.session,seq=self.seq,stamp=time.monotonic(),stop=stop))))
    def request(self,command,data):
        if self.future:self.log('正在处理上一项操作');return
        if not self.client.service_is_ready():self.log('导航服务尚未就绪');return
        r=NavCommand.Request();r.command=command;r.payload=json.dumps(dict(data,session=self.session))
        self.future=self.client.call_async(r);self.request_time=time.monotonic()
    def cancel(self):
        self.pressed.clear()
        if hasattr(self,'manual_check'):self.manual_check.setChecked(False)
        self.send_heartbeat(True)
        if self.client.service_is_ready() and not self.future:self.request('cancel',{})
    def goal(self,execute):
        self.view.cancel_pick();self.sync_goal()
        self.request('goal',dict(x=self.x.value(),y=self.y.value(),yaw=math.radians(self.heading.value()) if self.heading_check.isChecked() else None,execute=execute))
    def choose_map(self):
        self.cancel()
        path,_=W.QFileDialog.getOpenFileName(self,'选择完整三维地图',str(Path.home()/'ros2_ws/maps/airy_3d'),'地图 (manifest.json)')
        if path:self.map_path.setText(path)
    def start_localization(self):
        if time.monotonic()-self.backend_time<2:self.log('已有三维定位/建图进程，请在原界面操作；本界面不重复启动。');return
        if self.loc_proc and self.loc_proc.poll() is None:return
        if not Path(self.map_path.text()).is_file():self.log('请先选择已有完整地图');return
        logdir=Path.home()/'ros2_ws/logs/starpath_3d_navigation';logdir.mkdir(parents=True,exist_ok=True)
        with (logdir/'localization.log').open('a') as f:
            self.loc_proc=subprocess.Popen(['ros2','launch','starmap_3d_mapping','mapping.launch.py','mode:=localization','localization_algorithm:=context_localization','fixed_sector_filter:=true',f'map_path:={self.map_path.text()}'],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        self.log('已启动三维定位，请保持静止等待锁定。')
    def start(self):
        if self.proc and self.proc.poll() is None:return
        logdir=Path.home()/'ros2_ws/logs/starpath_3d_navigation';logdir.mkdir(parents=True,exist_ok=True)
        path=logdir/(time.strftime('%Y%m%d_%H%M%S')+'.log');self.logfile=path.open('w');self.logread=path.open('r')
        self.state={};self.stop_time=None
        self.proc=subprocess.Popen(['ros2','launch','starpath_3d_navigation','test.launch.py',f'session:={self.session}','allow_hardware:=true','fixed_sector_filter:=true'],stdout=self.logfile,stderr=subprocess.STDOUT,start_new_session=True)
        self.log('正在连接三维定位、启动规划器；底盘执行关闭。')
    def stop(self):
        self.restart_after_stop=False
        self.cancel()
        if self.proc and self.proc.poll() is None and self.stop_time is None:
            os.kill(self.proc.pid,signal.SIGINT);self.stop_time=time.monotonic()
    def backend_status(self,m):
        try:self.backend=json.loads(m.data);self.backend_time=time.monotonic()
        except ValueError:pass
    def status(self,m):
        try:d=json.loads(m.data)
        except ValueError:return
        if d.get('session')!=self.session:return
        message=d.get('state','')
        if message!=self.state.get('state') and any(v in message for v in ('拒绝','超时','受阻','被占用','触发停车','规划失败')):
            self.log(message)
        self.state=d;self.state_time=time.monotonic()
        self.view.body_parts=d.get('body_parts',[])
        self.view.collision=d.get('current_collision',{});self.view.collision_at=time.monotonic()
        self.view.sector_config=d.get('sector_config')
        if d.get('body') is not None:
            self.body=np.array(d['body']);self.view.body=self.body;self.view.position=tuple(self.body[:3,3]);self.view.update()
        if d.get('map'):self.map_path.setText(d['map'])
    def xyz(self,m,limit):
        a=point_cloud2.read_points(m,field_names=('x','y','z'),skip_nans=True);a=a[::max(1,math.ceil(len(a)/limit))]
        return np.column_stack([a['x'],a['y'],a['z']])
    def map_cloud(self,m):
        if m.header.frame_id=='airy_global':self.view.set_cloud(self.xyz(m,150000))
    def live_cloud(self,m):
        stamp=m.header.stamp.sec+m.header.stamp.nanosec*1e-9
        if m.header.frame_id=='airy_global' and -.05<=time.time()-stamp<.8:
            self.view.set_live(self.xyz(m,6000));self.last_live=time.monotonic()
    def filtered_layer(self,m,name):
        stamp=m.header.stamp.sec+m.header.stamp.nanosec*1e-9
        if m.header.frame_id=='airy_global' and -.05<=time.time()-stamp<.8:
            setattr(self.view,name,self.xyz(m,3500));self.view.update()
    def toggle_ground(self,v):self.view.ground_on=v;self.view.update()
    def self_cloud(self,m):
        stamp=m.header.stamp.sec+m.header.stamp.nanosec*1e-9
        if m.header.frame_id=='airy_global' and -.05<=time.time()-stamp<.8:
            self.view.self_returns=self.xyz(m,3000);self.view.update()
    def path(self,m):
        if m.header.frame_id=='airy_global':
            self.view.path=[(p.pose.position.x,p.pose.position.y,p.pose.position.z) for p in m.poses]
            self.view.path_headings=[(p.pose.position.x,p.pose.position.y,p.pose.position.z,2*math.atan2(p.pose.orientation.z,p.pose.orientation.w)) for p in m.poses]
            self.view.update()
    def toggle_live(self,v):self.view.live_on=v;self.view.update()
    def center_robot(self):
        if self.body is not None:self.view.center=self.body[:3,3].copy();self.view.radius=4.;self.view.pan[:]=0;self.view.update()
    def focus_collision(self):
        if self.body is None:return
        self.view.reset_view(False);self.view.center=self.body[:3,3].copy()+[0,0,.75]
        self.view.radius=1.5;self.view.zoom=1.;self.view.pan[:]=0
        self.collision_check.setChecked(True);self.view.update()
    def pick_goal(self):self.view.begin_pick();self.log('左键按下确定目标位置，拖动箭头确定朝向后松开；单击只选位置。选择后可预览或执行。')
    def sync_goal(self,*args):
        self.view.goal=[self.x.value(),self.y.value(),float(self.body[2,3]) if self.body is not None else 0.]
        self.view.goal_yaw=math.radians(self.heading.value()) if self.heading_check.isChecked() else None
        direction='朝向 %.1f°'%self.heading.value() if self.heading_check.isChecked() else '到点朝向不限'
        self.goal_hint.setText('目标 X %.2f m / Y %.2f m · %s'%(self.x.value(),self.y.value(),direction));self.view.update()
    def picked(self,x,y,heading):
        widgets=(self.x,self.y,self.heading,self.heading_check)
        for w in widgets:w.blockSignals(True)
        self.x.setValue(x);self.y.setValue(y);self.heading_check.setChecked(heading is not None)
        if heading is not None:self.heading.setValue(math.degrees(heading))
        for w in widgets:w.blockSignals(False)
        self.sync_goal()
    def tick(self):
        for _ in range(10):rclpy.spin_once(self.node,timeout_sec=0.)
        now=time.monotonic();running=self.proc is not None and self.proc.poll() is None
        fresh=now-self.state_time<1.;ready=running and fresh and self.state.get('ready') and self.stop_time is None
        self.start_btn.setEnabled(not running);self.stop_btn.setEnabled(running and self.stop_time is None)
        self.plan_btn.setEnabled(bool(ready and not self.future));self.execute_btn.setEnabled(bool(ready and not self.future and not self.state.get('armed')))
        self.status_label.setText(self.state.get('state','等待导航服务') if fresh and running else '已停止 / 未连接')
        self.detail.setText(('定位：'+str(self.backend.get('state','等待'))+'  |  '+self.state.get('reason','等待输入')+'  |  '+self.state.get('filter_status','等待自体过滤')+' · '+str(self.state.get('hard_filtered',0))+' 点/帧'))
        self.detail.setText(self.detail.text()+'\n'+self.state.get('manual_reason','手动控制等待实时输入'))
        gf=self.state.get('ground_filter',{});gm=self.state.get('global_map',{})
        mode='当前平面校正' if gf.get('mode')=='current_supported_plane' else '地图地面基准'
        self.detail.setText(self.detail.text()+'\n地面剔除 %d 点/帧（%s） · 规划有效点 %d · 历史残留不参与规划'%(gf.get('removed_points',0),mode,gf.get('planner_points',0)))
        tc=self.state.get('approach_control',{})
        if tc:
            phase='平滑接近终点' if tc.get('active') else '沿路径行驶'
            if not tc.get('active') and tc.get('distance',1)<.15 and tc.get('yaw_error',0)>.05:phase='终点朝向收尾'
            if tc.get('holding'):phase='到点停稳确认'
            self.detail.setText(self.detail.text()+'\n%s · 剩余路径 %.2f米 · 朝向误差 %.1f°'%(phase,tc.get('remaining',0),math.degrees(tc.get('yaw_error',0))))
        sf=gf.get('speckle_filter',{})
        if sf:self.detail.setText(self.detail.text()+'\n瞬态孤点剔除 %d 点/帧 · 连续帧确认稀疏点 %d · 处理 %.1f ms'%(sf.get('removed_points',0),sf.get('persistent_sparse_points',0),sf.get('processing_ms',0)))
        sector=self.state.get('sector_config')
        if sector:self.detail.setText(self.detail.text()+'\n固定扇形：中心 X %.0f cm / 半径 %.0f cm / 左右各 %.0f° / 高度 %.0f–%.0f cm'%(sector['origin_x']*100,sector['radius']*100,sector['half_angle_deg'],sector['z_min']*100,sector['z_max']*100))
        b=self.state.get('current_collision',{})
        if b.get('fresh') and b.get('count',0):
            x,y,z=b['xyz'];where=('%s%.0f厘米，%s%.0f厘米，高%.0f厘米'%('前' if x>=0 else '后',abs(x)*100,'左' if y>=0 else '右',abs(y)*100,z*100))
            self.detail.setText(self.detail.text()+'\n当前冲突 %d 点；最深点：%s（相对底盘中心），侵入轮廓 %.1f厘米'%(b['count'],where,-b['gap']*100))
        elif b.get('fresh'):self.detail.setText(self.detail.text()+'\n当前姿态：实时点云与碰撞轮廓无重叠（前进方向仍单独检查）')
        else:self.detail.setText(self.detail.text()+'\n碰撞显示等待新扫描；历史停车点不作为当前冲突显示')
        if now-self.last_live>1.:
            self.view.self_returns=np.empty((0,3));self.view.ground_returns=np.empty((0,3));self.view.planner_returns=np.empty((0,3))
            if len(self.view.live):self.view.set_live(np.empty((0,3)))
        if self.logread:
            text=self.logread.read(2500)
            if text:self.log(text.rstrip())
        if self.future:
            if self.future.done():
                try:self.log(self.future.result().message)
                except Exception as e:self.log(e)
                self.future=None
            elif now-self.request_time>4:self.future.cancel();self.future=None;self.send_heartbeat(True);self.log('请求超时，已发送停车')
        if running and self.stop_time and now-self.stop_time>10:
            try:os.killpg(self.proc.pid,signal.SIGTERM)
            except ProcessLookupError:pass
        if self.proc and not running:
            if self.logfile:self.logfile.close();self.logfile=None
            if self.logread:self.logread.close();self.logread=None
            if self.restart_after_stop:
                self.restart_after_stop=False;self.start()
    def manual_toggle(self,enabled):
        self.pressed.clear()
        if enabled:self.send_heartbeat(True);self.log('键盘手动模式已开启；按键前请点地图，自动导航已取消。')
        self.manual_tick(force=True)
    def manual_tick(self,force=False):
        if self.closed or not self.node.context.ok():return
        if not force and not self.manual_check.isChecked():return
        enabled=self.manual_check.isChecked() and self.isActiveWindow()
        keys=self.pressed if enabled else set()
        x=float(C.Qt.Key_W in keys)-float(C.Qt.Key_S in keys)
        y=float(C.Qt.Key_A in keys)-float(C.Qt.Key_D in keys)
        w=float(C.Qt.Key_Q in keys)-float(C.Qt.Key_E in keys)
        # Manual jogging stays separate from automatic navigation speed.
        linear=float(self.state.get('manual_linear',.15));angular=float(self.state.get('manual_angular',.30))
        norm=max(1.,math.hypot(x,y));x*=linear/norm;y*=linear/norm;w*=angular
        self.manual_seq+=1;self.manual_pub.publish(String(data=json.dumps(dict(session=self.session,seq=self.manual_seq,stamp=time.monotonic(),enabled=enabled,x=x,y=y,yaw=w))))
    def eventFilter(self,obj,e):
        if not hasattr(self,'manual_check'):return False
        if e.type() in (C.QEvent.WindowDeactivate,C.QEvent.FocusOut):
            self.pressed.clear()
        if self.manual_check.isChecked() and e.type() in (C.QEvent.KeyPress,C.QEvent.KeyRelease):
            if e.key() in (C.Qt.Key_Space,C.Qt.Key_Escape):self.cancel();return True
            keys=(C.Qt.Key_W,C.Qt.Key_A,C.Qt.Key_S,C.Qt.Key_D,C.Qt.Key_Q,C.Qt.Key_E)
            if e.key() in keys:
                if e.isAutoRepeat():return True
                if e.type()==C.QEvent.KeyRelease:self.pressed.discard(e.key());return True
                if isinstance(W.QApplication.focusWidget(),(W.QLineEdit,W.QTextEdit,W.QPlainTextEdit,W.QAbstractSpinBox)):return False
                if self.isActiveWindow():self.pressed.add(e.key())
                return True
        return False
    def changeEvent(self,e):
        if e.type()==C.QEvent.ActivationChange and not self.isActiveWindow() and hasattr(self,'heart'):self.send_heartbeat(True)
        super().changeEvent(e)
    def keyPressEvent(self,e):
        if e.key() in (C.Qt.Key_Space,C.Qt.Key_Escape):self.cancel();e.accept()
        else:super().keyPressEvent(e)
    def closeEvent(self,e):
        self.stop()
        if self.proc and self.proc.poll() is None:self.log('正在停止导航，请稍后关闭。');e.ignore();return
        if self.loc_proc and self.loc_proc.poll() is None:
            os.kill(self.loc_proc.pid,signal.SIGINT);self.loc_proc=None
        self.closed=True;self.timer.stop();self.heartbeat_timer.stop();self.manual_timer.stop();W.QApplication.instance().removeEventFilter(self);self.node.destroy_node();e.accept()

def main():
    app=W.QApplication(sys.argv)
    lock=(Path('/tmp')/('scan345_navigation_ui_'+os.environ.get('ROS_DOMAIN_ID','0')+'.lock')).open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:W.QMessageBox.information(None,'星途三维导航','测试界面已经打开。');return
    rclpy.init(args=[]);w=Console();w.show()
    if '--screenshot' in sys.argv:
        path=sys.argv[sys.argv.index('--screenshot')+1];C.QTimer.singleShot(6000,lambda:(w.grab().save(path),w.close()))
    code=app.exec_();rclpy.try_shutdown();sys.exit(code)
if __name__=='__main__':main()
