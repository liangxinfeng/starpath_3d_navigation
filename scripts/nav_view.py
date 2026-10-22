import sys,math,time
from pathlib import Path
import numpy as np
from PyQt5 import QtCore as C,QtGui as G
from ament_index_python.packages import get_package_prefix
sys.path.insert(0,str(Path(get_package_prefix('starmap_3d_mapping'))/'lib/starmap_3d_mapping'))
from airy_view import MapView
from airy_fixed_sector import FixedSectorFilter

class NavView(MapView):
    goalSelected=C.pyqtSignal(float,float,object)
    def __init__(self):
        super().__init__();self.live=np.empty((0,3));self.live_on=True;self.pick=False;self.goal=None;self.body=None;self.self_returns=np.empty((0,3));self.body_parts=[]
        self.path_headings=[];self.sector_config=None
        self.collision_on=True;self.filter_on=False;self.collision={};self.collision_at=0.
        self.ground_returns=np.empty((0,3));self.planner_returns=np.empty((0,3));self.ground_on=False
        self.goal_yaw=None;self.goal_drag=None;self.pick_previous=None
    def reset_view(self,top=False):
        if getattr(self,'goal_drag',None) is not None:return
        super().reset_view(True if getattr(self,'pick',False) else top)
    def begin_pick(self):
        self.cancel_pick();self.reset_view(True);self.pick=True
        self.setCursor(C.Qt.CrossCursor);self.setFocus();self.update()
    def cancel_pick(self):
        if self.goal_drag is not None and self.pick_previous is not None:
            self.goal,self.goal_yaw=self.pick_previous
        self.pick=False;self.goal_drag=None;self.pick_previous=None;self.drag=None
        self.unsetCursor();self.update()
    def screen_on_plane(self,pos,z):
        # Invert the SAME projection used for drawing, including pan, zoom,
        # camera yaw and the selected goal plane height.
        scale=min(self.width()-60,self.height()-95)/(2*self.radius)*self.zoom
        if scale<=0:raise ValueError('无效视图比例')
        b=self.basis();uv=np.array([(pos.x()-self.width()/2-self.pan[0])/scale,
            -(pos.y()-self.height()/2-10-self.pan[1])/scale])
        if abs(np.linalg.det(b[:2,:2]))<1e-6:raise ValueError('请切换到俯视后选择目标')
        return self.center[:2]+np.linalg.solve(b[:2,:2],uv-b[:2,2]*(z-self.center[2]))
    def update_goal_drag(self,pos):
        if self.goal_drag is None:return
        start,z=self.goal_drag
        delta=self.screen_on_plane(pos,z)-np.asarray(self.goal[:2])
        self.goal_yaw=math.atan2(delta[1],delta[0]) if (pos-start).manhattanLength()>=8 and np.linalg.norm(delta)>1e-6 else None
        self.update()
    def set_live(self,xyz):
        a=np.asarray(xyz).reshape(-1,3);a=a[np.isfinite(a).all(axis=1)]
        self.live=a[::max(1,math.ceil(len(a)/6000))];self.update()
    def mousePressEvent(self,e):
        if self.pick and e.button()==C.Qt.RightButton:self.cancel_pick();e.accept();return
        if self.pick and e.button()==C.Qt.LeftButton:
            if not 38<=e.y()<self.height()-90:e.accept();return
            z=float(self.body[2,3]) if self.body is not None else float(self.center[2])
            x,y=self.screen_on_plane(e.pos(),z)
            self.pick_previous=(None if self.goal is None else list(self.goal),self.goal_yaw)
            self.goal=[float(x),float(y),z];self.goal_yaw=None;self.goal_drag=(e.pos(),z)
            self.drag=None;self.setFocus();self.update();e.accept();return
        super().mousePressEvent(e)
    def mouseMoveEvent(self,e):
        if self.goal_drag is not None:self.update_goal_drag(e.pos());e.accept();return
        super().mouseMoveEvent(e)
    def mouseReleaseEvent(self,e):
        if self.goal_drag is not None and e.button()==C.Qt.LeftButton:
            self.update_goal_drag(e.pos());x,y,_=self.goal;heading=self.goal_yaw
            self.goal_drag=None;self.pick_previous=None;self.pick=False;self.unsetCursor()
            self.goalSelected.emit(x,y,heading);self.update();e.accept();return
        super().mouseReleaseEvent(e)
    def wheelEvent(self,e):
        if self.goal_drag is not None:e.accept();return
        super().wheelEvent(e)
    def mouseDoubleClickEvent(self,e):
        if self.pick or self.goal_drag is not None:e.accept();return
        super().mouseDoubleClickEvent(e)
    def keyPressEvent(self,e):
        if e.key()==C.Qt.Key_Escape and self.pick:self.cancel_pick();e.accept();return
        super().keyPressEvent(e)
    def paintEvent(self,e):
        super().paintEvent(e)
        p=G.QPainter(self);p.setClipRect(0,38,self.width(),self.height()-90)
        p.setPen(G.QPen(G.QColor('#ffbd58'),2))
        last=None
        for x,y,z,a in self.path_headings:
            if last is not None and math.hypot(x-last[0],y-last[1])<.25 and abs(math.remainder(a-last[2],2*math.pi))<.45:continue
            last=(x,y,a)
            tip=np.array([x+.14*math.cos(a),y+.14*math.sin(a),z])
            tail=np.array([x,y,z])
            wing1=tip+[-.045*math.cos(a)+.03*math.sin(a),-.045*math.sin(a)-.03*math.cos(a),0]
            wing2=tip+[-.045*math.cos(a)-.03*math.sin(a),-.045*math.sin(a)+.03*math.cos(a),0]
            for line in ([tail,tip],[wing1,tip,wing2]):
                q=self.project(line);p.drawPolyline(G.QPolygonF([C.QPointF(float(u),float(v)) for u,v in q]))
        if self.live_on and len(self.live):
            a=self.live
            z=self.ground_z()
            if z is not None and self.clip_high:a=a[(a[:,2]-z>=-.2)&(a[:,2]-z<=self.high)]
            if len(a):
                q=self.project(a);p.setPen(G.QPen(G.QColor(255,93,190,175),2))
                p.drawPoints(G.QPolygonF([C.QPointF(float(x),float(y)) for x,y in q]))
        for enabled,points,color,size in [
                (self.ground_on,self.ground_returns,G.QColor(160,175,185,120),2),
                (self.live_on,self.planner_returns,G.QColor(255,100,85,230),3)]:
            if enabled and len(points):
                q=self.project(points);p.setPen(G.QPen(color,size))
                p.drawPoints(G.QPolygonF([C.QPointF(float(x),float(y)) for x,y in q]))
        if self.body is not None:
            from nav_math import transform
            if self.sector_config and self.filter_on:
                p.setPen(G.QPen(G.QColor('#ffd35c'),2,C.Qt.DashLine))
                for line in FixedSectorFilter(self.sector_config).wireframe():
                    q=self.project(transform(line,self.body))
                    p.drawPolyline(G.QPolygonF([C.QPointF(float(x),float(y)) for x,y in q]))
            fresh=self.collision.get('fresh') and time.monotonic()-self.collision_at<.6
            active=self.collision.get('parts',[]) if fresh else []
            for i,(kind,cx,cy,cz,hx,hy,hz) in enumerate(self.body_parts if self.collision_on else []):
                color=G.QColor('#ff7b3d' if i in active else '#48b9ed')
                p.setPen(G.QPen(color,3 if i in active else 2))
                lines=[]
                if kind==1:
                    angles=np.linspace(0,2*math.pi,41)
                    for ax,bx in [(0,1),(0,2),(1,2)]:
                        ring=np.tile([cx,cy,cz],(len(angles),1))
                        ring[:,ax]+=hx*np.cos(angles);ring[:,bx]+=hx*np.sin(angles);lines.append(ring)
                else:
                    corners=np.array([[cx+x*hx,cy+y*hy,cz+z*hz] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
                    for a in range(8):
                        for b in (a^1,a^2,a^4):
                            if b>a:lines.append(corners[[a,b]])
                for line in lines:
                    q=self.project(transform(line,self.body))
                    p.drawPolyline(G.QPolygonF([C.QPointF(float(x),float(y)) for x,y in q]))
        if self.live_on and len(self.self_returns):
            q=self.project(self.self_returns);p.setPen(G.QPen(G.QColor('#5bdee2'),3))
            p.drawPoints(G.QPolygonF([C.QPointF(float(x),float(y)) for x,y in q]))
        if self.goal is not None:
            q=self.project([self.goal])[0];p.setPen(G.QPen(G.QColor('#ffbd58'),3))
            p.drawEllipse(C.QPointF(*map(float,q)),7,7)
            if self.goal_yaw is not None:
                a=self.goal_yaw;scale=min(self.width()-60,self.height()-95)/(2*self.radius)*self.zoom
                length=float(np.clip(55/max(scale,1),.15,1.2));tail=np.array(self.goal);tip=tail+[length*math.cos(a),length*math.sin(a),0]
                wing1=tip+[-length*.25*math.cos(a)+length*.16*math.sin(a),-length*.25*math.sin(a)-length*.16*math.cos(a),0]
                wing2=tip+[-length*.25*math.cos(a)-length*.16*math.sin(a),-length*.25*math.sin(a)+length*.16*math.cos(a),0]
                for line in ([tail,tip],[wing1,tip,wing2]):
                    uv=self.project(line);p.drawPolyline(G.QPolygonF([C.QPointF(float(x),float(y)) for x,y in uv]))
                p.drawText(C.QPointF(float(q[0]+12),float(q[1]-12)),'目标朝向 %.1f°'%math.degrees(a))
        if self.collision_on:
            fresh=self.collision.get('fresh') and time.monotonic()-self.collision_at<.6
            points=self.collision.get('points',[]) if fresh else []
            if len(points):
                uv=self.project(points)
                p.setPen(G.QPen(G.QColor('white'),10));p.drawPoints(G.QPolygonF([C.QPointF(float(x),float(y)) for x,y in uv]))
                p.setPen(G.QPen(G.QColor('#e92323'),6));p.drawPoints(G.QPolygonF([C.QPointF(float(x),float(y)) for x,y in uv]))
            p.setPen(G.QColor('#17324d'));p.setBrush(G.QColor(255,255,255,225));p.drawRoundedRect(12,42,455,52,5,5)
            p.drawText(22,63,'实际碰撞范围：蓝色包围盒（夹爪估算球已取消）')
            text=('当前重叠 %d 点 · 橙红部件 / 红色冲突点'%self.collision.get('count',0)) if fresh else '等待新扫描，未显示历史冲突点'
            p.drawText(22,83,text);p.setBrush(C.Qt.NoBrush)
        if self.pick:
            p.setPen(G.QColor('#ffdf99'));p.drawText(16,60,'按住左键拖出朝向，松开完成；单击只选位置；右键取消')
        p.end()
