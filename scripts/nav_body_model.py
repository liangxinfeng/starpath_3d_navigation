"""Read-only S3 kinematics and narrow mesh-surface self-return filtering.

Meshes stay on the robot. Gripper motor units are NOT interpreted as finger
angles: uncalibrated DM fingers are never masked. Their complete URDF chain
reach can optionally be reserved as a wrist-centred collision sphere.
The SCAN runtime explicitly disables these uncalibrated gripper estimates.
"""
import math
import time
import hashlib
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree, ConvexHull
from scipy.spatial.transform import Rotation
from nav_math import transform, inverse

GROUPS = {
    'leg_waist_joint_state': ['Joint_Ankle','Joint_Knee','Joint_Waist_Pitch','Joint_Waist_Yaw'],
    'neck_joint_state': ['Joint_Neck_Roll','Joint_Neck_Pitch','Joint_Neck_Yaw'],
    **{s.lower()+'_arm_joint_state': ['Joint_'+s+'_'+j for j in
        ['Shoulder_Inner','Shoulder_Outer','UpperArm','Elbow','Forearm','Wrist_Upper','Wrist_Lower']]
        for s in ('Left','Right')}}

def origin(e):
    t=np.eye(4)
    if e is not None:
        t[:3,3]=np.fromstring(e.get('xyz','0 0 0'),sep=' ')
        t[:3,:3]=Rotation.from_euler('xyz',np.fromstring(e.get('rpy','0 0 0'),sep=' ')).as_matrix()
    return t

def triangles(path):
    data=Path(path).read_bytes()
    if len(data)>=84:
        n=int.from_bytes(data[80:84],'little')
        if 84+50*n==len(data):
            dtype=np.dtype([('normal','<f4',(3,)),('v','<f4',(3,3)),('attr','<u2')])
            return np.frombuffer(data,dtype=dtype,count=n,offset=84)['v'].astype(float)
    vertices=[list(map(float,l.split()[1:])) for l in data.decode().splitlines() if l.strip().startswith('vertex ')]
    return np.array(vertices).reshape(-1,3,3)

def primitive_distances(points, primitives):
    """Signed distance to a union of base-frame boxes and wrist-centred spheres.

    Row = [kind, cx, cy, cz, hx, hy, hz], kind 0 box / 1 sphere.
    Bounds already contain the physical safety margin. No full-turn circle.
    """
    best=np.full(len(points),10.)
    for kind,cx,cy,cz,hx,hy,hz in primitives:
        d=np.abs(points-[cx,cy,cz])
        if kind==1:
            gap=np.linalg.norm(d,axis=1)-hx
        else:
            q=d-[hx,hy,hz]
            gap=np.linalg.norm(np.maximum(q,0),axis=1)+np.minimum(np.max(q,axis=1),0)
        best=np.minimum(best,gap)
    return best

def merge_primitives(previous, current):
    """Grow only within a verified stationary posture; avoid noisy shrink/reset."""
    if not len(previous):return np.asarray(current).copy()
    out=np.asarray(previous).copy()
    if out.shape!=np.asarray(current).shape:raise ValueError('机身部件集合改变')
    for old,new in zip(out,current):
        if old[0]!=new[0]:raise ValueError('机身部件类型改变')
        if old[0]==1:
            radius=max(old[4],np.linalg.norm(old[1:4]-new[1:4])+new[4])
            old[4:]=math.ceil((radius-1e-10)/.01)*.01
        else:
            low=np.floor((np.minimum(old[1:4]-old[4:],new[1:4]-new[4:])+1e-10)/.01)*.01
            high=np.ceil((np.maximum(old[1:4]+old[4:],new[1:4]+new[4:])-1e-10)/.01)*.01
            old[1:4]=(low+high)/2;old[4:]=(high-low)/2
    return out

def swept_poses(start,end,step=.02,angle_step=math.radians(1)):
    """Planar swept footprint, including rotations; preserve initial tilt."""
    a=math.atan2(start[1,0],start[0,0]);b=math.atan2(end[1,0],end[0,0])
    delta=(b-a+math.pi)%(2*math.pi)-math.pi
    n=max(1,math.ceil(np.linalg.norm(end[:3,3]-start[:3,3])/step),math.ceil(abs(delta)/angle_step))
    for f in np.linspace(0,1,n+1):
        t=start.copy();t[:3,3]=(1-f)*start[:3,3]+f*end[:3,3]
        t[:3,:3]=Rotation.from_euler('z',delta*f).as_matrix()@start[:3,:3]
        yield t

class BodyModel:
    def __init__(self,urdf,tolerance=.012,nominal_geometry=False,estimated_grippers=True):
        if not 0<tolerance<=.015:raise ValueError('自体过滤容差必须在0～15毫米')
        self.estimated_grippers=bool(estimated_grippers)
        self.nominal_geometry=nominal_geometry;self.mesh_hulls={}
        self.tolerance=tolerance;self.joints={};self.shapes={};self.grippers={}
        self.q=None;self.history=deque(maxlen=120);self.last=-1e9;self.producer=None;self.producer_rx=None;self.error='等待关节反馈'
        path=Path(urdf);self.model_hash=hashlib.sha256(path.read_bytes()).hexdigest();root=ET.parse(path).getroot()
        for j in root.findall('joint'):
            self.joints[j.find('child').get('link')]=dict(name=j.get('name'),kind=j.get('type'),
                parent=j.find('parent').get('link'),origin=origin(j.find('origin')),
                axis=np.fromstring(j.find('axis').get('xyz','1 0 0') if j.find('axis') is not None else '1 0 0',sep=' '))
        rng=np.random.default_rng(345)
        for link in root.findall('link'):
            name=link.get('name')
            # The rigid wrist adaptor is known; DM moving fingers are not calibrated.
            unknown='_Gripper' in name and not name.endswith('Wrist_Lower_to_Gripper')
            for c in link.findall('collision'):
                mesh=c.find('geometry/mesh')
                if mesh is None:raise ValueError('不支持的碰撞几何：'+name)
                filename=mesh.get('filename')
                if filename.startswith('package://'):raise ValueError('模型需要可解析的本机网格路径')
                tri=triangles(path.parent/filename)
                tri*=np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
                vertices=transform(tri.reshape(-1,3),origin(c.find('origin'))).reshape(-1,3,3)
                if unknown:
                    # Sum of joint-origin lengths + maximum mesh radius bounds all
                    # finger configurations, without inventing a motor-angle mapping.
                    current=name;reach=float(np.max(np.linalg.norm(vertices.reshape(-1,3),axis=1)))
                    while current in self.joints and not current.endswith('Wrist_Lower_to_Gripper'):
                        j=self.joints[current];reach+=np.linalg.norm(j['origin'][:3,3]);current=j['parent']
                    if not current.endswith('Wrist_Lower_to_Gripper'):raise ValueError('夹爪链不完整')
                    self.grippers[current]=max(self.grippers.get(current,0),reach+(0. if self.nominal_geometry else .03))
                    continue
                # Dense samples on actual triangle surfaces, not an enclosing box
                # or convex hull: open spaces beside/between arms are preserved.
                areas=np.linalg.norm(np.cross(vertices[:,1]-vertices[:,0],vertices[:,2]-vertices[:,0]),axis=1)/2
                if not np.isfinite(areas).all() or areas.sum()<=0:raise ValueError('模型网格无效')
                count=min(500000,max(1000,int(areas.sum()/0.0025**2)))
                chosen=vertices[rng.choice(len(vertices),count,p=areas/areas.sum())]
                u=np.sqrt(rng.random((count,1)));v=rng.random((count,1))
                surface=(1-u)*chosen[:,0]+u*(1-v)*chosen[:,1]+u*v*chosen[:,2]
                verts=vertices.reshape(-1,3)
                if name in self.shapes:
                    surface=np.vstack((self.shapes[name][0].data,surface))
                    verts=np.vstack((self.mesh_hulls[name],verts))
                # Hull vertices preserve every directional mesh extent with far
                # less work than transforming all triangles at 50 Hz.
                self.mesh_hulls[name]=verts[ConvexHull(verts).vertices]
                self.shapes[name]=(cKDTree(surface),verts.min(axis=0),verts.max(axis=0))
        if len(self.shapes)<15 or len(self.grippers)!=2:raise ValueError('S3机身模型不完整')
        self.transforms={};self.radius=.48;self.top=1.85;self.filtered=0
        self.profile=None;self.profile_trees={};self.gripper_q={}
        self.radii=np.zeros(26);self.profile_error='夹爪自体模板尚未标定'
        self.timed_states=deque(maxlen=300)
        self.primitives=np.empty((0,7));self.primitive_names=[]

    def feedback(self,data,now=None,source_time=None):
        now=time.monotonic() if now is None else now
        try:
            timed_stamp=time.time() if source_time is None else float(source_time)
            if not math.isfinite(timed_stamp):raise ValueError('关节发送时间无效')
            if source_time is not None and not -.02<=time.time()-timed_stamp<=.1:
                raise ValueError('关节发送时间过期或时钟不一致')
            if source_time is not None and self.timed_states and timed_stamp<=self.timed_states[-1][0]:
                raise ValueError('关节发送时间未推进')
            q={};speed=[]
            stamp=float(data['timestamp'])
            if not math.isfinite(stamp):raise ValueError('关节时间无效')
            if self.producer is not None:
                delta=stamp-self.producer
                if delta<=0:raise ValueError('关节反馈时间未推进')
                # SDK timestamp has its own clock offset; validate progression
                # against receipt time, not absolute wall-clock age.
                if abs(delta-(now-self.producer_rx))>.2:raise ValueError('关节时间发生跳变')
            for group,names in GROUPS.items():
                g=data[group];values=g['position'];lost=g['communication_lost'];vel=g['speed']
                if len(values)!=len(names) or len(lost)!=len(names) or len(vel)!=len(names):raise ValueError('关节反馈不完整')
                if any(lost):raise ValueError('机械臂/腰腿关节通信中断')
                for name,value in zip(names,values):
                    value=float(value)
                    if not math.isfinite(value) or abs(value)>360:raise ValueError('关节角度无效')
                    q[name]=value
                speed.extend(vel)
            gripper_q={}
            for side in ('left','right'):
                g=data[side+'_gripper_state']
                if len(g['position'])!=1 or len(g['communication_lost'])!=1 or any(g['communication_lost']):raise ValueError('夹爪通信异常')
                value=float(g['position'][0])
                if not math.isfinite(value):raise ValueError('夹爪反馈无效')
                gripper_q[side]=value
            if not np.isfinite(speed).all():raise ValueError('关节速度无效')
            self.producer=stamp;self.producer_rx=now;self.last=now;self.q=q
            self.gripper_q=gripper_q
            self.timed_states.append((timed_stamp,q.copy()));self.source_timeline=source_time is not None
            self.history.append((now,q))
            # A validated geometric guard can distinguish bounded servo
            # vibration from displacement of the calibrated body. Without
            # that guard, retain the original conservative velocity gate.
            self.speed_peak=max(abs(float(v)) for v in speed)
            self.error='' if self.speed_peak<3. or callable(getattr(self,'pose_guard',None)) else '请保持双臂及腰腿静止'
        except (KeyError,TypeError,ValueError) as e:
            self.error=str(e);self.last=-1e9;self.history.clear();self.timed_states.clear()
            # Resynchronise once, but require a fresh stationary window afterwards.
            self.producer=None
        return not self.error

    def health(self,now=None):
        now=time.monotonic() if now is None else now
        if self.error:return False,self.error
        if now-self.last>.2:return False,'关节反馈过期'
        if getattr(self,'dynamic_geometry',False):
            if not getattr(self,'source_timeline',False):return False,'等待带发送时间的关节反馈'
            if len(self.timed_states)<2:return False,'等待关节时间线'
            guard=getattr(self,'pose_guard',None)
            if not callable(guard):return False,'缺少动态轮廓检查器'
            return guard(self)
        recent=[(t,q) for t,q in self.history if now-t<.7]
        if not recent or now-recent[0][0]<.45:return False,'等待双臂及腰腿停稳'
        vals=np.array([[q[n] for n in GROUPS['leg_waist_joint_state']+GROUPS['left_arm_joint_state']+GROUPS['right_arm_joint_state']+GROUPS['neck_joint_state']] for _,q in recent])
        ranges=np.ptp(vals,axis=0)
        if np.max(ranges[:-3])>.5:return False,'双臂或腰腿姿态正在变化'
        guard=getattr(self,'pose_guard',None)
        if callable(guard):
            ok,why=guard(self)
            if not ok:return False,why
            return True,'标定姿态的部件位移在允许范围内'
        if np.max(ranges[-3:])>.5:return False,'头部姿态正在变化，请保持标定姿态'
        return True,'双臂模型有效'

    def pose(self,link,q,cache):
        if link in cache:return cache[link]
        if link=='Link_Zero_Point':return np.eye(4)
        j=self.joints[link];t=self.pose(j['parent'],q,cache)@j['origin']
        if j['kind']!='fixed':
            if j['name'] not in q:raise ValueError('缺少关节：'+j['name'])
            axis=j['axis']/np.linalg.norm(j['axis']);m=np.eye(4)
            m[:3,:3]=Rotation.from_rotvec(axis*math.radians(q[j['name']])).as_matrix();t=t@m
        cache[link]=t;return t

    def at_scan(self,stamp):
        """Interpolate bounded receipt-clock samples. The vendor JSON stamp has
        an unverified absolute offset, so stationary posture remains mandatory.
        This is not a claim of hardware/PTP or per-point deskew synchronisation.
        """
        states=list(self.timed_states)
        if not states or stamp<states[0][0]-.03 or stamp>states[-1][0]+.03:raise ValueError('扫描时刻缺少匹配的关节反馈')
        for (a,qa),(b,qb) in zip(states,states[1:]):
            if a<=stamp<=b:
                if b-a>.06:raise ValueError('关节时间轴有缺口')
                t=(stamp-a)/max(1e-9,b-a)
                return {n:qa[n]+(qb[n]-qa[n])*t for n in qa}
        return (states[0] if stamp<states[0][0] else states[-1])[1].copy()

    def update(self,q=None):
        if self.q is None:raise ValueError('缺少关节反馈')
        q=self.q if q is None else q
        nominal=getattr(self,'nominal_geometry',False);padding=0. if nominal else .04
        cache={'Link_Zero_Point':np.eye(4)};radius=0. if nominal else .48;top=0. if nominal else 1.85
        # Factory XY footprint plus 4 cm. Z must come from the chassis mesh:
        # extruding the footprint to a guessed 40 cm falsely catches overhead.
        if 'Link_Zero_Point' not in self.shapes:raise ValueError('缺少底盘高度模型')
        primitives=[] if nominal else [[0,.02,0.,0.,.36,.29,0.]];names=[] if nominal else ['底盘']
        radii=np.zeros(26)
        if not nominal:radii[:4]=.47
        def band(low,high,r):
            # Each entry conservatively covers a whole 10 cm vertical bin.
            a=max(0,int(math.floor(low/.1)));b=min(len(radii),int(math.floor(high/.1))+1)
            if b>a:radii[a:b]=np.maximum(radii[a:b],r)
        for name,(_,low,high) in self.shapes.items():
            t=self.pose(name,q,cache)
            corners=np.array([[x,y,z] for x in [low[0],high[0]] for y in [low[1],high[1]] for z in [low[2],high[2]]])
            p=transform(getattr(self,'mesh_hulls',{}).get(name,corners) if nominal else corners,t);r=float(np.max(np.linalg.norm(p[:,:2],axis=1)))+padding
            radius=max(radius,r);top=max(top,float(p[:,2].max())+padding)
            band(p[:,2].min()-padding,p[:,2].max()+padding,r)
            lo=p.min(axis=0)-padding
            if not nominal:lo=np.floor(lo/.01)*.01
            hi=p.max(axis=0)+padding
            if not nominal:hi=np.ceil(hi/.01)*.01
            if name=='Link_Zero_Point' and not nominal:
                primitives[0][3]=float((lo[2]+hi[2])/2)
                primitives[0][6]=float((hi[2]-lo[2])/2)
            primitives.append([0,*((lo+hi)/2),*((hi-lo)/2)]);names.append(name)
        for name,r in self.grippers.items():
            t=self.pose(name,q,cache)
            if not getattr(self,'estimated_grippers',True):continue
            rho=float(np.linalg.norm(t[:2,3]))
            radius=max(radius,rho+r+padding);top=max(top,float(t[2,3])+r+padding)
            # Unknown DM finger configuration remains conservatively bounded
            # about its actual wrist, not about the centre of the chassis.
            primitives.append([1,*t[:3,3],r+padding,r+padding,r+padding]);names.append(name+'夹爪（URDF链长估算，DM未标定）')
            for i in range(len(radii)):
                dz=max(0.,i*.1-t[2,3],t[2,3]-(i+1)*.1)
                if dz<=r+padding:radii[i]=max(radii[i],rho+math.sqrt(max(0.,(r+padding)**2-dz**2)))
        if radius>1.3 or top>2.5:raise ValueError('当前姿态超出局部导航模型范围')
        self.transforms=cache;self.radius=radius if nominal else math.ceil(radius/.02)*.02;self.top=top if nominal else math.ceil(top/.02)*.02
        self.radii=radii if nominal else np.ceil(radii/.01)*.01
        self.primitives=np.asarray(primitives) if nominal or getattr(self,'dynamic_geometry',False) else merge_primitives(self.primitives,np.asarray(primitives))
        self.primitive_names=names
        return self.radius,self.top

    def filter(self,points):
        mask=np.zeros(len(points),dtype=bool)
        for name,(tree,low,high) in self.shapes.items():
            local=transform(points,inverse(self.transforms[name]))
            ids=np.flatnonzero((~mask)&np.all(local>=low-self.tolerance,axis=1)&np.all(local<=high+self.tolerance,axis=1))
            if len(ids):
                dist,_=tree.query(local[ids],distance_upper_bound=self.tolerance)
                mask[ids[np.isfinite(dist)]]=True
        self.filtered=int(mask.sum())
        return mask

    def load_profile(self,path,calibration_hash):
        self.profile=None;self.profile_trees={}
        try:
            with np.load(path,allow_pickle=False) as f:
                if str(f['model_hash'])!=self.model_hash or str(f['calibration_hash'])!=calibration_hash:raise ValueError('模型或雷达外参已变化')
                names=f['joint_names'].tolist();values=f['joint_values'].tolist()
                self.profile=dict(q=dict(zip(names,values)),gripper=dict(zip(('left','right'),f['gripper_values'].tolist())))
                for side in ('Left','Right'):
                    points=f[side]
                    if points.ndim!=2 or points.shape[1]!=3 or not 20<=len(points)<=30000 or not np.isfinite(points).all():raise ValueError('夹爪模板点云无效')
                    self.profile_trees['Link_'+side+'_Wrist_Lower_to_Gripper']=cKDTree(points)
            self.profile_error=''
        except (OSError,ValueError,KeyError) as e:
            self.profile=None;self.profile_trees={};self.profile_error='夹爪模板不可用：'+str(e)

    def profile_health(self):
        if self.profile is None:return False,self.profile_error
        if self.q is None:return False,'等待关节反馈'
        if set(self.profile['q'])!=set(self.q):return False,'模板关节集合不一致'
        guard=getattr(self,'pose_guard',None)
        if callable(guard):
            ok,why=guard(self)
            if not ok:return False,why
        if not getattr(self,'dynamic_geometry',False) and any(abs(self.q[n]-v)>.75 for n,v in self.profile['q'].items()
               if not (callable(guard) and n.startswith('Joint_Neck_'))):
            return False,'双臂/腰腿姿态已改变，请恢复标定姿态或重新采集模板'
        if any(abs(self.gripper_q.get(n,float('inf'))-v)>3 for n,v in self.profile['gripper'].items()):return False,'夹爪开口已改变，请重新标定自体模板'
        return True,'前伸空手姿态模板有效'

    def filter_with_profile(self,points,sensor_origin=None):
        mask=self.filter(points)
        if not self.profile_health()[0]:return mask
        for name,tree in self.profile_trees.items():
            local=transform(points,inverse(self.transforms[name]))
            # The measured gripper surface contains scanning/noise gaps; allow
            # 20 mm there, still narrower than the 40 mm collision padding.
            # This mask remains tied to the explicitly verified fixed empty pose.
            dist,_=tree.query(local,distance_upper_bound=.020)
            mask|=np.isfinite(dist)
        if sensor_origin is not None:
            # Ray-shadow test: a return beyond a measured robot surface on the
            # same beam is not evidence of free space. Drop the whole ray; do not
            # insert a far "free" endpoint. Never ray-clear behind removed returns.
            surface=np.concatenate([transform(tree.data,self.transforms[name]) for name,tree in self.profile_trees.items()])
            a=surface-np.asarray(sensor_origin);ar=np.linalg.norm(a,axis=1)
            a=a[ar>.05];ar=ar[ar>.05]
            dirs=a/ar[:,None];tree=cKDTree(dirs)
            b=points-np.asarray(sensor_origin);br=np.linalg.norm(b,axis=1)
            use=np.flatnonzero((~mask)&(br>.1))
            if len(use) and len(a)>=3:
                _,idx=tree.query(b[use]/br[use,None],k=3)
                rays=b[use]/br[use,None]
                ahead=np.einsum('nkj,nj->nk',a[idx],rays)
                perp=np.linalg.norm(a[idx]-ahead[:,:,None]*rays[:,None,:],axis=2)
                # Three distinct support points, narrow 6 mm ray band, and
                # an endpoint at least 2 cm behind the observed body surface.
                shadow=((perp<.006)&(ahead>.05)&(ahead<br[use,None]-.02)).sum(axis=1)>=3
                mask[use[shadow]]=True
        self.filtered=int(mask.sum());return mask

    def clearance(self,points,body):
        if not len(points):return 10.
        # Reuse the world-space tree over a complete scan. This bounds the
        # work during a 50 Hz swept-body check without dropping obstacle points.
        if getattr(self,'_clearance_points',None) is not points:
            self._clearance_points=points;self._clearance_tree=cKDTree(points)
        best=.3
        parts=getattr(self,'protected_parts',self.primitives)
        centers=transform(parts[:,1:4],body)
        radii=np.where(parts[:,0]==1,parts[:,4],np.linalg.norm(parts[:,4:],axis=1))
        candidates=self._clearance_tree.query_ball_point(centers,radii+best)
        local_points=transform(points,inverse(body))
        for part,ids in zip(parts,candidates):
            if ids:
                best=min(best,float(primitive_distances(local_points[ids],[part]).min()))
                if best<0:return best
        return best
