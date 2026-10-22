import sys,unittest,time
from pathlib import Path
from collections import deque
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from nav_body_model import BodyModel,GROUPS,merge_primitives,swept_poses
from scipy.spatial.transform import Rotation

def bare():
    m=BodyModel.__new__(BodyModel)
    m.tolerance=.012;m.q=None;m.history=deque(maxlen=120);m.timed_states=deque(maxlen=300)
    m.last=-1e9;m.producer=None;m.producer_rx=None;m.error='';m.gripper_q={}
    m.profile=None;m.profile_error='missing';m.profile_trees={};m.shapes={};m.transforms={}
    m.radii=np.zeros(26);m.radii[:4]=.47;m.radii[9:13]=.7
    m.primitives=np.array([[0,.02,0,.22,.36,.29,.18],[0,.1,0,.9,.16,.20,.5],[1,.45,.2,1.05,.15,.15,.15]],float)
    return m

def feedback(stamp):
    d={'timestamp':stamp}
    for k,names in GROUPS.items():d[k]=dict(position=[0.]*len(names),speed=[0.]*len(names),communication_lost=[False]*len(names))
    for s in ('left','right'):d[s+'_gripper_state']=dict(position=[180.],communication_lost=[False])
    return d

class BodyTests(unittest.TestCase):
    def test_vendor_degrees_not_gripper_radians(self):
        m=bare();d=feedback(100);d['left_arm_joint_state']['position'][3]=-110
        self.assertTrue(m.feedback(d,1))
        self.assertEqual(m.q['Joint_Left_Elbow'],-110)
        self.assertNotIn('Joint_Left_Gripper',m.q)
        self.assertEqual(m.gripper_q['left'],180)
    def test_comm_failure_and_replay_close_gate(self):
        m=bare();self.assertTrue(m.feedback(feedback(100),1))
        self.assertFalse(m.feedback(feedback(100),1.01))
        d=feedback(101);d['left_arm_joint_state']['communication_lost'][0]=True
        self.assertFalse(m.feedback(d,2));self.assertFalse(m.health(2)[0])
    def test_stationary_window_and_feedback_timeout(self):
        m=bare()
        for i in range(60):self.assertTrue(m.feedback(feedback(100+i*.01),1+i*.01))
        self.assertTrue(m.health(1.59)[0]);self.assertFalse(m.health(1.9)[0])
    def test_startup_neck_servo_oscillation_is_not_arm_motion(self):
        m=bare();m.pose_guard=lambda model:(True,'geometry verified')
        for i in range(80):
            d=feedback(100+i*.01)
            d['neck_joint_state']['position'][1]=.08 if i%2 else -.08
            d['neck_joint_state']['speed'][1]=5.5 if i%2 else -5.5
            self.assertTrue(m.feedback(d,1+i*.01))
        self.assertTrue(m.health(1.79)[0])
    def test_arm_waist_speed_protection_remains_immediate(self):
        for group in ('leg_waist_joint_state','left_arm_joint_state','right_arm_joint_state'):
            m=bare();d=feedback(100);d[group]['speed'][0]=3.01
            self.assertFalse(m.feedback(d,1))
            self.assertEqual(m.health(1)[1],'请保持双臂及腰腿静止')
    def test_real_head_motion_and_invalid_speed_still_block(self):
        m=bare()
        for i in range(80):
            d=feedback(100+i*.01);d['neck_joint_state']['position'][1]=i*.01
            m.feedback(d,1+i*.01)
        self.assertIn('头部姿态',m.health(1.79)[1])
        d=feedback(101);d['neck_joint_state']['speed'][1]=float('nan')
        self.assertFalse(m.feedback(d,2))
    def test_precise_mode_requires_geometry_guard_to_pass(self):
        m=bare();m.pose_guard=lambda model:(False,'geometric displacement exceeded')
        for i in range(80):
            d=feedback(100+i*.01);d['left_arm_joint_state']['speed'][0]=7
            m.feedback(d,1+i*.01)
        self.assertEqual(m.health(1.79),(False,'geometric displacement exceeded'))
    def test_template_dies_on_pose_or_opening_change(self):
        m=bare();m.feedback(feedback(100),1)
        m.profile=dict(q=m.q.copy(),gripper=m.gripper_q.copy())
        self.assertTrue(m.profile_health()[0])
        m.q['Joint_Left_Elbow']=1.
        self.assertFalse(m.profile_health()[0]);m.q['Joint_Left_Elbow']=0.
        m.gripper_q['right']=170.;self.assertFalse(m.profile_health()[0])
    def test_surface_only_preserves_gap_and_near_external_points(self):
        m=bare();m.feedback(feedback(100),1);m.profile=dict(q=m.q.copy(),gripper=m.gripper_q.copy())
        m.profile_trees={'wrist':cKDTree([[.4,.2,1.05]])};m.transforms={'wrist':np.eye(4)}
        pts=np.array([[.4,.2,1.05],[.4,.2,1.06],[.4,.2,1.09],[.4,0,1.05],[.3,.38,.4]])
        self.assertEqual(m.filter_with_profile(pts).tolist(),[True,True,False,False,False])
    def test_current_shape_keeps_side_space_and_overhead(self):
        m=bare();p=np.array([[.55,0,1.05]])
        self.assertGreater(m.clearance(p,np.eye(4)),0)
        self.assertGreater(m.clearance(np.array([[0,.624,.92]]),np.eye(4)),.1)
        self.assertGreater(m.clearance(np.array([[0,0,2.5]]),np.eye(4)),.1)
        self.assertLess(m.clearance(np.array([[.37,0,.2]]),np.eye(4)),0)
        self.assertLess(m.clearance(np.array([[.45,.2,1.05]]),np.eye(4)),0)
    def test_rectangular_chassis_corner_and_side_clearance(self):
        m=bare()
        self.assertGreater(m.clearance(np.array([[0,.32,.2]]),np.eye(4)),.02)
        self.assertLess(m.clearance(np.array([[.36,.28,.2]]),np.eye(4)),0)
    def test_chassis_height_comes_from_mesh_not_fixed_extrusion(self):
        m=bare();m.q={};m.joints={};m.grippers={};m.primitives=np.empty((0,7))
        m.shapes={'Link_Zero_Point':(None,np.array([-.30,-.25,-.02]),np.array([.34,.25,.33]))}
        m.update()
        self.assertAlmostEqual(m.primitives[0,3]+m.primitives[0,6],.37)
        p=np.array([[.317,.347,.401]]);end=np.eye(4)
        end[:3,:3]=Rotation.from_euler('z',.3).as_matrix()
        self.assertGreater(min(m.clearance(p,t) for t in swept_poses(np.eye(4),end)),.02)
        # A real low obstacle still blocks a turn at the same horizontal spot.
        p[0,2]=.2
        self.assertLess(min(m.clearance(p,t) for t in swept_poses(np.eye(4),end)),0)
    def test_turn_sweep_catches_intermediate_arm_collision(self):
        m=bare();start=np.eye(4);end=np.eye(4)
        end[:3,:3]=Rotation.from_euler('z',np.pi/2).as_matrix()
        point=Rotation.from_euler('z',np.pi/4).apply([[.45,.2,1.05]])
        self.assertGreater(m.clearance(point,start),0)
        self.assertGreater(m.clearance(point,end),0)
        self.assertLess(min(m.clearance(point,t) for t in swept_poses(start,end)),0)
    def test_geometry_union_does_not_shrink_or_creep(self):
        a=bare().primitives;b=a.copy();b[0,1]+=.004;b[2,2]+=.002
        c=merge_primitives(a,b)
        for _ in range(50):np.testing.assert_allclose(merge_primitives(c,b),c,atol=1e-12)
        self.assertLessEqual(c[0,1]-c[0,4],a[0,1]-a[0,4]+1e-9)
        self.assertGreaterEqual(c[0,1]+c[0,4],b[0,1]+b[0,4]-1e-9)
    def test_shadow_requires_same_ray_and_keeps_front_obstacle(self):
        m=bare();m.feedback(feedback(100),1);m.profile=dict(q=m.q.copy(),gripper=m.gripper_q.copy())
        m.profile_trees={'wrist':cKDTree([[1,0,0],[1,.003,0],[1,-.003,0]])};m.transforms={'wrist':np.eye(4)}
        pts=np.array([[2.,0,0],[.8,0,0],[2.,.05,0],[2.,0,.05]])
        self.assertEqual(m.filter_with_profile(pts,[0,0,0]).tolist(),[True,False,False,False])
    def test_time_interpolation_and_missing_history(self):
        m=bare();m.timed_states.extend([(10.,{'a':1.}),(10.02,{'a':3.})])
        self.assertAlmostEqual(m.at_scan(10.01)['a'],2.)
        with self.assertRaises(ValueError):m.at_scan(10.1)
        m.timed_states=deque([(10.,{'a':1.}),(10.2,{'a':3.})])
        with self.assertRaises(ValueError):m.at_scan(10.1)

if __name__=='__main__':unittest.main()
