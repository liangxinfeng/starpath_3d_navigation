import unittest,math
from types import SimpleNamespace as S
import numpy as np
from scipy.spatial.transform import Rotation
from nav_chassis import TrackingFrame,ChassisTrajectory,advance,at_global_goal
from nav_math import inverse,transform,yaw
from nav_body_model import primitive_distances
from nav_runtime import Runtime

def mat(x=0,y=0,a=0):
    t=np.eye(4);t[:3,3]=[x,y,0];t[:3,:3]=Rotation.from_euler('z',a).as_matrix();return t
def pose(t):
    q=Rotation.from_matrix(t[:3,:3]).as_quat()
    return S(position=S(x=t[0,3],y=t[1,3],z=t[2,3]),orientation=S(x=q[0],y=q[1],z=q[2],w=q[3]))

class TrackingFrameTests(unittest.TestCase):
    def test_map_correction_does_not_change_feedback_or_command(self):
        anchor=mat(3,4,.3);frame=TrackingFrame(anchor);local=mat(.1,0,.1)
        original=anchor@local;changed=mat(3.22,3.92,.34)
        np.testing.assert_allclose(frame.body(changed@local,changed),original,atol=1e-12)
        path=ChassisTrajectory([[0,0,0],[1,0,0]],[.1],[0],[10],True,0)
        f=TrackingFrame(np.eye(4));old=mat(.3,0,0)
        a=path.control(np.array([.3,0,0]),3,.02)
        b=f.body(changed@old,changed);out=path.control(np.array([b[0,3],b[1,3],yaw(b)]),3,.02)
        np.testing.assert_allclose(a,out,atol=1e-12)
    def test_actual_local_departure_still_stops(self):
        f=TrackingFrame(np.eye(4));c=mat(.2,0,.05);body=f.body(c@mat(0,.20),c)
        path=ChassisTrajectory([[0,0,0],[1,0,0]],[.1],[0],[10],True,0)
        with self.assertRaises(ValueError):path.control(np.array([body[0,3],body[1,3],yaw(body)]),0,.02)
    def test_display_tracks_anchor_but_global_goal_does_not_move(self):
        f=TrackingFrame(np.eye(4));p,a=f.display(np.array([[0,0,0],[1,0,0]]),0,mat(.2,0,.1))
        np.testing.assert_allclose(p[0],[.2,0,0]);self.assertAlmostEqual(a[0],.1)
        self.assertFalse(at_global_goal(mat(1.2,0),dict(point=[1,0,0],yaw=None)))
        self.assertTrue(at_global_goal(mat(1.02,0),dict(point=[1,0,0],yaw=None)))
        self.assertFalse(at_global_goal(mat(1,0,.2),dict(point=[1,0,0],yaw=0)))
    def test_turn_clock_not_frozen_by_uncorrectable_xy_error(self):
        path=ChassisTrajectory([[0,0,0],[0,0,1.2]],[0],[.24],[5],True,0)
        v,w,t,_=path.control(np.array([-.09,.04,0]),0,.02)
        self.assertEqual(v,0);self.assertAlmostEqual(w,.24);self.assertAlmostEqual(t,.02)
    def test_reference_waits_for_lagging_turn_before_error_limit(self):
        path=ChassisTrajectory([[0,0,0],[0,0,1.2]],[0],[.24],[5],True,0)
        t=0.
        for _ in range(1000):_,_,t,_=path.control(np.zeros(3),t,.02)
        self.assertLess(path.reference(t)[0][2],.21)
    def test_cached_cloud_and_robot_reframed_together(self):
        points=np.array([[.5,0,1],[.1,.1,1]])
        r=S(correction=np.eye(4),body=mat(.1,0),points=points.copy(),trajectory=None,tracking_frame=None,global_stamp=0.,safety=S(local=0),odoms={1:S(pose=S(pose=pose(np.eye(4))))})
        r.abort=lambda *args:None
        before=transform(r.points,inverse(r.body));new=mat(.22,-.03,.03)
        Runtime.update_correction(r,1,S(pose=pose(new),header=S(stamp=S(sec=1,nanosec=0))))
        np.testing.assert_allclose(transform(r.points,inverse(r.body)),before,atol=1e-12)
        np.testing.assert_allclose(r.correction,new,atol=1e-12)
    def test_simulation_global_jumps_do_not_make_velocity_sawtooth(self):
        traj=ChassisTrajectory([[0,0,0],advance([0,0,0],.12,.2,5)],[.12],[.2],[5],True,0)
        frame=TrackingFrame(np.eye(4));p=np.zeros(3);t=0
        for i in range(300):
            c=mat(.2 if i%30<15 else -.08,.03, .03 if i%30<15 else 0)
            body=frame.body(c@mat(*p),c);current=np.array([body[0,3],body[1,3],yaw(body)])
            actual=traj.control(current,t,.02);expected=traj.control(p,t,.02)
            np.testing.assert_allclose(actual,expected,atol=1e-10)
            v,w,t,done=actual;p=advance(p,v,w,.02)
            if done:break
        self.assertTrue(done)

if __name__=='__main__':unittest.main()
