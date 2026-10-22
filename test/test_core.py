import sys,unittest,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from nav_math import *
from nav_safety import Safety

class CoreTests(unittest.TestCase):
    def test_extrinsic_roundtrip(self):
        calibration=dict(extrinsic_R=Rotation.from_euler('xyz',[.1,.2,3.]).as_matrix().ravel(),extrinsic_T=[.004,.005,-.004])
        base_imu,imu_lidar=extrinsics([.2634,0,.2],[0,.436332,0],calibration)
        base_lidar=base_imu@imu_lidar
        np.testing.assert_allclose(base_lidar[:3,3],[.2634,0,.2],atol=1e-8)
        global_base=np.eye(4);global_base[:3,:3]=Rotation.from_euler('z',1.2).as_matrix();global_base[:3,3]=[4,-2,0]
        global_imu=global_base@base_imu
        np.testing.assert_allclose(global_imu@inverse(base_imu),global_base,atol=1e-8)
    def test_timestamp_prediction_transform(self):
        correction=np.eye(4);correction[:3,:3]=Rotation.from_euler('z',.6).as_matrix();correction[:3,3]=[3,4,0]
        old=np.eye(4);old[:3,3]=[1,2,0];new=np.eye(4);new[:3,3]=[1.3,2,0]
        recovered=(correction@old)@inverse(old)
        np.testing.assert_allclose(recovered@new,correction@new)
    def test_spline_valid_and_invalid(self):
        t=Trajectory([[0,0,0],[.3,0,0],[.7,0,0],[1,0,0]],[0,0,0,0,10,10,10,10],3)
        np.testing.assert_allclose(t.at(0),[0,0,0]);np.testing.assert_allclose(t.at(10),[1,0,0])
        for knots in [[0,1],[0,0,0,0,float('nan'),10,10,10],[0,0,0,0,-1,10,10,10]]:
            with self.assertRaises(ValueError):Trajectory([[0,0,0]]*4,knots,3)
    def test_collision_floor_overhead_and_body(self):
        body=np.eye(4)
        self.assertEqual(clearance(np.array([[0,0,0],[0,0,2.2]]),body),10.)
        self.assertLess(clearance(np.array([[.2,0,1.2]]),body),0.)
        self.assertGreater(clearance(np.array([[2,0,1.2]]),body),1.)
    def healthy(self):
        now=[10.];s=Safety(lambda:now[0]);s.locked=True
        s.ui=s.odom=s.cloud=s.local=s.planner=10.
        return now,s
    def test_explicit_arm_required_and_ui_loss_latches(self):
        now,s=self.healthy();self.assertFalse(s.check());s.arm();self.assertTrue(s.check())
        now[0]+=.31;self.assertFalse(s.check());s.ui=now[0];self.assertFalse(s.check())
    def test_each_input_loss_stops(self):
        for field in ['odom','cloud','local','planner']:
            now,s=self.healthy();s.arm();setattr(s,field,0.);self.assertFalse(s.check(),field)
    def test_localization_loss_stops(self):
        now,s=self.healthy();s.arm();s.locked=False;self.assertFalse(s.check())
    def test_nonfinite_rejected_and_no_reverse(self):
        self.assertEqual(Safety.bounded(-1.,20.),(0.,.3))
        with self.assertRaises(ValueError):Safety.bounded(float('nan'),0.)
if __name__=='__main__':unittest.main()
