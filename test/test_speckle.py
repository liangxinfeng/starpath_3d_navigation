import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from nav_speckle import TransientSpeckleFilter

class SpeckleTests(unittest.TestCase):
    def test_isolated_flash_removed(self):
        f=TransientSpeckleFilter();f.filter(np.empty((0,3)),1.)
        self.assertFalse(f.filter([[1,2,3]],1.1)[0])
    def test_four_point_flash_removed(self):
        f=TransientSpeckleFilter();f.filter(np.empty((0,3)),1.)
        p=np.array([[0,0,1],[.01,0,1],[0,.01,1],[.01,.01,1]])
        self.assertFalse(f.filter(p,1.1).any())
        self.assertTrue(f.filter(p,1.2).all())
    def test_dense_new_obstacle_kept_immediately(self):
        f=TransientSpeckleFilter();f.filter(np.empty((0,3)),1.)
        p=np.array([[x,0,1] for x in np.arange(6)*.01])
        self.assertTrue(f.filter(p,1.1).all())
    def test_thin_persistent_obstacle_survives(self):
        f=TransientSpeckleFilter();f.filter([[1,0,1]],1.)
        self.assertTrue(f.filter([[1.01,.005,1]],1.1)[0])
    def test_body_motion_does_not_shift_stationary_obstacle(self):
        f=TransientSpeckleFilter();p=np.array([[1,0,.8]])
        f.filter(p,1.)
        body_frame=p-[.1,0,0];odom_frame=body_frame+[.1,0,0]
        self.assertTrue(f.filter(odom_frame,1.1)[0])
    def test_different_speckles_do_not_confirm(self):
        f=TransientSpeckleFilter();f.filter([[0,0,1]],1.)
        self.assertFalse(f.filter([[.1,0,1]],1.1)[0])
    def test_no_historical_obstacles(self):
        f=TransientSpeckleFilter();f.filter([[0,0,1]],1.)
        self.assertEqual(len(f.filter(np.empty((0,3)),1.1)),0)
        self.assertFalse(f.filter([[0,0,1]],1.2)[0])
    def test_warmup_and_gap_fail_closed(self):
        f=TransientSpeckleFilter()
        self.assertTrue(f.filter([[0,0,1]],1.)[0])
        self.assertTrue(f.filter([[2,0,1]],1.5)[0])
        self.assertTrue(f.filter([[4,0,1]],1.5)[0])
        self.assertTrue(f.filter([[6,0,1]],1.4)[0])
    def test_invalid_input_rejected(self):
        f=TransientSpeckleFilter()
        with self.assertRaises(ValueError):f.filter([[float('nan'),0,0]],1.)
    def test_noise_and_wall_mixture(self):
        f=TransientSpeckleFilter();f.filter([[3,0,1]],1.)
        wall=np.array([[0,y,z] for y in np.arange(0,.5,.025) for z in np.arange(0,.5,.025)])
        points=np.vstack([wall,[[3,0,1],[5,0,1]]]);mask=f.filter(points,1.1)
        self.assertTrue(mask[:-1].all());self.assertFalse(mask[-1])

if __name__=='__main__':unittest.main()
