import sys,unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from nav_math import planar_velocity

class VelocityTests(unittest.TestCase):
    def test_stationary_odom_stays_stationary_under_map_rotation(self):
        body=np.eye(4)
        for angle in (0,.4,-.5):
            np.testing.assert_array_equal(planar_velocity(body,body,Rotation.from_euler('z',angle).as_matrix(),.1),[0,0,0])
    def test_continuous_velocity_rotates_to_map_without_vertical_motion(self):
        a=np.eye(4);b=a.copy();b[:3,3]=[.01,0,.02]
        r=Rotation.from_euler('z',np.pi/2).as_matrix()
        np.testing.assert_allclose(planar_velocity(a,b,r,.1),[0,.1,0],atol=1e-10)
        np.testing.assert_array_equal(planar_velocity(a,b,r,0),[0,0,0])
