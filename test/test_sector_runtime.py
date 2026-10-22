import unittest,time
from types import SimpleNamespace as S
import numpy as np
from nav_runtime import Runtime
from airy_fixed_sector import FixedSectorFilter

class SectorRuntimeTests(unittest.TestCase):
    def fixture(self):
        sector=FixedSectorFilter(dict(origin_x=-.08,origin_y=0,radius=.65,half_angle_deg=100,z_min=.35,z_max=1.9))
        def unexpected():raise AssertionError('Old gripper template must not be queried')
        return S(model=S(health=lambda:(True,''),profile_health=unexpected),residual=None,sector=sector,dynamic_geometry=True,sensor_filter={'filter_mode':'fixed_sector_volume','sector_hash':sector.digest,'last_valid_scan_age':.1},sensor_filter_time=time.monotonic(),handoff=S(health=lambda:(True,'')),safety=S(health=lambda:(True,'')))
    def test_no_dependency_on_old_gripper_profile(self):self.assertTrue(Runtime.health(self.fixture())[0])
    def test_legacy_upstream_rejected(self):
        h=self.fixture();h.sensor_filter['filter_mode']='dynamic_link_volumes';self.assertFalse(Runtime.health(h)[0])
    def test_mismatched_sector_rejected(self):
        h=self.fixture();h.sensor_filter['sector_hash']='different';self.assertFalse(Runtime.health(h)[0])
    def test_stale_scan_or_feedback_still_stops(self):
        h=self.fixture();h.sensor_filter_time-=3;self.assertFalse(Runtime.health(h)[0])
        h=self.fixture();h.model.health=lambda:(False,'feedback missing');self.assertFalse(Runtime.health(h)[0])
    def test_joint_changes_do_not_change_filter(self):
        h=self.fixture();points=np.array([[.3,.1,1],[.8,.1,1]])
        expected=h.sector.mask(points).copy()
        for q in [0,30,-40]:
            h.model.q={'Joint_Waist_Pitch':q};np.testing.assert_array_equal(h.sector.mask(points),expected)

if __name__=='__main__':unittest.main()
