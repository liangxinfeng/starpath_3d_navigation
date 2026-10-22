import unittest
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from nav_math import transform,inverse
from nav_body_model import primitive_distances
from nav_link_self import LinkSelfFilter

def sample():
    def pose(name,q,cache):
        t=np.eye(4);t[:3,:3]=Rotation.from_euler('z',q['wrist'],degrees=True).as_matrix()
        cache[name]=t;return t
    m=SimpleNamespace(profile_trees={},primitives=np.empty((0,7)),pose=pose)
    def update(q):m.primitives=np.array([[0,0,0,0,.02,.02,.02]])
    m.update=update
    f=LinkSelfFilter.__new__(LinkSelfFilter)
    f.model=m;f.reference=SimpleNamespace(q={'wrist':0.},grippers=[180.,180.])
    f.parts=[('link',np.array([.39,-.01,1.19]),np.array([.41,.01,1.21]))]
    f.cells=[('link',np.array([.5,0,1.2]),np.eye(4))]
    f.extra_cells=[];f.q=None;f.transforms={};f.profile_enabled=False;f.gripper_trees={}
    return f

class LinkTests(unittest.TestCase):
    def test_motion_follows_link_and_releases_old_position(self):
        f=sample();f.prepare({'wrist':90.},{'left':180.,'right':180.})
        self.assertEqual(f.mask(np.array([[0,.4,1.2],[.4,0,1.2],[0,.5,1.2],[.5,0,1.2]])).tolist(),[True,False,True,False])
    def test_residual_cube_rotates_instead_of_world_aligned_growth(self):
        f=sample();f.prepare({'wrist':45.},{'left':180.,'right':180.})
        t=f.transforms['link'];p=transform([[.539,.039,1.239],[.541,0,1.2]],t)
        self.assertEqual(f.mask(p).tolist(),[True,False])
    def test_known_rear_table_never_removed(self):
        f=sample();table=np.array([[-.1448,-.1666,.8581],[-.1419,-.1654,.8539]])
        for a in (-90,0,5,90):
            f.prepare({'wrist':a},{'left':180.,'right':180.});self.assertFalse(f.mask(table).any())
    def test_unknown_gripper_opening_disables_measured_cells(self):
        f=sample();f.model.profile_trees={'link':None}
        f.prepare({'wrist':0.},{'left':184.,'right':180.})
        self.assertFalse(f.profile_enabled);self.assertFalse(f.mask(np.array([[.5,0,1.2]])).any())
    def test_missing_nonfinite_pose_rejected(self):
        f=sample()
        for q in ({},{'wrist':float('nan')},{'wrist':361.}):
            with self.assertRaises(ValueError):f.prepare(q,{'left':180.,'right':180.})
    def test_each_crop_is_inside_outer_collision_guard(self):
        f=sample()
        for a in (0,33,90):
            f.prepare({'wrist':a},{'left':180.,'right':180.})
            p=transform([[.539,.039,1.239]],f.transforms['link'])
            self.assertTrue(f.mask(p)[0]);self.assertLess(primitive_distances(p,f.collision_parts())[0],-.02)
    def test_collision_geometry_does_not_accumulate_old_postures(self):
        f=sample();f.prepare({'wrist':0.},{'left':180.,'right':180.});initial=f.collision_parts()
        for a in range(0,90,5):f.prepare({'wrist':a},{'left':180.,'right':180.});f.collision_parts()
        f.prepare({'wrist':0.},{'left':180.,'right':180.});np.testing.assert_allclose(initial,f.collision_parts())
    def test_no_filter_before_pose_prepared(self):
        f=sample()
        with self.assertRaises(ValueError):f.mask(np.zeros((1,3)))
    def test_broad_phase_matches_exact_geometry_including_far_points(self):
        f=sample();f.prepare({'wrist':33.},{'left':180.,'right':180.})
        rng=np.random.default_rng(345)
        near=np.column_stack([rng.uniform(.35,.55,3000),rng.uniform(-.06,.06,3000),rng.uniform(1.14,1.26,3000)])
        p=np.vstack([transform(near,f.transforms['link']),rng.uniform(-30,30,(4000,3))])
        expected=np.zeros(len(p),bool)
        for name,lo,hi in f.parts:
            local=transform(p,inverse(f.transforms[name]));expected|=np.all((local>=lo)&(local<=hi),axis=1)
        for name,center,ref_inv in f.cells:
            local=transform(p,inverse(f.transforms[name]@ref_inv));expected|=np.max(np.abs(local-center),axis=1)<=.04+1e-9
        np.testing.assert_array_equal(f.mask(p),expected)

if __name__=='__main__':unittest.main()
