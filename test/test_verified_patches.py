import unittest
from types import SimpleNamespace
import numpy as np
from scipy.spatial import cKDTree
from nav_math import transform
from nav_body_model import primitive_distances
from test_link_self import sample

HEAD='Link_Neck_Yaw_to_Head'
HAND='Link_Right_Wrist_Lower_to_Gripper'

class VerifiedPatchTests(unittest.TestCase):
    def make(self):
        f=sample();f.parts=[];f.cells=[]
        f.model.shapes={n:(cKDTree([[.3,0,1.2]]),np.zeros(3),np.ones(3)) for n in [HEAD,HAND]}
        return f
    def test_follows_link_and_releases_old_patch(self):
        f=self.make();f.add_verified_link_patches([[.35,0,1.2]],[HEAD],{'wrist':0.})
        f.prepare({'wrist':90.},{'left':180.,'right':180.})
        self.assertEqual(f.mask(np.array([[0,.35,1.2],[.35,0,1.2],[0,.39,1.2]])).tolist(),[True,False,False])
    def test_mask_small_and_guard_contains_corners(self):
        f=self.make();f.add_verified_link_patches([[.35,0,1.2]],[HEAD],{'wrist':0.})
        f.prepare({'wrist':33.},{'left':180.,'right':180.})
        p=transform([[.369,.019,1.219],[.386,0,1.2]],f.transforms[HEAD])
        self.assertEqual(f.mask(p).tolist(),[True,False])
        self.assertLess(primitive_distances(p[:1],f.collision_parts())[0],-.02)
    def test_real_table_and_thin_adjacent_object_preserved(self):
        f=self.make();f.add_verified_link_patches([[.35,0,1.2]],[HEAD],{'wrist':0.})
        f.prepare({'wrist':0.},{'left':180.,'right':180.})
        p=np.array([[-.1448,-.1666,.8581],[.4,0,1.2],[.4,0,1.21],[.4,0,1.22]])
        self.assertFalse(f.mask(p).any())
    def test_invalid_scope_is_atomic(self):
        for pts,links,q in [([[.35,0,1.2]],['other'],{'wrist':0.}),([[.8,0,1.2]],[HEAD],{'wrist':0.}),([[.35,0,1.2]],[HEAD],{}),([[.35,0,1.2],[.35,.08,1.2]],[HEAD,HEAD],{'wrist':0.}),([[float('nan'),0,1.2]],[HEAD],{'wrist':0.})]:
            f=self.make()
            with self.assertRaises(ValueError):f.add_verified_link_patches(pts,links,q)
            self.assertEqual(f.extra_cells,[])
    def test_gripper_opening_cannot_reuse_unknown_patch(self):
        f=self.make();f.model.profile_trees={HAND:None}
        f.add_verified_link_patches([[.35,0,1.2]],[HAND],{'wrist':0.})
        f.prepare({'wrist':0.},{'left':180.,'right':184.})
        self.assertFalse(f.mask(np.array([[.35,0,1.2]])).any())
    def test_scope_does_not_grow_with_repeated_prepare(self):
        f=self.make();f.add_verified_link_patches([[.35,0,1.2]],[HEAD],{'wrist':0.})
        f.prepare({'wrist':0.},{'left':180.,'right':180.});original=f.collision_parts()
        for angle in range(-90,90,5):f.prepare({'wrist':angle},{'left':180.,'right':180.})
        f.prepare({'wrist':0.},{'left':180.,'right':180.})
        np.testing.assert_allclose(original,f.collision_parts())
    def test_existing_gripper_calibration_gets_same_4cm_tolerance_as_residual_cells(self):
        f=self.make();f.extra_cells=[(HAND,np.array([.35,0,1.2]),np.eye(4))]
        f.prepare({'wrist':0.},{'left':180.,'right':180.})
        p=np.array([[.389,0,1.2],[.391,0,1.2]])
        self.assertEqual(f.mask(p).tolist(),[True,False])
        self.assertLess(primitive_distances(p[:1],f.collision_parts())[0],-.02)

if __name__=='__main__':unittest.main()
