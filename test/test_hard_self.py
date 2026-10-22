import sys,tempfile,hashlib,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from nav_hard_self import HardSelfFilter,cell_guards,protected_union
from nav_body_model import primitive_distances

class HardTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.profile=self.root/'gripper';self.profile.write_bytes(b'calibrated-empty')
        self.model=SimpleNamespace(model_hash='m',q={str(i):0. for i in range(21)},gripper_q={'left':180.,'right':180.},
            shapes={'torso':(None,np.array([-.05,-.05,.95]),np.array([.05,.05,1.05]))},
            pose=lambda name,q,cache:np.eye(4))
        self.data=dict(cells=np.array([[18,14,60],[3,-10,57]]),mode='hard_voxel_v3',frame='scan345_base',
            model_hash='m',calibration_hash='c',profile_hash=hashlib.sha256(self.profile.read_bytes()).hexdigest(),
            authorised_empty_clear_body=True,joint_names=list(self.model.q),joint_values=list(self.model.q.values()),gripper_values=[180.,180.])
        self.path=self.root/'hard.npz';self.load()
    def load(self):
        np.savez(self.path,**self.data);self.f=HardSelfFilter(self.path,self.model,'c',self.profile)
    def tearDown(self):self.tmp.cleanup()
    def test_model_interior_and_exact_twenty_mm_padding(self):
        p=np.array([[0,0,1.],[.069,0,1],[.071,0,1],[0,.071,1],[0,0,.929]])
        self.assertEqual(self.f.mask(p,self.model).tolist(),[True,True,False,False,False])
    def test_measured_cells_are_deterministic_and_do_not_fill_gaps(self):
        p=np.array([[.37,.29,1.21],[.409,.29,1.21],[.411,.29,1.21],[.2,0,1.2]])
        for _ in range(8):self.assertEqual(self.f.mask(p,self.model).tolist(),[True,True,False,False])
    def test_single_external_point_and_vertical_pole_preserved(self):
        p=np.array([[.44,.29,z] for z in (1.1,1.15,1.2,1.25)]+[[0,.6,1.2],[.3,0,.2]])
        self.assertFalse(self.f.mask(p,self.model).any())
    def test_every_rejection_cell_has_an_outer_collision_guard(self):
        centers=(self.data['cells']+.5)*.02
        for delta in np.eye(3)*.049:
            for sign in (-1,1):
                p=centers+sign*delta
                self.assertFalse(self.f.mask(p,self.model).any())
                self.assertTrue(np.all(primitive_distances(p,self.f.extra_parts)<0))
    def test_pose_opening_missing_and_hash_mismatch_fail_closed(self):
        self.model.q['0']=.36;self.assertFalse(self.f.health(self.model)[0])
        self.model.q['0']=0;self.model.gripper_q['left']=184;self.assertFalse(self.f.health(self.model)[0])
        self.model.gripper_q['left']=180;self.data['calibration_hash']='wrong';self.load();self.assertFalse(self.f.health(self.model)[0])
        self.path.unlink();self.assertFalse(HardSelfFilter(self.path,self.model,'c',self.profile).health(self.model)[0])
    def test_geometric_pose_displacement_guard(self):
        def pose(name,q,cache):
            t=np.eye(4);t[0,3]=q['0']*.1;return t
        self.model.pose=pose;self.load();self.model.q['0']=.1
        self.assertFalse(self.f.health(self.model)[0])
    def test_invalid_or_overlarge_crops_rejected(self):
        for cells in (np.array([[100,100,100]]),np.array([[18.1,14,60]]),np.zeros((0,3))):
            self.data['cells']=cells;self.load();self.assertFalse(self.f.health(self.model)[0])
    def test_floor_and_far_overhead_are_never_hard_rejected(self):
        p=np.array([[0,0,0],[.4,.3,0],[.4,.3,2.4],[1.5,0,1.]])
        self.assertFalse(self.f.mask(p,self.model).any())
    def test_guard_pruning_preserves_gap_between_parts(self):
        core=np.array([[0,-.1,0,0,.03,.2,.2],[0,.1,0,0,.03,.2,.2]])
        # The corners lie inside different boxes, but the middle is a gap:
        # dropping this guard would remove collision coverage.
        extra=np.array([[0,0,0,0,.1,.02,.02],[0,-.1,0,0,.01,.01,.01]])
        protected=protected_union(core,extra)
        self.assertEqual(len(protected),3)
        p=np.array([[0,0,0],[-.1,0,0],[.1,0,0]])
        np.testing.assert_allclose(primitive_distances(p,protected),primitive_distances(p,np.vstack([core,extra])))
    def test_real_rear_table_coordinates_never_cropped(self):
        table=np.array([[-.1419,-.1654,.8539],[-.128,-.1604,.834]])
        self.assertFalse(self.f.mask(table,self.model).any())
    def test_head_jitter_uses_displacement_instead_of_universal_angle(self):
        self.model.q['Joint_Neck_Pitch']=self.model.q.pop('0')
        self.data['joint_names']=list(self.model.q);self.data['joint_values']=list(self.model.q.values())
        def pose(name,q,cache):
            t=np.eye(4);t[0,3]=q['Joint_Neck_Pitch']*.005;return t
        self.model.pose=pose;self.load()
        self.model.q['Joint_Neck_Pitch']=.9
        self.assertTrue(self.f.health(self.model)[0])
        self.model.q['Joint_Neck_Pitch']=2.
        self.assertFalse(self.f.health(self.model)[0])
    def test_measured_gripper_tip_displacement_is_checked(self):
        from scipy.spatial import cKDTree
        self.model.profile_trees={'finger':cKDTree([[.1,0,0],[.5,0,0]])}
        def pose(name,q,cache):
            t=np.eye(4)
            if name=='finger':t[0,3]=q['0']*.1
            return t
        self.model.pose=pose;self.load();self.model.q['0']=.1
        self.assertFalse(self.f.health(self.model)[0])

if __name__=='__main__':unittest.main()
