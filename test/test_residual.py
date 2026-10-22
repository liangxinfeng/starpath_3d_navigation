import sys,tempfile,unittest,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from nav_residual import ResidualFilter,repeated_surface

class ResidualTests(unittest.TestCase):
    def setUp(self):
        self.dir=tempfile.TemporaryDirectory();self.root=Path(self.dir.name)
        self.profile=self.root/'gripper';self.profile.write_bytes(b'empty clear pose')
        self.points=np.array([[.36,.28,1.21],[.07,-.18,1.16],[.08,.18,1.42]])
        self.model=SimpleNamespace(q={str(i):0. for i in range(21)},gripper_q={'left':180.,'right':180.})
        self.data=dict(points=self.points,model_hash='model',calibration_hash='cal',
            profile_hash=hashlib.sha256(self.profile.read_bytes()).hexdigest(),
            authorised_empty_clear_body=True,frame='scan345_base',
            joint_names=list(self.model.q),joint_values=list(self.model.q.values()),
            gripper_values=[180.,180.])
        self.path=self.root/'residual.npz';self.save()
    def tearDown(self):self.dir.cleanup()
    def save(self):
        np.savez(self.path,**self.data)
        self.f=ResidualFilter(self.path,'model','cal',self.profile)
    def test_only_thin_calibrated_returns_removed(self):
        near=self.points+[0,0,.01];outside=self.points+[0,0,.03]
        points=np.vstack([self.points,near,outside,[[.35,0,.25],[.5,0,1.2],[0,1,1.2]]])
        self.assertEqual(self.f.mask(points,self.model).tolist(),[True]*6+[False]*6)
    def test_does_not_fill_space_between_calibrated_points(self):
        middle=(self.points[0]+self.points[1])/2
        self.assertFalse(self.f.mask(np.array([middle]),self.model)[0])
    def test_single_external_point_and_thin_pole_retained(self):
        p=np.array([[.4,.32,1.2],[.4,.32,1.21],[.4,.32,1.22],[.6,0,1.2]])
        self.assertFalse(self.f.mask(p,self.model).any())
    def test_pose_and_opening_change_fail_closed(self):
        self.model.q['0']=.36
        with self.assertRaises(ValueError):self.f.mask(self.points,self.model)
        self.model.q['0']=0;self.model.gripper_q['left']=184
        self.assertFalse(self.f.health(self.model)[0])
    def test_missing_corrupt_unapproved_and_wrong_frame_fail_closed(self):
        for key,value in [('model_hash','wrong'),('calibration_hash','wrong'),
                          ('authorised_empty_clear_body',False),('frame','airy_global'),
                          ('points',np.array([[1.,1.,1.]]))]:
            orig=self.data[key];self.data[key]=value;self.save()
            self.assertFalse(self.f.health(self.model)[0]);self.data[key]=orig
        self.path.unlink()
        self.assertFalse(ResidualFilter(self.path,'model','cal',self.profile).health(self.model)[0])
    def test_template_update_invalidates_residual(self):
        self.profile.write_bytes(b'changed')
        self.assertFalse(ResidualFilter(self.path,'model','cal',self.profile).health(self.model)[0])
    def test_calibration_requires_distinct_frames(self):
        same=np.tile(self.points[:1],(100,1))
        self.assertEqual(len(repeated_surface([same])),0)
        frames=[self.points+i*.0001 for i in range(3)]
        result=repeated_surface(frames)
        self.assertGreaterEqual(len(result),3)
    def test_empty_scan_safe(self):
        self.assertEqual(len(self.f.mask(np.empty((0,3)),self.model)),0)
    def test_radius_filter_only_calibrated_patch_and_persistent_return_rescued(self):
        base=np.array([[.7,.6,1.]])
        self.f.speckle_mask(base,np.zeros(1,bool),9.9,np.eye(4))
        self.f.speckle_mask(base,np.zeros(1,bool),10,np.eye(4))
        p=np.vstack([base,self.points[0]+[0,0,.025],[.6,0,1.2]])
        mask=self.f.speckle_mask(p,np.zeros(3,bool),10.1,np.eye(4))
        self.assertEqual(mask.tolist(),[False,True,False])
        self.f.speckle_mask(p,np.zeros(3,bool),10.2,np.eye(4))
        self.assertFalse(self.f.speckle_mask(p,np.zeros(3,bool),10.3,np.eye(4)).any())
    def test_thin_supported_structure_not_radius_filtered(self):
        self.f.speckle_mask(np.array([[.7,.6,1.]]),np.zeros(1,bool),10,np.eye(4))
        self.f.speckle_mask(np.array([[.7,.6,1.]]),np.zeros(1,bool),10.05,np.eye(4))
        p=np.array([self.points[0]+[0,0,d] for d in (.018,.025,.032)])
        self.assertFalse(self.f.speckle_mask(p,np.zeros(3,bool),10.1,np.eye(4)).any())
    def test_gap_and_clock_reversal_preserve_unknown_points(self):
        p=np.array([self.points[0]+[0,0,.025]])
        self.assertFalse(self.f.speckle_mask(p,np.zeros(1,bool),10,np.eye(4)).any())
        self.assertFalse(self.f.speckle_mask(p,np.zeros(1,bool),11,np.eye(4)).any())
        self.assertFalse(self.f.speckle_mask(p,np.zeros(1,bool),9,np.eye(4)).any())
    def test_world_persistence_survives_robot_motion(self):
        point=self.points[0]+[0,0,.025]
        self.f.speckle_mask(np.array([point]),np.zeros(1,bool),9.9,np.eye(4))
        self.f.speckle_mask(np.array([point]),np.zeros(1,bool),10,np.eye(4))
        body=np.eye(4);body[0,3]=.015
        self.assertFalse(self.f.speckle_mask(np.array([point-[.015,0,0]]),np.zeros(1,bool),10.1,body).any())

if __name__=='__main__':unittest.main()
