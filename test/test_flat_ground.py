import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from nav_ground import low_return_outliers,supported_low_geometry
from nav_static_map import StaticMap

class FlatGroundTests(unittest.TestCase):
    def test_sparse_surrounding_floor_interpolates(self):
        ground=np.array([[-.07,0,0],[.08,.20,0],[.08,-.20,0]])
        p=np.array([[0,0,-.2]])
        self.assertTrue(low_return_outliers(p,p[:,2],ground).all())
    def test_large_unobserved_gap_not_filled(self):
        ground=np.array([[-.07,0,0],[.2,.40,0],[.2,-.40,0]])
        p=np.array([[0,0,-.2]])
        self.assertFalse(low_return_outliers(p,p[:,2],ground).any())
    def test_recorded_three_speckles_are_not_a_surface(self):
        p=np.array([[-.58246356,-1.62888718,-.2152],[-.64010566,-1.65108669,-.1916],[-.68117654,-1.69790220,-.1759]])
        self.assertFalse(supported_low_geometry(p[:,:2]))
        self.assertTrue(low_return_outliers(p,p[:,2],self.ground).all())
    def test_small_but_spatially_supported_depression(self):
        p=np.array([[0,0],[.1,0],[.1,.1],[0,.1]])
        self.assertTrue(supported_low_geometry(p))
    def test_long_narrow_edge_stays_protected(self):
        p=np.array([[x,0,-.2] for x in np.arange(-.2,.21,.05)])
        self.assertFalse(low_return_outliers(p,p[:,2],self.ground).any())
    def test_three_speckles_without_overlying_floor_stay_blocked(self):
        p=np.array([[.01,.01,-.2],[.06,.02,-.21],[.11,.03,-.19]])
        floor=self.ground[np.linalg.norm(self.ground[:,:2],axis=1)>.4]
        self.assertFalse(low_return_outliers(p,p[:,2],floor).any())
    def test_geometry_invariant_to_rotation_and_translation(self):
        p=np.array([[0,0],[.052,.022],[.098,.069]])
        for a in [0,.3,1.2,2.4]:
            c,s=np.cos(a),np.sin(a);v=p@np.array([[c,-s],[s,c]]).T+[.173,-.391]
            self.assertFalse(supported_low_geometry(v))
    def setUp(self):
        x,y=np.meshgrid(np.arange(-2.,2.01,.05),np.arange(-2.,2.01,.05))
        self.ground=np.column_stack([x.ravel(),y.ravel(),np.zeros(x.size)])
        self.bases=np.array([[x,0,.25] for x in np.arange(-1,1,.2)])
    def test_sparse_returns_below_observed_floor(self):
        p=np.array([[.02,.01,-.33],[.16,.01,-.34]])
        self.assertTrue(low_return_outliers(p,p[:,2],self.ground).all())
    def test_dense_duplicates_not_multiple_spatial_evidence(self):
        p=np.repeat([[.02,.01,-.3]],100,axis=0)
        self.assertTrue(low_return_outliers(p,p[:,2],self.ground).all())
    def test_coherent_drop_retained(self):
        x,y=np.meshgrid(np.arange(-.2,.21,.05),np.arange(-.2,.21,.05))
        p=np.column_stack([x.ravel(),y.ravel(),np.full(x.size,-.2)])
        self.assertFalse(low_return_outliers(p,p[:,2],self.ground).any())
    def test_no_floor_evidence_keeps_low_return(self):
        p=np.array([[0,0,-.2]])
        floor=self.ground[np.linalg.norm(self.ground[:,:2],axis=1)>.4]
        self.assertFalse(low_return_outliers(p,p[:,2],floor).any())
    def test_floor_on_one_side_is_not_support(self):
        p=np.array([[.02,0,-.2]])
        self.assertFalse(low_return_outliers(p,p[:,2],self.ground[self.ground[:,0]<0]).any())
    def test_raised_object_never_ignored(self):
        p=np.array([[0,0,.07],[0,0,.15],[0,0,.3]])
        self.assertFalse(low_return_outliers(p,p[:,2],self.ground).any())
    def test_static_map_corrects_false_occupied_cell(self):
        p=np.array([[.02,.01,-.33],[.16,.01,-.34],[.5,.5,.15]])
        m=StaticMap(np.vstack([self.ground,p]),self.bases)
        ids,_=m.cell_indices(p)
        np.testing.assert_array_equal(m.grid.ravel()[ids],[0,0,100])
        self.assertEqual(len(m.rejected_low_points),2)
    def test_unsupported_low_point_cannot_become_free(self):
        floor=self.ground[np.linalg.norm(self.ground[:,:2],axis=1)>.4]
        p=np.array([[0,0,-.2]])
        m=StaticMap(np.vstack([floor,p]),self.bases);ids,_=m.cell_indices(p)
        self.assertEqual(m.grid.ravel()[ids[0]],-1)
    def test_gentle_map_tilt_and_height_noise(self):
        rng=np.random.default_rng(5);p=self.ground.copy()
        p[:,2]=.006*p[:,0]+.012*p[:,1]+rng.normal(0,.006,len(p))
        m=StaticMap(p,self.bases)
        np.testing.assert_allclose(m.plane[:2],[.006,.012],atol=.001)

if __name__=='__main__':unittest.main()
