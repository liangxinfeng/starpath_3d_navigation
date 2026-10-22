"""Ground removal and nonpersistent live occupancy regression tests."""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from nav_ground import separate_ground
from nav_static_map import StaticMap,LiveLayer


class GroundAndLiveTests(unittest.TestCase):
    def setUp(self):
        x,y=np.meshgrid(np.arange(-2,2.01,.1),np.arange(-2,2.01,.1))
        self.floor=np.column_stack([x.ravel(),y.ravel(),np.zeros(x.size)])

    def test_noisy_sloped_floor_and_low_obstacle(self):
        rng=np.random.default_rng(7);p=self.floor.copy()
        p[:,2]=.012*p[:,0]+.008*p[:,1]+.02+rng.normal(0,.006,len(p))
        obstacle=np.array([[.3,.3,.11],[.4,.3,.11],[.5,.3,.15],[.3,.3,.7]])
        mask,h,d=separate_ground(np.vstack([p,obstacle]),[0,0,0],[0,0,0])
        self.assertEqual(d['mode'],'current_supported_plane')
        self.assertGreater(mask[:len(p)].mean(),.99)
        self.assertFalse(mask[len(p):].any())

    def test_dense_raised_surface_cannot_replace_floor(self):
        raised=self.floor.copy();raised[:,2]=.09
        mask,h,d=separate_ground(raised,[0,0,0],[0,0,0])
        self.assertEqual(d['mode'],'saved_plane');self.assertFalse(mask.any())

    def test_limited_spatial_support_cannot_tilt_plane(self):
        p=self.floor[np.abs(self.floor[:,1])<.01].copy();p[:,2]=.02
        _,_,d=separate_ground(p,[0,0,0],[0,0,0])
        self.assertEqual(d['mode'],'saved_plane')

    def test_static_unknown_and_latest_scan(self):
        wall=np.array([[1.,y,.2] for y in np.arange(-1,1,.1)])
        bases=np.array([[x,0,.25] for x in np.arange(-1,1,.2)])
        static=StaticMap(np.vstack([self.floor,wall]),bases);live=LiveLayer(static)
        baseline=static.grid.copy();p=np.array([[0.,0.,.15]])
        ids,_=static.cell_indices(p)
        live.update(p,preclassified=True);self.assertEqual(live.combined().ravel()[ids[0]],100)
        live.update(np.array([[.5,.5,.15]]),preclassified=True)
        self.assertEqual(live.combined().ravel()[ids[0]],baseline.ravel()[ids[0]])
        self.assertEqual(live.removed_previous_cells,1)
        live.update(np.empty((0,3)),preclassified=True)
        np.testing.assert_array_equal(live.combined(),baseline)
        self.assertTrue((baseline==100).any());self.assertTrue((baseline==-1).any())


if __name__=='__main__':unittest.main()
