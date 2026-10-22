import sys,math,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from nav_footprint import chassis_footprint,intersecting_cells
from nav_grid import ObservedGrid

class FootprintTests(unittest.TestCase):
    def grid(self,points=(),unknown=False):
        a=np.zeros((200,200),np.int8)
        for x,y in points:a[int((y+2)/.02),int((x+2)/.02)]=-1 if unknown else 100
        return ObservedGrid(a,200,200,.02,[-2,-2])
    def test_factory_dimensions(self):
        f=chassis_footprint()
        np.testing.assert_allclose(f.min(0),[-.26756573,-.17791685],atol=.00001)
        np.testing.assert_allclose(f.max(0),[.30326781,.17791685],atol=.00001)
    def test_previous_false_block_outside_model(self):
        self.assertEqual(self.grid([[-.338,-.217]]).pose_reason(0,0,0),'')
    def test_interior_and_edges_remain_blocked(self):
        for point in [[0,0],[.28,0],[-.25,0],[0,.16],[0,-.16]]:
            self.assertIn('障碍',self.grid([point]).pose_reason(0,0,0))
    def test_rotated_outline(self):
        self.assertIn('障碍',self.grid([[0,.28]]).pose_reason(0,0,math.pi/2))
        self.assertEqual(self.grid([[.28,0]]).pose_reason(0,0,math.pi/2),'')
    def test_unknown_not_cleared(self):
        self.assertIn('未知',self.grid([[0,0]],True).pose_reason(0,0,0))
    def test_out_of_map(self):
        self.assertIn('超出',self.grid().pose_reason(1.9,0,0))
    def test_rounded_corner_not_rectangle(self):
        self.assertEqual(self.grid([[.29,.17]]).pose_reason(0,0,0),'')
    def test_square_edge_touch(self):
        poly=np.array([[-.1,-.1],[.1,-.1],[.1,.1],[-.1,.1]])
        actual=intersecting_cells(np.array([[0,0],[.12,0],[.14,0],[.12,.12]]),poly,.04)
        np.testing.assert_array_equal(actual,[True,True,False,True])
    def test_rotated_grid_frame(self):
        # Same free grid in a rotated map frame must preserve physical overlap.
        a=np.zeros((200,200),np.int8);a[100,100]=100
        g=ObservedGrid(a,200,200,.02,[2,-2],angle=math.pi/2)
        self.assertIn('障碍',g.pose_reason(0,0,.7))

if __name__=='__main__':unittest.main()
