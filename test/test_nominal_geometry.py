import unittest
import numpy as np
from test_body_model import bare
from nav_body_model import primitive_distances
from nav_envelope import EnvelopeHandoff

class NominalGeometryTests(unittest.TestCase):
    def model(self):
        m=bare();m.q={};m.joints={};m.grippers={};m.nominal_geometry=True
        m.shapes={'Link_Zero_Point':(None,np.array([-.301,-.251,-.021]),np.array([.341,.251,.331]))}
        return m
    def test_chassis_has_only_urdf_extents_no_rounding_or_extra_box(self):
        m=self.model();m.update();self.assertEqual(len(m.primitives),1)
        p=m.primitives[0]
        np.testing.assert_allclose(p[1:4]-p[4:],[-.301,-.251,-.021])
        np.testing.assert_allclose(p[1:4]+p[4:],[.341,.251,.331])
        self.assertGreater(m.clearance(np.array([[0,.252,.2]]),np.eye(4)),0)
        self.assertLess(m.clearance(np.array([[0,.250,.2]]),np.eye(4)),0)
    def test_bounds_follow_actual_mesh_vertices_and_shrink(self):
        m=self.model();m.mesh_hulls={'Link_Zero_Point':np.array([[0,0,0],[.1,.2,.3],[-.1,-.2,-.3]])}
        m.update();np.testing.assert_allclose(m.primitives[0,4:],[.1,.2,.3])
        m.mesh_hulls['Link_Zero_Point']*=.5;m.update()
        np.testing.assert_allclose(m.primitives[0,4:],[.05,.1,.15])
    def test_gripper_no_added_padding(self):
        m=self.model();m.grippers={'Link_Zero_Point':.18123};m.update()
        np.testing.assert_allclose(m.primitives[-1,4:],[.18123]*3)
    def test_zero_padding_handoff_and_settling(self):
        h=EnvelopeHandoff(lambda:0,padding=0)
        a=np.array([[0,0,0,1,.1,.1,.1]],float);b=a.copy();b[0,1]=.02
        h.update(a);h.acknowledge(h.sent);h.update(b)
        p=h.guard()[0]
        np.testing.assert_allclose(p[1:4]-p[4:],[-.1,-.1,.9])
        np.testing.assert_allclose(p[1:4]+p[4:],[.12,.1,1.1])
        h.acknowledge(h.sent);np.testing.assert_array_equal(h.guard(),b)
    def test_sphere_jitter_does_not_create_false_corner_collision(self):
        h=EnvelopeHandoff(lambda:0,padding=0)
        a=np.array([[1,0,0,1,.1,.1,.1]],float);b=a.copy();b[0,1]=.001
        h.update(a);h.acknowledge(h.sent);h.update(b);p=h.guard()
        self.assertEqual(p[0,0],1)
        self.assertAlmostEqual(p[0,4],.101)
        self.assertGreater(primitive_distances(np.array([[.09,.09,1]]),p)[0],0)
        self.assertLessEqual(primitive_distances(np.array([[-.1,0,1],[.101,0,1]]),p).max(),1e-12)
        h.acknowledge(h.sent);np.testing.assert_array_equal(h.guard(),b)

if __name__=='__main__':unittest.main()
