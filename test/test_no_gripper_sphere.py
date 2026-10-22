import numpy as np
import test_nominal_geometry as nominal
from nav_envelope import EnvelopeHandoff

def test_disabling_estimates_preserves_every_mesh_box():
    m=nominal.NominalGeometryTests().model();m.grippers={'Link_Zero_Point':.8}
    m.update();before=m.primitives.copy();names=list(m.primitive_names)
    m.estimated_grippers=False;m.update()
    np.testing.assert_array_equal(m.primitives,before[before[:,0]==0])
    assert m.primitive_names==[n for p,n in zip(before,names) if p[0]==0]
    assert np.all(m.primitives[:,0]==0)
    assert m.clearance(np.array([[.7,0,0]]),np.eye(4))>0
    assert m.clearance(np.array([[0,0,.1]]),np.eye(4))<0
    assert m.radius<.8

def test_updated_model_never_reintroduces_spheres_in_handoff():
    m=nominal.NominalGeometryTests().model();m.grippers={'Link_Zero_Point':.8};m.estimated_grippers=False
    h=EnvelopeHandoff(lambda:0,padding=0)
    for z in [0.,.02,-.02]:
        m.mesh_hulls={'Link_Zero_Point':np.array([[-.1,-.2,z-.1],[.1,.2,z+.1]])}
        m.update();h.update(m.primitives);h.acknowledge(h.sent)
        assert len(h.guard())==1 and np.all(h.guard()[:,0]==0)
        np.testing.assert_array_equal(h.guard(),m.primitives)
