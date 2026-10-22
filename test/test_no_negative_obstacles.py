import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from nav_static_map import StaticMap,LiveLayer

def scene():
    x,y=np.meshgrid(np.arange(-2,2.01,.05),np.arange(-2,2.01,.05))
    floor=np.column_stack([x.ravel(),y.ravel(),np.zeros(x.size)])
    bases=np.array([[x,0,.25] for x in np.arange(-1,1,.2)])
    return floor,bases

def test_dense_negative_patch_excluded_without_removing_raised_obstacles():
    floor,bases=scene()
    x,y=np.meshgrid(np.arange(-.2,.21,.05),np.arange(-.2,.21,.05))
    negative=np.column_stack([x.ravel(),y.ravel(),np.full(x.size,-.2)])
    positive=np.array([[.7,.7,.07],[.8,.8,.15],[.9,.9,.3]])
    m=StaticMap(np.vstack([floor,negative,positive]),bases)
    np.testing.assert_allclose(m.obstacles,positive)
    assert not m.classify(negative)[1].any()
    assert m.classify(positive)[1].all()
    ids,valid=m.cell_indices(negative);assert valid.all()
    assert (m.grid.ravel()[ids]==0).all()
    live=LiveLayer(m);live.update(negative)
    assert not live.hits
    live.update(positive);assert len(live.hits)==3

def test_negative_returns_do_not_fill_unobserved_hole():
    floor,bases=scene()
    floor=floor[np.linalg.norm(floor[:,:2],axis=1)>.5]
    negative=np.array([[0.,0.,-.2]])
    m=StaticMap(np.vstack([floor,negative]),bases)
    ids,valid=m.cell_indices(negative)
    assert valid.all() and m.grid.ravel()[ids[0]]==-1
    assert len(m.obstacles)==0

def test_recorded_point_excluded_using_its_height_relative_to_tilted_floor():
    floor,bases=scene()
    plane=np.array([.006596162837,.012510463801,-.155349082432])
    floor[:,2]=floor[:,:2]@plane[:2]+plane[2]
    bases[:,2]+=plane[2]
    low=np.array([[-.394997686,-.934537232,-.276105970]])
    raised=low.copy();raised[:,2]=raised[:,:2]@plane[:2]+plane[2]+.1
    m=StaticMap(np.vstack([floor,low,raised]),bases)
    assert m.heights(low)[0]<-.1
    assert not m.classify(low)[1].any()
    assert m.classify(raised)[1].all()
