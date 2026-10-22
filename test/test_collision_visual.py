import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from nav_collision_visual import collision_visual
from nav_body_model import primitive_distances
from nav_math import transform,Rotation

def test_visual_matches_exact_collision_union_in_moving_frame():
    rng=np.random.default_rng(345)
    parts=np.array([[0,0,0,0,.3,.2,.2],[1,.5,.4,.9,.15,0,0]])
    body=np.eye(4);body[:3,:3]=Rotation.from_euler('z',.7).as_matrix();body[:3,3]=[2,-3,.1]
    local=rng.uniform(-1,1,(10000,3));world=transform(local,body)
    expected=primitive_distances(local,parts)<=0
    d=collision_visual(world,body,parts)
    assert d['count']==int(expected.sum()) and d['display_count']==d['count']
    assert np.allclose(np.array(d['points']),world[expected])
    assert set(d['parts'])=={0,1}
    assert np.isclose(d['gap'],primitive_distances(local,parts).min())

def test_stale_and_empty_never_show_old_blocker():
    p=np.array([[0,0,0]])
    parts=np.array([[0,0,0,0,1,1,1]])
    assert collision_visual(p,np.eye(4),parts)['count']==1
    for points,fresh in [(p,False),(np.empty((0,3)),True)]:
        d=collision_visual(points,np.eye(4),parts,fresh)
        assert d['count']==0 and not d['points'] and d['xyz'] is None

def test_high_points_preserved_and_dense_cap_keeps_deepest():
    p=np.column_stack([np.linspace(0,.1,5000),np.zeros(5000),np.ones(5000)*1.5])
    parts=np.array([[1,0,0,1.5,.2,0,0]])
    d=collision_visual(p,np.eye(4),parts,limit=100)
    assert d['count']==5000 and d['display_count']<=101
    assert [0.,0.,1.5] in d['points'] and d['gap']==-.2
