"""Planar holonomic command integration and independent command validation."""
import math
import numpy as np
from nav_math import wrap

def advance_omni(p,vx,vy,w,t):
    x,y,a=map(float,p)
    if abs(w)<1e-8:
        x+=(math.cos(a)*vx-math.sin(a)*vy)*t
        y+=(math.sin(a)*vx+math.cos(a)*vy)*t
    else:
        b=a+w*t
        x+=(vx*(math.sin(b)-math.sin(a))+vy*(math.cos(b)-math.cos(a)))/w
        y+=(vx*(math.cos(a)-math.cos(b))+vy*(math.sin(b)-math.sin(a)))/w
    return np.array([x,y,wrap(a+w*t)])

def valid_command(v,linear=.15,angular=.3):
    return (len(v)==3 and all(math.isfinite(x) for x in v) and
            -.4*linear-.001<=v[0]<=linear+.001 and abs(v[1])<=min(.12,.5*linear)+.001 and abs(v[2])<=angular+.001)

def bounded_twist(v,linear):
    # Preserve the complete holonomic screw motion when limiting vector speed.
    # The runtime checks the resulting actual command against grid + 3-D geometry.
    scale=min(1.,linear/max(math.hypot(v[0],v[1]),1e-9))
    return tuple(float(x)*scale for x in v)

def planar(t):
    a=math.atan2(t[1,0],t[0,0]);out=np.eye(4)
    out[:2,:2]=[[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]];out[:2,3]=t[:2,3]
    return out
