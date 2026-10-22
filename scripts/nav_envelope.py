"""Collision-envelope handoff without cancelling a goal on every pose update.

Execution checks the union of current and last acknowledged geometry while a
new shape is in flight. Stale/missing ACKs still fail; old ACKs cannot refresh
the deadline. No navigation commands are produced by this module.
"""
import numpy as np


class EnvelopeHandoff:
    def __init__(self,clock,padding=.01):
        self.padding=float(padding)
        if not 0<=self.padding<=.05:raise ValueError('无效轮廓交接余量')
        self.clock=clock;self.version=0;self.current=np.empty((0,7))
        self.ack_parts=np.empty((0,7));self.sent=[];self.sent_at=None
        self.acked=[];self.ack_at=None

    @staticmethod
    def validate(parts):
        a=np.asarray(parts,dtype=float)
        if a.ndim!=2 or a.shape[1]!=7 or not len(a) or not np.isfinite(a).all():raise ValueError('无效碰撞轮廓')
        if not np.isin(a[:,0],[0,1]).all() or np.any(a[:,4:]<=0):raise ValueError('无效碰撞部件尺寸')
        return a.copy()

    def update(self,parts):
        self.current=self.validate(parts)
        # Keep one in-flight message. Continuous joint updates must not make
        # every ACK obsolete before it arrives. Latest shape remains in guard().
        if self.sent and self.sent!=self.acked:return
        if self.sent and np.array_equal(self.current,self.ack_parts):return
        self.version+=1;self.sent=[float(self.version)]+self.current.ravel().tolist();self.sent_at=self.clock()

    def acknowledge(self,payload):
        if not self.sent or list(payload)!=self.sent:return False
        self.acked=self.sent.copy();self.ack_parts=np.asarray(payload[1:]).reshape(-1,7).copy();self.ack_at=self.clock()
        return True

    def guard(self):
        if not len(self.ack_parts):return self.current.copy()
        pending=np.asarray(self.sent[1:]).reshape(-1,7)
        shapes=[self.current,self.ack_parts,pending]
        # Collision checking must cover the intervening motion too. Build a
        # hull per corresponding link, never one box around the whole robot.
        # Keep pending geometry even if the arm returned before its ACK.
        if all(a.shape==self.current.shape for a in shapes):
            a=np.stack(shapes);changed=np.any(a!=a[0:1],axis=(0,2));result=self.current.copy()
            low=np.min(a[:,:,1:4]-a[:,:,4:],axis=0)-self.padding
            high=np.max(a[:,:,1:4]+a[:,:,4:],axis=0)+self.padding
            result[changed,0]=0;result[changed,1:4]=((low+high)/2)[changed];result[changed,4:]=((high-low)/2)[changed]
            # Never turn jittering spheres into cubes. A sphere centred on the
            # current wrist covers old/pending spheres and their translation.
            for i in np.flatnonzero(changed & np.all(a[:,:,0]==1,axis=0)):
                result[i,0]=1;result[i,1:4]=self.current[i,1:4]
                radius=np.max(np.linalg.norm(a[:,i,1:4]-result[i,1:4],axis=1)+a[:,i,4])+self.padding
                result[i,4:]=radius
            return result
        return np.unique(np.vstack(shapes),axis=0)

    def health(self):
        now=self.clock()
        if not self.acked:return False,'等待规划器应用整机碰撞外形'
        if self.current.shape!=self.ack_parts.shape:return False,'碰撞部件集合变化，等待规划器重新确认'
        if now-self.ack_at>.8:return False,'规划器碰撞轮廓确认中断'
        if self.sent!=self.acked and now-self.sent_at>.3:return False,'新碰撞轮廓应用超时'
        return True,'实时碰撞轮廓有效'

