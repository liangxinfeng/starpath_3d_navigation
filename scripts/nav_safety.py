"""Explicit arm, bounded commands and fail-closed freshness checks."""
import math
class Safety:
    def __init__(self,clock):
        self.clock=clock;self.ui=-1e9;self.odom=-1e9;self.cloud=-1e9;self.local=-1e9;self.planner=-1e9
        self.locked=False;self.reason='等待三维定位';self.armed=False;self.fault=''
    def health(self):
        now=self.clock()
        for condition,text in [(self.locked,'三维定位未锁定'),(now-self.local<1.2,'全局定位过期'),(now-self.odom<.5,'里程计中断'),(now-self.cloud<.6,'实时点云中断'),(now-self.planner<.5,'规划器响应超时')]:
            if not condition:return False,text
        return True,'定位与实时点云正常'
    def arm(self):
        ok,why=self.health()
        if not ok or self.clock()-self.ui>.3:raise ValueError(why if not ok else '界面心跳未就绪')
        self.armed=True;self.fault=''
    def stop(self,why):self.armed=False;self.reason=why
    def check(self):
        ok,why=self.health()
        if self.armed and (not ok or self.clock()-self.ui>.3):
            self.fault=why if not ok else '界面心跳中断';self.stop(self.fault)
        return self.armed
    @staticmethod
    def bounded(x,yaw,linear=.15,angular=.3):
        if not math.isfinite(x) or not math.isfinite(yaw):raise ValueError('nonfinite command')
        return max(0.,min(linear,x)),max(-angular,min(angular,yaw))
