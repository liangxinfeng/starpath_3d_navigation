"""Obstacle recovery keeps the goal, but never authorizes hardware itself."""
class ObstacleWait:
    def __init__(self):self.clear()
    def clear(self):
        self.active=False;self.clear_scans=0;self.last_scan=-1;self.last_update=-1e9
    def begin(self,scan):
        if not self.active:
            self.active=True;self.clear_scans=0;self.last_scan=scan;self.last_update=-1e9
    def observe(self,scan,unblocked,now):
        if not self.active or scan==self.last_scan:return
        if self.last_scan>=scan or now-self.last_update>.6:self.clear_scans=0
        self.last_scan=scan;self.last_update=now
        self.clear_scans=self.clear_scans+1 if unblocked else 0
    def ready(self,now):
        return self.active and self.clear_scans>=3 and now-self.last_update<.6
