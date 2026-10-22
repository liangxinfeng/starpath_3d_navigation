import unittest,time
from unittest.mock import patch
from types import SimpleNamespace as S
from collections import deque
import numpy as np
from nav_runtime import Runtime

class ObstacleReplanTests(unittest.TestCase):
    def fixture(self):
        log=[];goal=dict(point=[1,2,0],yaw=.5,execute=True)
        r=S(goal=goal,safety=S(armed=True),obstacle_replans=0,scans=51,send_drive=lambda *a:log.append(('drive',a)),get_logger=lambda:S(info=lambda m:log.append(('info',m))),trajectory=object(),tracking_frame=object(),reset_pub=S(publish=lambda m:log.append(('reset',m.data))))
        r.next_segment=lambda now,reason:Runtime.next_segment(r,now,reason)
        r.path_pub=S(publish=lambda m:log.append(('path_clear',len(m.poses))))
        def abort(reason):r.goal=None;r.safety.armed=False;r.trajectory=None;log.append(('abort',reason))
        r.abort=abort
        return r,log,goal
    def test_stop_preserve_goal_and_wait_for_new_scan(self):
        r,log,goal=self.fixture()
        with patch('nav_runtime.time.monotonic',return_value=10.):self.assertTrue(Runtime.queue_obstacle_replan(r,'obstacle'))
        self.assertEqual(log[0],('drive',(0.,0.,False)));self.assertIs(r.goal,goal)
        self.assertIsNone(r.trajectory);self.assertTrue(r.waiting_for_stop);self.assertEqual(r.retry_scan_index,51)
        self.assertGreater(r.pending_goal,10.);self.assertEqual(r.obstacle_replans,1)
    def test_persistent_obstacle_has_bounded_retries(self):
        r,log,_=self.fixture()
        for _ in range(4):self.assertTrue(Runtime.queue_obstacle_replan(r,'obstacle'))
        self.assertEqual(r.obstacle_replans,3);self.assertIsNone(r.goal);self.assertFalse(r.safety.armed)
        self.assertIn('3次',log[-1][1])
    def test_preview_never_auto_arms(self):
        r,log,_=self.fixture();r.goal['execute']=False;r.safety.armed=False
        self.assertFalse(Runtime.queue_obstacle_replan(r,'obstacle'));self.assertEqual(log,[])
    def test_stationarity_requires_fresh_window_and_no_translation(self):
        r=S(motion_history=deque((t,np.eye(4)) for t in [9.65,9.75,9.85,9.95]))
        self.assertTrue(Runtime.settled_for_replan(r,10.))
        self.assertFalse(Runtime.settled_for_replan(r,10.5))
        r.motion_history[0][1][0,3]=.02
        self.assertFalse(Runtime.settled_for_replan(r,10.))
    def test_turning_chassis_not_considered_stopped(self):
        from scipy.spatial.transform import Rotation
        r=S(motion_history=deque((t,np.eye(4)) for t in [9.65,9.75,9.85,9.95]))
        r.motion_history[0][1][:3,:3]=Rotation.from_euler('z',.05).as_matrix()
        self.assertFalse(Runtime.settled_for_replan(r,10.))

if __name__=='__main__':unittest.main()
