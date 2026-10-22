import sys,unittest,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from nav_chassis import ChassisTrajectory,advance

class ChassisTests(unittest.TestCase):
    def test_arc_keeps_feedforward(self):
        end=advance([0,0,0],.12,.24,1.)
        t=ChassisTrajectory([[0,0,0],end],[.12],[.24],[1.],True,0)
        x,w,_,_=t.control(np.zeros(3),0,.02)
        self.assertAlmostEqual(x,.12);self.assertAlmostEqual(w,.24)
        self.assertGreater(end[1],0)
    def test_in_place(self):
        t=ChassisTrajectory([[0,0,0],[0,0,.5]],[0],[.25],[2.],True,0)
        x,w,_,_=t.control(np.zeros(3),0,.02)
        self.assertEqual(x,0);self.assertGreater(w,0)
    def test_reject_invalid_geometry_and_speed(self):
        for end,v in (([.2,0,0],.1),([1.,0,0],1.)):
            with self.assertRaises(ValueError):ChassisTrajectory([[0,0,0],end],[v],[0],[1],True,0)
    def test_feedback_closed_loop_arcs_and_turn(self):
        p=np.array([0.,0.,0.]);poses=[p.copy()]
        vs=[.12,0,.12];ws=[.20,-.24,0];durations=[4,2,4]
        for v,w,t in zip(vs,ws,durations):poses.append(advance(poses[-1],v,w,t))
        traj=ChassisTrajectory(poses,vs,ws,durations,True,0)
        progress=0.;done=False;actual_v=actual_w=0.
        for _ in range(1500):
            v,w,progress,done=traj.control(p,progress,.02)
            # Acceleration-limited chassis, not a perfect reference replay.
            actual_v+=np.clip(v-actual_v,-.006,.006)
            actual_w+=np.clip(w-actual_w,-.012,.012)
            p=advance(p,actual_v,actual_w,.02)
            if done:break
        self.assertTrue(done);self.assertLess(np.linalg.norm(p[:2]-poses[-1][:2]),.045)
    def test_large_tracking_error_stops(self):
        t=ChassisTrajectory([[0,0,0],[1,0,0]],[.1],[0],[10.],True,0)
        with self.assertRaises(ValueError):t.control(np.array([0,.2,0]),0,.02)
    def test_zero_length_goal(self):
        t=ChassisTrajectory([[1,2,.3]],[],[],[],True,0)
        self.assertTrue(t.control(np.array([1,2,.3]),0,.02)[3])

if __name__=='__main__':unittest.main()
