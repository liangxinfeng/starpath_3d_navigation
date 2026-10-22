import time,unittest
from unittest.mock import patch
from test_body_model import bare,feedback

class Tests(unittest.TestCase):
    def test_dynamic_motion_does_not_require_reference_pose(self):
        m=bare();m.dynamic_geometry=True;m.pose_guard=lambda _:(True,'dynamic geometry valid')
        with patch('nav_body_model.time.time',return_value=100.):
            self.assertTrue(m.feedback(feedback(90),1,source_time=99.99))
        m.profile=dict(q=m.q.copy(),gripper=m.gripper_q.copy())
        d=feedback(90.01);d['right_arm_joint_state']['position'][4]=30.;d['right_arm_joint_state']['speed'][4]=15.
        with patch('nav_body_model.time.time',return_value=100.01):
            self.assertTrue(m.feedback(d,1.01,source_time=100.))
        self.assertTrue(m.health(1.01)[0]);self.assertTrue(m.profile_health()[0])
    def test_queued_message_does_not_become_fresh_pose(self):
        m=bare()
        with patch('nav_body_model.time.time',return_value=100.):
            self.assertFalse(m.feedback(feedback(90),1,source_time=99.8))
        self.assertFalse(m.health(1)[0])
    def test_source_clock_reversal_is_rejected(self):
        m=bare()
        with patch('nav_body_model.time.time',return_value=100.):
            self.assertTrue(m.feedback(feedback(90),1,source_time=99.99))
            self.assertFalse(m.feedback(feedback(90.01),1.01,source_time=99.98))
    def test_dynamic_mode_requires_source_time_and_valid_communication(self):
        m=bare();m.dynamic_geometry=True;m.pose_guard=lambda _:(True,'ok')
        m.feedback(feedback(90),1);self.assertFalse(m.health(1)[0])
        d=feedback(91);d['right_arm_joint_state']['communication_lost'][0]=True
        self.assertFalse(m.feedback(d,2,source_time=time.time()))

if __name__=='__main__':unittest.main()
