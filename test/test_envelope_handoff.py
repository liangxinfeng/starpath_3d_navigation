import unittest
import numpy as np
from nav_envelope import EnvelopeHandoff

class Tests(unittest.TestCase):
    def setUp(self):
        self.t=0.;self.h=EnvelopeHandoff(lambda:self.t)
        self.a=np.array([[0,0,0,1,.1,.1,.1]])
        self.b=np.array([[0,.3,0,1,.1,.1,.1]])
        self.c=np.array([[0,.6,0,1,.1,.1,.1]])
    def ready(self):
        self.h.update(self.a);self.h.acknowledge(self.h.sent)
    def test_no_initial_ack_is_not_ready(self):
        self.h.update(self.a);self.assertFalse(self.h.health()[0])
    def test_shape_change_keeps_ready_and_guards_both_shapes(self):
        self.ready();self.h.update(self.b)
        self.assertTrue(self.h.health()[0]);self.assertEqual(len(self.h.guard()),1)
        self.assertTrue(any(np.all(np.abs(self.a[0,1:4]-p[1:4])+self.a[0,4:]<=p[4:]+1e-9) for p in self.h.guard()))
        # Midway obstacle must be inside the swept collision guard.
        self.assertTrue(any(np.all(np.abs(np.array([.15,0,1])-p[1:4])<=p[4:]) for p in self.h.guard()))
    def test_continuous_motion_does_not_starve_ack(self):
        self.ready();self.h.update(self.b);message=self.h.sent.copy()
        self.h.update(self.c);self.assertEqual(self.h.sent,message)
        self.assertTrue(self.h.acknowledge(message));self.h.update(self.c)
        self.assertNotEqual(self.h.sent,message)
        self.assertTrue(any(np.all(np.abs(self.c[0,1:4]-p[1:4])+self.c[0,4:]<=p[4:]+1e-9) for p in self.h.guard()))
    def test_stale_ack_cannot_refresh_health(self):
        self.ready();old=self.h.sent.copy();self.h.update(self.b)
        self.t=.31;self.assertFalse(self.h.acknowledge(old));self.assertFalse(self.h.health()[0])
    def test_pending_shape_timeout_and_planner_loss(self):
        self.ready();self.t=.81;self.assertFalse(self.h.health()[0])
        self.t=0.;self.ready();self.h.update(self.b);self.t=.31;self.assertFalse(self.h.health()[0])
    def test_returning_shape_does_not_accumulate_history(self):
        self.ready()
        for p in [self.b,self.c,self.a]:self.h.update(p);self.h.acknowledge(self.h.sent)
        np.testing.assert_array_equal(self.h.guard(),self.a)
    def test_invalid_parts_rejected(self):
        for p in [[],[[0]*7],[[2,0,0,0,1,1,1]],[[0,0,0,float('nan'),1,1,1]]]:
            with self.assertRaises(ValueError):self.h.update(p)

if __name__=='__main__':unittest.main()
