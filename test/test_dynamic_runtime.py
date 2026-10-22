import time,unittest
from types import SimpleNamespace as S
import numpy as np
from nav_runtime import Runtime
from nav_envelope import EnvelopeHandoff

class Tests(unittest.TestCase):
    def test_runtime_shape_update_preserves_existing_goal(self):
        a=np.array([[0,0,0,1,.1,.1,.1]]);b=np.array([[0,.2,0,1,.1,.1,.1]])
        h=EnvelopeHandoff(time.monotonic);h.update(a);h.acknowledge(h.sent)
        model=S(q={},gripper_q={},health=lambda:(True,'ok'))
        r=S(dynamic_geometry=True,sector=None,model=model,handoff=h,residual=S(prepare=lambda *x:None,collision_parts=lambda:b),goal={'execute':False},
            last_tick=time.monotonic(),backend_time=time.monotonic(),pending_goal=0,trajectory=object(),
            safety=S(armed=False,check=lambda:None,fault=''),ready_pub=S(publish=lambda x:None),freeze_pub=S(publish=lambda x:None),
            value=lambda key:.15 if key=='max_speed' else .3,send_drive=lambda *args:None)
        r.update_dynamic_envelope=lambda:Runtime.update_dynamic_envelope(r)
        r.health=h.health
        def unexpected_abort(*args):raise AssertionError('Geometry update cancelled the goal')
        r.abort=unexpected_abort
        Runtime.tick(r)
        self.assertIsNotNone(r.goal);self.assertTrue(h.health()[0]);self.assertNotEqual(h.sent,h.acked)
        Runtime.envelope_applied(r,S(data=h.sent));self.assertTrue(h.health()[0])

if __name__=='__main__':unittest.main()
